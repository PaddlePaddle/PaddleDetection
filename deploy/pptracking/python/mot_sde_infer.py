# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import yaml
import cv2
import re
import numpy as np
from collections import defaultdict

import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

from picodet_postprocess import PicoDetPostProcess
from utils import argsparser, Timer, get_current_memory_mb, _is_valid_video, video2frames
from det_infer import Detector, DetectorPicoDet, get_test_images, print_arguments, PredictConfig
from det_infer import load_predictor
from benchmark_utils import PaddleInferBenchmark
from visualize import plot_tracking

from mot.tracker import DeepSORTTracker
from mot.utils import MOTTimer, write_mot_results, flow_statistic, scale_coords, clip_box, preprocess_reid

from mot.mtmct.utils import parse_bias
from mot.mtmct.postprocess import trajectory_fusion, sub_cluster, gen_res, print_mtmct_result
from mot.mtmct.postprocess import get_mtmct_matching_results, save_mtmct_crops, save_mtmct_vis_results

# Global dictionary
MOT_SUPPORT_MODELS = {'DeepSORT'}


def bench_log(detector, img_list, model_info, batch_size=1, name=None):
    mems = {
        'cpu_rss_mb': detector.cpu_mem / len(img_list),
        'gpu_rss_mb': detector.gpu_mem / len(img_list),
        'gpu_util': detector.gpu_util * 100 / len(img_list)
    }
    perf_info = detector.det_times.report(average=True)
    data_info = {
        'batch_size': batch_size,
        'shape': "dynamic_shape",
        'data_num': perf_info['img_num']
    }
    log = PaddleInferBenchmark(detector.config, model_info, data_info,
                               perf_info, mems)
    log(name)


class SDE_Detector(Detector):
    """
    Detector of SDE methods

    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of per batch in inference, default is 1 in tracking models
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
    """

    def __init__(self,
                 pred_config,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1088,
                 trt_opt_shape=608,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False):
        super(SDE_Detector, self).__init__(
            pred_config=pred_config,
            model_dir=model_dir,
            device=device,
            run_mode=run_mode,
            batch_size=batch_size,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn)
        assert batch_size == 1, "The detector of tracking models only supports batch_size=1 now"
        self.pred_config = pred_config

    def postprocess(self,
                    boxes,
                    ori_image_shape,
                    threshold,
                    inputs,
                    scaled=False):
        over_thres_idx = np.nonzero(boxes[:, 1:2] >= threshold)[0]
        if len(over_thres_idx) == 0:
            pred_dets = np.zeros((1, 6), dtype=np.float32)
            pred_xyxys = np.zeros((1, 4), dtype=np.float32)
            return pred_dets, pred_xyxys
        else:
            boxes = boxes[over_thres_idx]

        if not scaled:
            # scaled means whether the coords after detector outputs
            # have been scaled back to the original image, set True 
            # in general detector, set False in JDE YOLOv3.
            input_shape = inputs['image'].shape[2:]
            im_shape = inputs['im_shape'][0]
            scale_factor = inputs['scale_factor'][0]
            pred_bboxes = scale_coords(boxes[:, 2:], input_shape, im_shape,
                                       scale_factor)
        else:
            pred_bboxes = boxes[:, 2:]

        pred_xyxys, keep_idx = clip_box(pred_bboxes, ori_image_shape)

        if len(keep_idx[0]) == 0:
            pred_dets = np.zeros((1, 6), dtype=np.float32)
            pred_xyxys = np.zeros((1, 4), dtype=np.float32)
            return pred_dets, pred_xyxys

        pred_scores = boxes[:, 1:2][keep_idx[0]]
        pred_cls_ids = boxes[:, 0:1][keep_idx[0]]
        pred_tlwhs = np.concatenate(
            (pred_xyxys[:, 0:2], pred_xyxys[:, 2:4] - pred_xyxys[:, 0:2] + 1),
            axis=1)

        pred_dets = np.concatenate(
            (pred_tlwhs, pred_scores, pred_cls_ids), axis=1)

        return pred_dets, pred_xyxys

    def predict(self,
                image_path,
                ori_image_shape,
                threshold=0.5,
                scaled=False,
                repeats=1,
                add_timer=True):
        '''
        Args:
            image_path (list[str]): path of images, only support one image path
                (batch_size=1) in tracking model
            ori_image_shape (list[int]: original image shape
            threshold (float): threshold of predicted box' score
            scaled (bool): whether the coords after detector outputs are scaled,
                default False in jde yolov3, set True in general detector.
            repeats (int): repeat number for prediction
            add_timer (bool): whether add timer during prediction
           
        Returns:
            pred_dets (np.ndarray, [N, 6]): 'x,y,w,h,score,cls_id'
            pred_xyxys (np.ndarray, [N, 4]): 'x1,y1,x2,y2'
        '''
        # preprocess
        if add_timer:
            self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(image_path)

        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])
        if add_timer:
            self.det_times.preprocess_time_s.end()
            self.det_times.inference_time_s.start()

        # model prediction
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            boxes = boxes_tensor.copy_to_cpu()
        if add_timer:
            self.det_times.inference_time_s.end(repeats=repeats)
            self.det_times.postprocess_time_s.start()

        # postprocess
        if len(boxes) == 0:
            pred_dets = np.zeros((1, 6), dtype=np.float32)
            pred_xyxys = np.zeros((1, 4), dtype=np.float32)
        else:
            pred_dets, pred_xyxys = self.postprocess(
                boxes, ori_image_shape, threshold, inputs, scaled=scaled)
        if add_timer:
            self.det_times.postprocess_time_s.end()
            self.det_times.img_num += 1
        return pred_dets, pred_xyxys


class SDE_DetectorPicoDet(DetectorPicoDet):
    """
    PicoDet of SDE methods, the postprocess of PicoDet has not been exported as
        other detectors, so do postprocess here.

    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of per batch in inference, default is 1 in tracking models
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
    """

    def __init__(self,
                 pred_config,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1088,
                 trt_opt_shape=608,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False):
        super(SDE_DetectorPicoDet, self).__init__(
            pred_config=pred_config,
            model_dir=model_dir,
            device=device,
            run_mode=run_mode,
            batch_size=batch_size,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn)
        assert batch_size == 1, "The detector of tracking models only supports batch_size=1 now"
        self.pred_config = pred_config

    def postprocess(self, boxes, ori_image_shape, threshold):
        over_thres_idx = np.nonzero(boxes[:, 1:2] >= threshold)[0]
        if len(over_thres_idx) == 0:
            pred_dets = np.zeros((1, 6), dtype=np.float32)
            pred_xyxys = np.zeros((1, 4), dtype=np.float32)
            return pred_dets, pred_xyxys
        else:
            boxes = boxes[over_thres_idx]

        pred_bboxes = boxes[:, 2:]

        pred_xyxys, keep_idx = clip_box(pred_bboxes, ori_image_shape)
        if len(keep_idx[0]) == 0:
            pred_dets = np.zeros((1, 6), dtype=np.float32)
            pred_xyxys = np.zeros((1, 4), dtype=np.float32)
            return pred_dets, pred_xyxys

        pred_scores = boxes[:, 1:2][keep_idx[0]]
        pred_cls_ids = boxes[:, 0:1][keep_idx[0]]
        pred_tlwhs = np.concatenate(
            (pred_xyxys[:, 0:2], pred_xyxys[:, 2:4] - pred_xyxys[:, 0:2] + 1),
            axis=1)

        pred_dets = np.concatenate(
            (pred_tlwhs, pred_scores, pred_cls_ids), axis=1)

        return pred_dets, pred_xyxys

    def predict(self,
                image_path,
                ori_image_shape,
                threshold=0.5,
                scaled=False,
                repeats=1,
                add_timer=True):
        '''
        Args:
            image_path (list[str]): path of images, only support one image path
                (batch_size=1) in tracking model
            ori_image_shape (list[int]: original image shape
            threshold (float): threshold of predicted box' score
            scaled (bool): whether the coords after detector outputs are scaled,
                default False in jde yolov3, set True in general detector.
            repeats (int): repeat number for prediction
            add_timer (bool): whether add timer during prediction
        Returns:
            pred_dets (np.ndarray, [N, 6]): 'x,y,w,h,score,cls_id'
            pred_xyxys (np.ndarray, [N, 4]): 'x1,y1,x2,y2'
        '''
        # preprocess
        if add_timer:
            self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(image_path)

        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])
        if add_timer:
            self.det_times.preprocess_time_s.end()
            self.det_times.inference_time_s.start()

        np_score_list, np_boxes_list = [], []

        # model prediction
        for i in range(repeats):
            self.predictor.run()
            np_score_list.clear()
            np_boxes_list.clear()
            output_names = self.predictor.get_output_names()
            num_outs = int(len(output_names) / 2)
            for out_idx in range(num_outs):
                np_score_list.append(
                    self.predictor.get_output_handle(output_names[out_idx])
                    .copy_to_cpu())
                np_boxes_list.append(
                    self.predictor.get_output_handle(output_names[
                        out_idx + num_outs]).copy_to_cpu())
        if add_timer:
            self.det_times.inference_time_s.end(repeats=repeats)
            self.det_times.postprocess_time_s.start()

        # postprocess
        self.picodet_postprocess = PicoDetPostProcess(
            inputs['image'].shape[2:],
            inputs['im_shape'],
            inputs['scale_factor'],
            strides=self.pred_config.fpn_stride,
            nms_threshold=self.pred_config.nms['nms_threshold'])
        boxes, boxes_num = self.picodet_postprocess(np_score_list,
                                                    np_boxes_list)

        if len(boxes) == 0:
            pred_dets = np.zeros((1, 6), dtype=np.float32)
            pred_xyxys = np.zeros((1, 4), dtype=np.float32)
        else:
            pred_dets, pred_xyxys = self.postprocess(boxes, ori_image_shape,
                                                     threshold)
        if add_timer:
            self.det_times.postprocess_time_s.end()
            self.det_times.img_num += 1

        return pred_dets, pred_xyxys


class SDE_ReID(object):
    """
    ReID of SDE methods

    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of per batch in inference, default 50 means at most
            50 sub images can be made a batch and send into ReID model
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
    """

    def __init__(self,
                 pred_config,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=50,
                 trt_min_shape=1,
                 trt_max_shape=1088,
                 trt_opt_shape=608,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(
            model_dir,
            run_mode=run_mode,
            batch_size=batch_size,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            device=device,
            use_dynamic_shape=self.pred_config.use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn)
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.batch_size = batch_size
        assert pred_config.tracker, "Tracking model should have tracker"
        pt = pred_config.tracker
        max_age = pt['max_age'] if 'max_age' in pt else 30
        max_iou_distance = pt[
            'max_iou_distance'] if 'max_iou_distance' in pt else 0.7
        self.tracker = DeepSORTTracker(
            max_age=max_age, max_iou_distance=max_iou_distance)

    def get_crops(self, xyxy, ori_img):
        w, h = self.tracker.input_size
        self.det_times.preprocess_time_s.start()
        crops = []
        xyxy = xyxy.astype(np.int64)
        ori_img = ori_img.transpose(1, 0, 2)  # [h,w,3]->[w,h,3]
        for i, bbox in enumerate(xyxy):
            crop = ori_img[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
            crops.append(crop)
        crops = preprocess_reid(crops, w, h)
        self.det_times.preprocess_time_s.end()

        return crops

    def preprocess(self, crops):
        # to keep fast speed, only use topk crops
        crops = crops[:self.batch_size]
        inputs = {}
        inputs['crops'] = np.array(crops).astype('float32')
        return inputs

    def postprocess(self, pred_dets, pred_embs):
        tracker = self.tracker
        tracker.predict()
        online_targets = tracker.update(pred_dets, pred_embs)

        online_tlwhs, online_scores, online_ids = [], [], []
        for t in online_targets:
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            tlwh = t.to_tlwh()
            tscore = t.score
            tid = t.track_id
            if tlwh[2] * tlwh[3] <= tracker.min_box_area:
                continue
            if tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                    3] > tracker.vertical_ratio:
                continue
            online_tlwhs.append(tlwh)
            online_scores.append(tscore)
            online_ids.append(tid)

        tracking_outs = {
            'online_tlwhs': online_tlwhs,
            'online_scores': online_scores,
            'online_ids': online_ids,
        }
        return tracking_outs

    def postprocess_mtmct(self, pred_dets, pred_embs, frame_id, seq_name):
        tracker = self.tracker
        tracker.predict()
        online_targets = tracker.update(pred_dets, pred_embs)

        online_tlwhs, online_scores, online_ids = [], [], []
        online_tlbrs, online_feats = [], []
        for t in online_targets:
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            tlwh = t.to_tlwh()
            tscore = t.score
            tid = t.track_id
            if tlwh[2] * tlwh[3] <= tracker.min_box_area:
                continue
            if tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                    3] > tracker.vertical_ratio:
                continue
            online_tlwhs.append(tlwh)
            online_scores.append(tscore)
            online_ids.append(tid)

            online_tlbrs.append(t.to_tlbr())
            online_feats.append(t.feat)

        tracking_outs = {
            'online_tlwhs': online_tlwhs,
            'online_scores': online_scores,
            'online_ids': online_ids,
            'feat_data': {},
        }
        for _tlbr, _id, _feat in zip(online_tlbrs, online_ids, online_feats):
            feat_data = {}
            feat_data['bbox'] = _tlbr
            feat_data['frame'] = f"{frame_id:06d}"
            feat_data['id'] = _id
            _imgname = f'{seq_name}_{_id}_{frame_id}.jpg'
            feat_data['imgname'] = _imgname
            feat_data['feat'] = _feat
            tracking_outs['feat_data'].update({_imgname: feat_data})
        return tracking_outs

    def predict(self,
                crops,
                pred_dets,
                repeats=1,
                add_timer=True,
                MTMCT=False,
                frame_id=0,
                seq_name=''):
        # preprocess
        if add_timer:
            self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(crops)
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        if add_timer:
            self.det_times.preprocess_time_s.end()
            self.det_times.inference_time_s.start()

        # model prediction
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            feature_tensor = self.predictor.get_output_handle(output_names[0])
            pred_embs = feature_tensor.copy_to_cpu()
        if add_timer:
            self.det_times.inference_time_s.end(repeats=repeats)
            self.det_times.postprocess_time_s.start()

        # postprocess
        if MTMCT == False:
            tracking_outs = self.postprocess(pred_dets, pred_embs)
        else:
            tracking_outs = self.postprocess_mtmct(pred_dets, pred_embs,
                                                   frame_id, seq_name)
        if add_timer:
            self.det_times.postprocess_time_s.end()
            self.det_times.img_num += 1

        return tracking_outs


def predict_image(detector,
                  reid_model,
                  image_list,
                  threshold,
                  output_dir,
                  scaled=True,
                  save_images=True,
                  run_benchmark=False):
    image_list.sort()
    for i, img_file in enumerate(image_list):
        frame = cv2.imread(img_file)
        ori_image_shape = list(frame.shape[:2])
        if run_benchmark:
            # warmup
            pred_dets, pred_xyxys = detector.predict(
                [img_file],
                ori_image_shape,
                threshold,
                scaled,
                repeats=10,
                add_timer=False)
            # run benchmark
            pred_dets, pred_xyxys = detector.predict(
                [img_file],
                ori_image_shape,
                threshold,
                scaled,
                repeats=10,
                add_timer=True)

            cm, gm, gu = get_current_memory_mb()
            detector.cpu_mem += cm
            detector.gpu_mem += gm
            detector.gpu_util += gu
            print('Test iter {}, file name:{}'.format(i, img_file))
        else:
            pred_dets, pred_xyxys = detector.predict(
                [img_file], ori_image_shape, threshold, scaled)

        if len(pred_dets) == 1 and np.sum(pred_dets) == 0:
            print('Frame {} has no object, try to modify score threshold.'.
                  format(i))
            online_im = frame
        else:
            # reid process
            crops = reid_model.get_crops(pred_xyxys, frame)

            if run_benchmark:
                # warmup
                tracking_outs = reid_model.predict(
                    crops, pred_dets, repeats=10, add_timer=False)
                # run benchmark 
                tracking_outs = reid_model.predict(
                    crops, pred_dets, repeats=10, add_timer=True)

            else:
                tracking_outs = reid_model.predict(crops, pred_dets)

                online_tlwhs = tracking_outs['online_tlwhs']
                online_scores = tracking_outs['online_scores']
                online_ids = tracking_outs['online_ids']

                online_im = plot_tracking(
                    frame, online_tlwhs, online_ids, online_scores, frame_id=i)

        if save_images:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            img_name = os.path.split(img_file)[-1]
            out_path = os.path.join(output_dir, img_name)
            cv2.imwrite(out_path, online_im)
            print("save result to: " + out_path)


def predict_video(detector,
                  reid_model,
                  video_file,
                  scaled,
                  threshold,
                  output_dir,
                  save_images=True,
                  save_mot_txts=True,
                  draw_center_traj=False,
                  secs_interval=10,
                  do_entrance_counting=False,
                  camera_id=-1):
    video_name = 'mot_output.mp4'
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
    else:
        capture = cv2.VideoCapture(video_file)
        video_name = os.path.split(video_file)[-1]

    # Get Video info : resolution, fps, frame count
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps: %d, frame_count: %d" % (fps, frame_count))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_path = os.path.join(output_dir, video_name)
    if not save_images:
        video_format = 'mp4v'
        fourcc = cv2.VideoWriter_fourcc(*video_format)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    frame_id = 0
    timer = MOTTimer()
    results = defaultdict(list)
    id_set = set()
    interval_id_set = set()
    in_id_list = list()
    out_id_list = list()
    prev_center = dict()
    records = list()
    entrance = [0, height / 2., width, height / 2.]
    video_fps = fps

    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        timer.tic()
        ori_image_shape = list(frame.shape[:2])
        pred_dets, pred_xyxys = detector.predict([frame], ori_image_shape,
                                                 threshold, scaled)

        if len(pred_dets) == 1 and np.sum(pred_dets) == 0:
            print('Frame {} has no object, try to modify score threshold.'.
                  format(frame_id))
            timer.toc()
            im = frame
        else:
            # reid process
            crops = reid_model.get_crops(pred_xyxys, frame)
            tracking_outs = reid_model.predict(crops, pred_dets)

            online_tlwhs = tracking_outs['online_tlwhs']
            online_scores = tracking_outs['online_scores']
            online_ids = tracking_outs['online_ids']

            results[0].append(
                (frame_id + 1, online_tlwhs, online_scores, online_ids))
            # NOTE: just implement flow statistic for one class
            result = (frame_id + 1, online_tlwhs, online_scores, online_ids)
            statistic = flow_statistic(
                result, secs_interval, do_entrance_counting, video_fps,
                entrance, id_set, interval_id_set, in_id_list, out_id_list,
                prev_center, records)
            id_set = statistic['id_set']
            interval_id_set = statistic['interval_id_set']
            in_id_list = statistic['in_id_list']
            out_id_list = statistic['out_id_list']
            prev_center = statistic['prev_center']
            records = statistic['records']

            timer.toc()

            fps = 1. / timer.duration
            im = plot_tracking(
                frame,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=frame_id,
                fps=fps,
                do_entrance_counting=do_entrance_counting,
                entrance=entrance)

        if save_images:
            save_dir = os.path.join(output_dir, video_name.split('.')[-2])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(
                os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), im)
        else:
            writer.write(im)

        frame_id += 1
        print('detect frame:%d, fps: %f' % (frame_id, fps))

        if camera_id != -1:
            cv2.imshow('Tracking Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if save_mot_txts:
        result_filename = os.path.join(output_dir,
                                       video_name.split('.')[-2] + '.txt')
        write_mot_results(result_filename, results)

        result_filename = os.path.join(
            output_dir, video_name.split('.')[-2] + '_flow_statistic.txt')
        f = open(result_filename, 'w')
        for line in records:
            f.write(line)
        print('Flow statistic save in {}'.format(result_filename))
        f.close()

    if save_images:
        save_dir = os.path.join(output_dir, video_name.split('.')[-2])
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg {}'.format(save_dir,
                                                              out_path)
        os.system(cmd_str)
        print('Save video in {}.'.format(out_path))
    else:
        writer.release()


def predict_mtmct_seq(detector,
                      reid_model,
                      mtmct_dir,
                      seq_name,
                      scaled,
                      threshold,
                      output_dir,
                      save_images=True,
                      save_mot_txts=True):
    fpath = os.path.join(mtmct_dir, seq_name)
    if os.path.exists(os.path.join(fpath, 'img1')):
        fpath = os.path.join(fpath, 'img1')

    assert os.path.isdir(fpath), '{} should be a directory'.format(fpath)
    image_list = os.listdir(fpath)
    image_list.sort()
    assert len(image_list) > 0, '{} has no images.'.format(fpath)

    results = defaultdict(list)
    mot_features_dict = {}  # cid_tid_fid feats
    print('Totally {} frames found in seq {}.'.format(
        len(image_list), seq_name))

    for frame_id, img_file in enumerate(image_list):
        if frame_id % 10 == 0:
            print('Processing frame {} of seq {}.'.format(frame_id, seq_name))
        frame = cv2.imread(os.path.join(fpath, img_file))
        ori_image_shape = list(frame.shape[:2])
        frame_path = os.path.join(fpath, img_file)
        pred_dets, pred_xyxys = detector.predict([frame_path], ori_image_shape,
                                                 threshold, scaled)

        if len(pred_dets) == 1 and np.sum(pred_dets) == 0:
            print('Frame {} has no object, try to modify score threshold.'.
                  format(frame_id))
            online_im = frame
        else:
            # reid process
            crops = reid_model.get_crops(pred_xyxys, frame)

            tracking_outs = reid_model.predict(
                crops,
                pred_dets,
                MTMCT=True,
                frame_id=frame_id,
                seq_name=seq_name)

            feat_data_dict = tracking_outs['feat_data']
            mot_features_dict = dict(mot_features_dict, **feat_data_dict)

            online_tlwhs = tracking_outs['online_tlwhs']
            online_scores = tracking_outs['online_scores']
            online_ids = tracking_outs['online_ids']

            online_im = plot_tracking(frame, online_tlwhs, online_ids,
                                      online_scores, frame_id)
            results[0].append(
                (frame_id + 1, online_tlwhs, online_scores, online_ids))

        if save_images:
            save_dir = os.path.join(output_dir, seq_name)
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            img_name = os.path.split(img_file)[-1]
            out_path = os.path.join(save_dir, img_name)
            cv2.imwrite(out_path, online_im)

    if save_mot_txts:
        result_filename = os.path.join(output_dir, seq_name + '.txt')
        write_mot_results(result_filename, results)

    return mot_features_dict


def predict_mtmct(detector,
                  reid_model,
                  mtmct_dir,
                  mtmct_cfg,
                  scaled,
                  threshold,
                  output_dir,
                  save_images=True,
                  save_mot_txts=True):
    MTMCT = mtmct_cfg['MTMCT']
    assert MTMCT == True, 'predict_mtmct should be used for MTMCT.'

    cameras_bias = mtmct_cfg['cameras_bias']
    cid_bias = parse_bias(cameras_bias)
    scene_cluster = list(cid_bias.keys())

    # 1.zone releated parameters
    use_zone = mtmct_cfg['use_zone']
    zone_path = mtmct_cfg['zone_path']

    # 2.tricks parameters, can be used for other mtmct dataset
    use_ff = mtmct_cfg['use_ff']
    use_rerank = mtmct_cfg['use_rerank']

    # 3.camera releated parameters
    use_camera = mtmct_cfg['use_camera']
    use_st_filter = mtmct_cfg['use_st_filter']

    # 4.zone releated parameters
    use_roi = mtmct_cfg['use_roi']
    roi_dir = mtmct_cfg['roi_dir']

    mot_list_breaks = []
    cid_tid_dict = dict()

    if not os.path.exists(output_dir): os.makedirs(output_dir)

    seqs = os.listdir(mtmct_dir)
    seqs.sort()

    for seq in seqs:
        fpath = os.path.join(mtmct_dir, seq)
        if os.path.isfile(fpath) and _is_valid_video(fpath):
            ext = seq.split('.')[-1]
            seq = seq.split('.')[-2]
            print('ffmpeg processing of video {}'.format(fpath))
            frames_path = video2frames(
                video_path=fpath, outpath=mtmct_dir, frame_rate=25)
            fpath = os.path.join(mtmct_dir, seq)

        if os.path.isdir(fpath) == False:
            print('{} is not a image folder.'.format(fpath))
            continue

        mot_features_dict = predict_mtmct_seq(
            detector, reid_model, mtmct_dir, seq, scaled, threshold, output_dir,
            save_images, save_mot_txts)

        cid = int(re.sub('[a-z,A-Z]', "", seq))
        tid_data, mot_list_break = trajectory_fusion(
            mot_features_dict,
            cid,
            cid_bias,
            use_zone=use_zone,
            zone_path=zone_path)
        mot_list_breaks.append(mot_list_break)
        # single seq process
        for line in tid_data:
            tracklet = tid_data[line]
            tid = tracklet['tid']
            if (cid, tid) not in cid_tid_dict:
                cid_tid_dict[(cid, tid)] = tracklet

    map_tid = sub_cluster(
        cid_tid_dict,
        scene_cluster,
        use_ff=use_ff,
        use_rerank=use_rerank,
        use_camera=use_camera,
        use_st_filter=use_st_filter)

    pred_mtmct_file = os.path.join(output_dir, 'mtmct_result.txt')
    if use_camera:
        gen_res(pred_mtmct_file, scene_cluster, map_tid, mot_list_breaks)
    else:
        gen_res(
            pred_mtmct_file,
            scene_cluster,
            map_tid,
            mot_list_breaks,
            use_roi=use_roi,
            roi_dir=roi_dir)

    if FLAGS.save_images:
        camera_results, cid_tid_fid_res = get_mtmct_matching_results(
            pred_mtmct_file)

        crops_dir = os.path.join(output_dir, 'mtmct_crops')
        save_mtmct_crops(
            cid_tid_fid_res, images_dir=mtmct_dir, crops_dir=crops_dir)

        save_dir = os.path.join(output_dir, 'mtmct_vis')
        save_mtmct_vis_results(
            camera_results,
            images_dir=mtmct_dir,
            save_dir=save_dir,
            save_videos=FLAGS.save_images)

    # evalution metrics
    data_root_gt = os.path.join(mtmct_dir, '..', 'gt', 'gt.txt')
    if os.path.exists(data_root_gt):
        print_mtmct_result(data_root_gt, pred_mtmct_file)


def predict_naive(model_dir,
                  reid_model_dir,
                  video_file,
                  image_dir,
                  mtmct_dir=None,
                  mtmct_cfg=None,
                  scaled=True,
                  device='gpu',
                  threshold=0.5,
                  output_dir='output'):
    pred_config = PredictConfig(model_dir)
    detector_func = 'SDE_Detector'
    if pred_config.arch == 'PicoDet':
        detector_func = 'SDE_DetectorPicoDet'
    detector = eval(detector_func)(pred_config, model_dir, device=device)

    pred_config = PredictConfig(reid_model_dir)
    reid_model = SDE_ReID(pred_config, reid_model_dir, device=device)

    if video_file is not None:
        predict_video(
            detector,
            reid_model,
            video_file,
            scaled=scaled,
            threshold=threshold,
            output_dir=output_dir,
            save_images=True,
            save_mot_txts=True,
            draw_center_traj=False,
            secs_interval=10,
            do_entrance_counting=False)
    elif mtmct_dir is not None:
        with open(mtmct_cfg) as f:
            mtmct_cfg_file = yaml.safe_load(f)
        predict_mtmct(
            detector,
            reid_model,
            mtmct_dir,
            mtmct_cfg_file,
            scaled=scaled,
            threshold=threshold,
            output_dir=output_dir,
            save_images=True,
            save_mot_txts=True)
    else:
        img_list = get_test_images(image_dir, infer_img=None)
        predict_image(
            detector,
            reid_model,
            img_list,
            threshold=threshold,
            output_dir=output_dir,
            save_images=True)


def main():
    pred_config = PredictConfig(FLAGS.model_dir)
    detector_func = 'SDE_Detector'
    if pred_config.arch == 'PicoDet':
        detector_func = 'SDE_DetectorPicoDet'

    detector = eval(detector_func)(pred_config,
                                   FLAGS.model_dir,
                                   device=FLAGS.device,
                                   run_mode=FLAGS.run_mode,
                                   batch_size=FLAGS.batch_size,
                                   trt_min_shape=FLAGS.trt_min_shape,
                                   trt_max_shape=FLAGS.trt_max_shape,
                                   trt_opt_shape=FLAGS.trt_opt_shape,
                                   trt_calib_mode=FLAGS.trt_calib_mode,
                                   cpu_threads=FLAGS.cpu_threads,
                                   enable_mkldnn=FLAGS.enable_mkldnn)

    pred_config = PredictConfig(FLAGS.reid_model_dir)
    reid_model = SDE_ReID(
        pred_config,
        FLAGS.reid_model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        batch_size=FLAGS.reid_batch_size,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn)

    # predict from video file or camera video stream
    if FLAGS.video_file is not None or FLAGS.camera_id != -1:
        predict_video(
            detector,
            reid_model,
            FLAGS.video_file,
            scaled=FLAGS.scaled,
            threshold=FLAGS.threshold,
            output_dir=FLAGS.output_dir,
            save_images=FLAGS.save_images,
            save_mot_txts=FLAGS.save_mot_txts,
            draw_center_traj=FLAGS.draw_center_traj,
            secs_interval=FLAGS.secs_interval,
            do_entrance_counting=FLAGS.do_entrance_counting,
            camera_id=FLAGS.camera_id)

    elif FLAGS.mtmct_dir is not None:
        mtmct_cfg_file = FLAGS.mtmct_cfg
        with open(mtmct_cfg_file) as f:
            mtmct_cfg = yaml.safe_load(f)
        predict_mtmct(
            detector,
            reid_model,
            FLAGS.mtmct_dir,
            mtmct_cfg,
            scaled=FLAGS.scaled,
            threshold=FLAGS.threshold,
            output_dir=FLAGS.output_dir,
            save_images=FLAGS.save_images,
            save_mot_txts=FLAGS.save_mot_txts)
    else:
        # predict from image
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        predict_image(
            detector,
            reid_model,
            img_list,
            threshold=FLAGS.threshold,
            output_dir=FLAGS.output_dir,
            save_images=FLAGS.save_images,
            run_benchmark=FLAGS.run_benchmark)

        if not FLAGS.run_benchmark:
            detector.det_times.info(average=True)
            reid_model.det_times.info(average=True)
        else:
            mode = FLAGS.run_mode
            det_model_dir = FLAGS.model_dir
            det_model_info = {
                'model_name': det_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(detector, img_list, det_model_info, name='Det')

            reid_model_dir = FLAGS.reid_model_dir
            reid_model_info = {
                'model_name': reid_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(reid_model, img_list, reid_model_info, name='ReID')


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()
