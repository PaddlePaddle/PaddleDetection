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
from utils import argsparser, Timer, get_current_memory_mb
from det_infer import Detector, DetectorPicoDet, get_test_images, print_arguments, PredictConfig
from det_infer import load_predictor
from benchmark_utils import PaddleInferBenchmark
from visualize import plot_tracking

from mot.tracker import DeepSORTTracker
from mot.utils import MOTTimer, write_mot_results, flow_statistic

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


def scale_coords(coords, input_shape, im_shape, scale_factor):
    im_shape = im_shape[0]
    ratio = scale_factor[0][0]
    pad_w = (input_shape[1] - int(im_shape[1])) / 2
    pad_h = (input_shape[0] - int(im_shape[0])) / 2
    coords[:, 0::2] -= pad_w
    coords[:, 1::2] -= pad_h
    coords[:, 0:4] /= ratio
    coords[:, :4] = np.clip(coords[:, :4], a_min=0, a_max=coords[:, :4].max())
    return coords.round()


def clip_box(xyxy, input_shape, im_shape, scale_factor):
    im_shape = im_shape[0]
    ratio = scale_factor[0][0]
    img0_shape = [int(im_shape[0] / ratio), int(im_shape[1] / ratio)]
    xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], a_min=0, a_max=img0_shape[1])
    xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], a_min=0, a_max=img0_shape[0])
    w = xyxy[:, 2:3] - xyxy[:, 0:1]
    h = xyxy[:, 3:4] - xyxy[:, 1:2]
    mask = np.logical_and(h > 0, w > 0)
    keep_idx = np.nonzero(mask)
    return xyxy[keep_idx[0]], keep_idx


def preprocess_reid(imgs,
                    w=64,
                    h=192,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]):
    im_batch = []
    for img in imgs:
        img = cv2.resize(img, (w, h))
        img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
        img_mean = np.array(mean).reshape((3, 1, 1))
        img_std = np.array(std).reshape((3, 1, 1))
        img -= img_mean
        img /= img_std
        img = np.expand_dims(img, axis=0)
        im_batch.append(img)
    im_batch = np.concatenate(im_batch, 0)
    return im_batch


class SDE_Detector(Detector):
    """
    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16)
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
                 run_mode='fluid',
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
        assert batch_size == 1, "The JDE Detector only supports batch size=1 now"
        self.pred_config = pred_config

    def postprocess(self, boxes, input_shape, im_shape, scale_factor, threshold,
                    scaled):
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
            pred_bboxes = scale_coords(boxes[:, 2:], input_shape, im_shape,
                                       scale_factor)
        else:
            pred_bboxes = boxes[:, 2:]

        pred_xyxys, keep_idx = clip_box(pred_bboxes, input_shape, im_shape,
                                        scale_factor)
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

    def predict(self, image, scaled, threshold=0.5, warmup=0, repeats=1):
        '''
        Args:
            image (np.ndarray): image numpy data
            threshold (float): threshold of predicted box' score
            scaled (bool): whether the coords after detector outputs are scaled,
                default False in jde yolov3, set True in general detector.
        Returns:
            pred_dets (np.ndarray, [N, 6])
        '''
        self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(image)
        self.det_times.preprocess_time_s.end()

        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            boxes = boxes_tensor.copy_to_cpu()

        self.det_times.inference_time_s.start()
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            boxes = boxes_tensor.copy_to_cpu()
        self.det_times.inference_time_s.end(repeats=repeats)

        self.det_times.postprocess_time_s.start()
        if len(boxes) == 0:
            pred_dets = np.zeros((1, 6), dtype=np.float32)
            pred_xyxys = np.zeros((1, 4), dtype=np.float32)
        else:
            input_shape = inputs['image'].shape[2:]
            im_shape = inputs['im_shape']
            scale_factor = inputs['scale_factor']

            pred_dets, pred_xyxys = self.postprocess(
                boxes, input_shape, im_shape, scale_factor, threshold, scaled)

        self.det_times.postprocess_time_s.end()
        self.det_times.img_num += 1
        return pred_dets, pred_xyxys


class SDE_DetectorPicoDet(DetectorPicoDet):
    """
    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16)
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
                 run_mode='fluid',
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
        assert batch_size == 1, "The JDE Detector only supports batch size=1 now"
        self.pred_config = pred_config

    def postprocess_bboxes(self, boxes, input_shape, im_shape, scale_factor,
                           threshold):
        over_thres_idx = np.nonzero(boxes[:, 1:2] >= threshold)[0]
        if len(over_thres_idx) == 0:
            pred_dets = np.zeros((1, 6), dtype=np.float32)
            pred_xyxys = np.zeros((1, 4), dtype=np.float32)
            return pred_dets, pred_xyxys
        else:
            boxes = boxes[over_thres_idx]

        pred_bboxes = boxes[:, 2:]

        pred_xyxys, keep_idx = clip_box(pred_bboxes, input_shape, im_shape,
                                        scale_factor)
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

    def predict(self, image, scaled, threshold=0.5, warmup=0, repeats=1):
        '''
        Args:
            image (np.ndarray): image numpy data
            threshold (float): threshold of predicted box' score
            scaled (bool): whether the coords after detector outputs are scaled,
                default False in jde yolov3, set True in general detector.
        Returns:
            pred_dets (np.ndarray, [N, 6])
        '''
        self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(image)
        self.det_times.preprocess_time_s.end()

        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        np_score_list, np_boxes_list = [], []
        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            boxes = boxes_tensor.copy_to_cpu()

        self.det_times.inference_time_s.start()
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

        self.det_times.inference_time_s.end(repeats=repeats)
        self.det_times.img_num += 1
        self.det_times.postprocess_time_s.start()
        self.postprocess = PicoDetPostProcess(
            inputs['image'].shape[2:],
            inputs['im_shape'],
            inputs['scale_factor'],
            strides=self.pred_config.fpn_stride,
            nms_threshold=self.pred_config.nms['nms_threshold'])
        boxes, boxes_num = self.postprocess(np_score_list, np_boxes_list)

        if len(boxes) == 0:
            pred_dets = np.zeros((1, 6), dtype=np.float32)
            pred_xyxys = np.zeros((1, 4), dtype=np.float32)
        else:
            input_shape = inputs['image'].shape[2:]
            im_shape = inputs['im_shape']
            scale_factor = inputs['scale_factor']
            pred_dets, pred_xyxys = self.postprocess_bboxes(
                boxes, input_shape, im_shape, scale_factor, threshold)

        return pred_dets, pred_xyxys


class SDE_ReID(object):
    def __init__(self,
                 pred_config,
                 model_dir,
                 device='CPU',
                 run_mode='fluid',
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
            if tlwh[2] * tlwh[3] <= tracker.min_box_area: continue
            if tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                    3] > tracker.vertical_ratio:
                continue
            online_tlwhs.append(tlwh)
            online_scores.append(tscore)
            online_ids.append(tid)

        return online_tlwhs, online_scores, online_ids

    def predict(self, crops, pred_dets, warmup=0, repeats=1):
        self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(crops)
        self.det_times.preprocess_time_s.end()

        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            feature_tensor = self.predictor.get_output_handle(output_names[0])
            pred_embs = feature_tensor.copy_to_cpu()

        self.det_times.inference_time_s.start()
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            feature_tensor = self.predictor.get_output_handle(output_names[0])
            pred_embs = feature_tensor.copy_to_cpu()
        self.det_times.inference_time_s.end(repeats=repeats)

        self.det_times.postprocess_time_s.start()
        online_tlwhs, online_scores, online_ids = self.postprocess(pred_dets,
                                                                   pred_embs)
        self.det_times.postprocess_time_s.end()
        self.det_times.img_num += 1

        return online_tlwhs, online_scores, online_ids


def predict_image(detector, reid_model, image_list):
    image_list.sort()
    for i, img_file in enumerate(image_list):
        frame = cv2.imread(img_file)
        if FLAGS.run_benchmark:
            pred_dets, pred_xyxys = detector.predict(
                [frame], FLAGS.scaled, FLAGS.threshold, warmup=10, repeats=10)
            cm, gm, gu = get_current_memory_mb()
            detector.cpu_mem += cm
            detector.gpu_mem += gm
            detector.gpu_util += gu
            print('Test iter {}, file name:{}'.format(i, img_file))
        else:
            pred_dets, pred_xyxys = detector.predict([frame], FLAGS.scaled,
                                                     FLAGS.threshold)

        if len(pred_dets) == 1 and np.sum(pred_dets) == 0:
            print('Frame {} has no object, try to modify score threshold.'.
                  format(i))
            online_im = frame
        else:
            # reid process
            crops = reid_model.get_crops(pred_xyxys, frame)

            if FLAGS.run_benchmark:
                online_tlwhs, online_scores, online_ids = reid_model.predict(
                    crops, pred_dets, warmup=10, repeats=10)
            else:
                online_tlwhs, online_scores, online_ids = reid_model.predict(
                    crops, pred_dets)
                online_im = plot_tracking(
                    frame, online_tlwhs, online_ids, online_scores, frame_id=i)

        if FLAGS.save_images:
            if not os.path.exists(FLAGS.output_dir):
                os.makedirs(FLAGS.output_dir)
            img_name = os.path.split(img_file)[-1]
            out_path = os.path.join(FLAGS.output_dir, img_name)
            cv2.imwrite(out_path, online_im)
            print("save result to: " + out_path)


def predict_video(detector, reid_model, camera_id):
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
        video_name = 'mot_output.mp4'
    else:
        capture = cv2.VideoCapture(FLAGS.video_file)
        video_name = os.path.split(FLAGS.video_file)[-1]
    # Get Video info : resolution, fps, frame count
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps: %d, frame_count: %d" % (fps, frame_count))

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    out_path = os.path.join(FLAGS.output_dir, video_name)
    if not FLAGS.save_images:
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
        pred_dets, pred_xyxys = detector.predict([frame], FLAGS.scaled,
                                                 FLAGS.threshold)

        if len(pred_dets) == 1 and np.sum(pred_dets) == 0:
            print('Frame {} has no object, try to modify score threshold.'.
                  format(frame_id))
            timer.toc()
            im = frame
        else:
            # reid process
            crops = reid_model.get_crops(pred_xyxys, frame)
            online_tlwhs, online_scores, online_ids = reid_model.predict(
                crops, pred_dets)
            results[0].append(
                (frame_id + 1, online_tlwhs, online_scores, online_ids))
            # NOTE: just implement flow statistic for one class
            result = (frame_id + 1, online_tlwhs, online_scores, online_ids)
            statistic = flow_statistic(
                result, FLAGS.secs_interval, FLAGS.do_entrance_counting,
                video_fps, entrance, id_set, interval_id_set, in_id_list,
                out_id_list, prev_center, records)
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
                do_entrance_counting=FLAGS.do_entrance_counting,
                entrance=entrance)

        if FLAGS.save_images:
            save_dir = os.path.join(FLAGS.output_dir, video_name.split('.')[-2])
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

    if FLAGS.save_mot_txts:
        result_filename = os.path.join(FLAGS.output_dir,
                                       video_name.split('.')[-2] + '.txt')
        write_mot_results(result_filename, results)

        result_filename = os.path.join(
            FLAGS.output_dir, video_name.split('.')[-2] + '_flow_statistic.txt')
        f = open(result_filename, 'w')
        for line in records:
            f.write(line)
        print('Flow statistic save in {}'.format(result_filename))
        f.close()

    if FLAGS.save_images:
        save_dir = os.path.join(FLAGS.output_dir, video_name.split('.')[-2])
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg {}'.format(save_dir,
                                                              out_path)
        os.system(cmd_str)
        print('Save video in {}.'.format(out_path))
    else:
        writer.release()


def predict_mtmct_per_video(detector, reid_model, video_file, output_dir):
    capture = cv2.VideoCapture(video_file)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps: %d, frame_count: %d" % (fps, frame_count))

    video_name = os.path.split(video_file)[-1]
    out_path = os.path.join(output_dir, video_name)
    if not FLAGS.save_images:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    frame_id = 0
    timer = MOTTimer()
    results = defaultdict(list)
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        timer.tic()
        pred_dets, pred_xyxys = detector.predict([frame], FLAGS.scaled,
                                                 FLAGS.threshold)

        if len(pred_dets) == 1 and np.sum(pred_dets) == 0:
            print('Frame {} has no object, try to modify score threshold.'.
                  format(frame_id))
            timer.toc()
            im = frame
        else:
            # reid process
            crops = reid_model.get_crops(pred_xyxys, frame)
            online_tlwhs, online_scores, online_ids = reid_model.predict(
                crops, pred_dets)
            results[0].append(
                (frame_id + 1, online_tlwhs, online_scores, online_ids))
            timer.toc()
            
            fps = 1. / timer.average_time
            im = plot_tracking(
                frame,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=frame_id,
                fps=fps)
        if FLAGS.save_images:
            save_dir = os.path.join(output_dir, video_name.split('.')[-2])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(
                os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), im)
        else:
            writer.write(im)

        frame_id += 1
        print('detect frame:%d' % (frame_id))

    if FLAGS.save_mot_txts:
        result_filename = os.path.join(output_dir,
                                       video_name.split('.')[-2] + '.txt')
        write_mot_results(result_filename, results)

    if FLAGS.save_images:
        save_dir = os.path.join(FLAGS.output_dir, video_name.split('.')[-2])
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg {}'.format(save_dir,
                                                              out_path)
        os.system(cmd_str)
        print('Save video in {}.'.format(out_path))
    else:
        writer.release()

    return results[0]

def predict_mtmct_per_folder(detector, reid_model, fpath, output_dir):
    image_list = os.listdir(fpath)
    image_list.sort()
    results = defaultdict(list)
    seq_name = fpath.split('/')[-1]
    for frame_id, img_file in enumerate(image_list):
        frame = cv2.imread(os.path.join(fpath, img_file))
        pred_dets, pred_xyxys = detector.predict([frame], FLAGS.scaled,
                                                 FLAGS.threshold)
        if len(pred_dets) == 1 and np.sum(pred_dets) == 0:
            print('Frame {} has no object, try to modify score threshold.'.
                  format(frame_id))
            online_im = frame
        else:
            # reid process
            crops = reid_model.get_crops(pred_xyxys, frame)

            if FLAGS.run_benchmark:
                online_tlwhs, online_scores, online_ids = reid_model.predict(
                    crops, pred_dets, warmup=10, repeats=10)
            else:
                online_tlwhs, online_scores, online_ids = reid_model.predict(
                    crops, pred_dets)
                online_im = plot_tracking(
                    frame, online_tlwhs, online_ids, online_scores, frame_id)
                results[0].append(
                    (frame_id + 1, online_tlwhs, online_scores, online_ids))
        if FLAGS.save_images:
            save_dir = os.path.join(output_dir, seq_name)
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            img_name = os.path.split(img_file)[-1]
            out_path = os.path.join(save_dir, img_name)
            cv2.imwrite(out_path, online_im)
            if frame_id % 40 == 0:
                print("save result to: " + out_path)
    return results[0]

def predict_mtmct(detector, reid_model, mtmct_dir, mtmct_cfg):
    MTMCT = mtmct_cfg.MTMCT
    assert MTMCT == True, 'predict_mtmct should be used for MTMCT.'

    cameras_bias = mtmct_cfg.cameras_bias
    cid_bias = parse_bias(cameras_bias)
    scene_cluster = list(cid_bias.keys())

    # 1.zone releated parameters
    use_zone = mtmct_cfg.use_zone
    zone_path = mtmct_cfg.zone_path
    
    # 2.tricks parameters, can be used for other mtmct dataset
    use_ff = mtmct_cfg.use_ff
    use_rerank = mtmct_cfg.use_rerank

    # 3.camera releated parameters
    use_camera = mtmct_cfg.use_camera
    use_st_filter = mtmct_cfg.use_st_filter

    # 4.zone releated parameters
    use_roi = mtmct_cfg.use_roi
    roi_dir = mtmct_cfg.roi_dir

    mot_features_list = []
    mot_list_breaks = []
    cid_tid_dict = dict()

    output_dir = FLAGS.output_dir
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    seqs = os.listdir(mtmct_dir)
    seqs.sort()
    for seq in seqs:
        fpath = os.path.join(mtmct_dir, seq)
        if os.path.isfile(fpath):
            mot_features = predict_mtmct_per_video(detector, reid_model, fpath, output_dir)
        elif os.path.isdir(fpath):
            mot_features = predict_mtmct_per_folder(detector, reid_model, fpath, output_dir)
        else:
            print('{} is not a video or image folder.'.format(fpath))
            continue

        # from mot_feature gen mot_list
        mot_features_list.append(results)
        cid = int(re.sub('[a-z,A-Z]', "", seq))
        tid_data, mot_list_break = trajectory_fusion(results, cid, cid_bias, use_zone=use_zone, zone_path=zone_path)
        mot_list_breaks.append(mot_list_break)
        # single seq process
        for line in tid_data:
            tracklet = tid_data[line]
            tid = tracklet['tid']
            if (cid, tid) not in cid_tid_dict:
                cid_tid_dict[(cid, tid)] = tracklet

    map_tid = sub_cluster(cid_tid_dict, scene_cluster, use_ff=use_ff, use_rerank=use_rerank, use_camera=use_camera, use_st_filter=use_st_filter)

    pred_mtmct_file = osp.join(output_dir, 'mtmct_result.txt')
    if use_camera:
        gen_res(pred_mtmct_file, scene_cluster, map_tid, mot_features_list)
    else:
        gen_res(pred_mtmct_file, scene_cluster, map_tid, mot_list_breaks, use_roi=use_roi, roi_dir=data_root)
    # accumulate metric to log out
    data_root_gt = osp.join(str.join('/', data_root.split('/')[0:-1]),'gt','gt.txt')
    print_mtmct_result(data_root_gt, pred_mtmct_file)

    carame_results, cid_tid_fid_res = get_mtmct_matching_results(pred_mtmct_file)
    save_mtmct_crops(cid_tid_fid_res, images_dir=data_root, crops_dir=output_dir)
    save_mtmct_vis_results(carame_results, images_dir=data_root, save_dir=output_dir, save_videos=save_videos)


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
        predict_video(detector, reid_model, FLAGS.camera_id)

    elif FLAGS.mtmct_dir is not None:
        mtmct_cfg_file = FLAGS.mtmct_cfg
        with open(mtmct_cfg_file) as f:
            mtmct_cfg = yaml.safe_load(f)
        predict_mtmct(detector, reid_model, FLAGS.mtmct_dir, mtmct_cfg)

    else:
        # predict from image
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        predict_image(detector, reid_model, img_list)

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
