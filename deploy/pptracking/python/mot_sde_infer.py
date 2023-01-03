# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import glob
import numpy as np
from collections import defaultdict
import paddle

from benchmark_utils import PaddleInferBenchmark
from preprocess import decode_image

# add python path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'])))
sys.path.insert(0, parent_path)

from det_infer import Detector, get_test_images, print_arguments, bench_log, PredictConfig, load_predictor
from mot_utils import argsparser, Timer, get_current_memory_mb, video2frames, _is_valid_video
from mot.tracker import JDETracker, DeepSORTTracker, OCSORTTracker, BOTSORTTracker
from mot.utils import MOTTimer, write_mot_results, get_crops, clip_box, flow_statistic
from mot.visualize import plot_tracking, plot_tracking_dict

from mot.mtmct.utils import parse_bias
from mot.mtmct.postprocess import trajectory_fusion, sub_cluster, gen_res, print_mtmct_result
from mot.mtmct.postprocess import get_mtmct_matching_results, save_mtmct_crops, save_mtmct_vis_results


class SDE_Detector(Detector):
    """
    Args:
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        tracker_config (str): tracker config path
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
        output_dir (string): The path of output, default as 'output'
        threshold (float): Score threshold of the detected bbox, default as 0.5
        save_images (bool): Whether to save visualization image results, default as False
        save_mot_txts (bool): Whether to save tracking results (txt), default as False
        draw_center_traj (bool): Whether drawing the trajectory of center, default as False
        secs_interval (int): The seconds interval to count after tracking, default as 10
        skip_frame_num (int): Skip frame num to get faster MOT results, default as -1
        warmup_frame (int):Warmup frame num to test speed of MOT,default as 50
        do_entrance_counting(bool): Whether counting the numbers of identifiers entering 
            or getting out from the entrance, default as False，only support single class
            counting in MOT, and the video should be taken by a static camera.
        do_break_in_counting(bool): Whether counting the numbers of identifiers break in
            the area, default as False，only support single class counting in MOT,
            and the video should be taken by a static camera.
        region_type (str): Area type for entrance counting or break in counting, 'horizontal'
            and 'vertical' used when do entrance counting. 'custom' used when do break in counting. 
            Note that only support single-class MOT, and the video should be taken by a static camera.
        region_polygon (list): Clockwise point coords (x0,y0,x1,y1...) of polygon of area when
            do_break_in_counting. Note that only support single-class MOT and
            the video should be taken by a static camera.
        reid_model_dir (str): reid model dir, default None for ByteTrack, but set for DeepSORT
        mtmct_dir (str): MTMCT dir, default None, set for doing MTMCT
    """

    def __init__(self,
                 model_dir,
                 tracker_config,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 output_dir='output',
                 threshold=0.5,
                 save_images=False,
                 save_mot_txts=False,
                 draw_center_traj=False,
                 secs_interval=10,
                 skip_frame_num=-1,
                 warmup_frame=50,
                 do_entrance_counting=False,
                 do_break_in_counting=False,
                 region_type='horizontal',
                 region_polygon=[],
                 reid_model_dir=None,
                 mtmct_dir=None):
        super(SDE_Detector, self).__init__(
            model_dir=model_dir,
            device=device,
            run_mode=run_mode,
            batch_size=batch_size,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            output_dir=output_dir,
            threshold=threshold, )
        self.save_images = save_images
        self.save_mot_txts = save_mot_txts
        self.draw_center_traj = draw_center_traj
        self.secs_interval = secs_interval
        self.skip_frame_num = skip_frame_num
        self.warmup_frame = warmup_frame
        self.do_entrance_counting = do_entrance_counting
        self.do_break_in_counting = do_break_in_counting
        self.region_type = region_type
        self.region_polygon = region_polygon
        if self.region_type == 'custom':
            assert len(
                self.region_polygon
            ) > 6, 'region_type is custom, region_polygon should be at least 3 pairs of point coords.'

        assert batch_size == 1, "MOT model only supports batch_size=1."
        self.det_times = Timer(with_tracker=True)
        self.num_classes = len(self.pred_config.labels)
        if self.skip_frame_num > 1:
            self.previous_det_result = None

        # reid config
        self.use_reid = False if reid_model_dir is None else True
        if self.use_reid:
            self.reid_pred_config = self.set_config(reid_model_dir)
            self.reid_predictor, self.config = load_predictor(
                reid_model_dir,
                run_mode=run_mode,
                batch_size=50,  # reid_batch_size
                min_subgraph_size=self.reid_pred_config.min_subgraph_size,
                device=device,
                use_dynamic_shape=self.reid_pred_config.use_dynamic_shape,
                trt_min_shape=trt_min_shape,
                trt_max_shape=trt_max_shape,
                trt_opt_shape=trt_opt_shape,
                trt_calib_mode=trt_calib_mode,
                cpu_threads=cpu_threads,
                enable_mkldnn=enable_mkldnn)
        else:
            self.reid_pred_config = None
            self.reid_predictor = None

        assert tracker_config is not None, 'Note that tracker_config should be set.'
        self.tracker_config = tracker_config
        tracker_cfg = yaml.safe_load(open(self.tracker_config))
        cfg = tracker_cfg[tracker_cfg['type']]

        # tracker config
        self.use_deepsort_tracker = True if tracker_cfg[
            'type'] == 'DeepSORTTracker' else False
        self.use_ocsort_tracker = True if tracker_cfg[
            'type'] == 'OCSORTTracker' else False
        self.use_botsort_tracker = True if tracker_cfg[
            'type'] == 'BOTSORTTracker' else False

        if self.use_deepsort_tracker:
            if self.reid_pred_config is not None and hasattr(
                    self.reid_pred_config, 'tracker'):
                cfg = self.reid_pred_config.tracker
            budget = cfg.get('budget', 100)
            max_age = cfg.get('max_age', 30)
            max_iou_distance = cfg.get('max_iou_distance', 0.7)
            matching_threshold = cfg.get('matching_threshold', 0.2)
            min_box_area = cfg.get('min_box_area', 0)
            vertical_ratio = cfg.get('vertical_ratio', 0)

            self.tracker = DeepSORTTracker(
                budget=budget,
                max_age=max_age,
                max_iou_distance=max_iou_distance,
                matching_threshold=matching_threshold,
                min_box_area=min_box_area,
                vertical_ratio=vertical_ratio, )

        elif self.use_ocsort_tracker:
            det_thresh = cfg.get('det_thresh', 0.4)
            max_age = cfg.get('max_age', 30)
            min_hits = cfg.get('min_hits', 3)
            iou_threshold = cfg.get('iou_threshold', 0.3)
            delta_t = cfg.get('delta_t', 3)
            inertia = cfg.get('inertia', 0.2)
            min_box_area = cfg.get('min_box_area', 0)
            vertical_ratio = cfg.get('vertical_ratio', 0)
            use_byte = cfg.get('use_byte', False)
            use_angle_cost = cfg.get('use_angle_cost', False)

            self.tracker = OCSORTTracker(
                det_thresh=det_thresh,
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold,
                delta_t=delta_t,
                inertia=inertia,
                min_box_area=min_box_area,
                vertical_ratio=vertical_ratio,
                use_byte=use_byte,
                use_angle_cost=use_angle_cost)

        elif self.use_botsort_tracker:
            track_high_thresh = cfg.get('track_high_thresh', 0.3)
            track_low_thresh = cfg.get('track_low_thresh', 0.2)
            new_track_thresh = cfg.get('new_track_thresh', 0.4)
            match_thresh = cfg.get('match_thresh', 0.7)
            track_buffer = cfg.get('track_buffer', 30)
            camera_motion = cfg.get('camera_motion', False)
            cmc_method = cfg.get('cmc_method', 'sparseOptFlow')

            self.tracker = BOTSORTTracker(
                track_high_thresh=track_high_thresh,
                track_low_thresh=track_low_thresh,
                new_track_thresh=new_track_thresh,
                match_thresh=match_thresh,
                track_buffer=track_buffer,
                camera_motion=camera_motion,
                cmc_method=cmc_method)

        else:
            # use ByteTracker
            use_byte = cfg.get('use_byte', False)
            det_thresh = cfg.get('det_thresh', 0.3)
            min_box_area = cfg.get('min_box_area', 0)
            vertical_ratio = cfg.get('vertical_ratio', 0)
            match_thres = cfg.get('match_thres', 0.9)
            conf_thres = cfg.get('conf_thres', 0.6)
            low_conf_thres = cfg.get('low_conf_thres', 0.1)

            self.tracker = JDETracker(
                use_byte=use_byte,
                det_thresh=det_thresh,
                num_classes=self.num_classes,
                min_box_area=min_box_area,
                vertical_ratio=vertical_ratio,
                match_thres=match_thres,
                conf_thres=conf_thres,
                low_conf_thres=low_conf_thres, )

        self.do_mtmct = False if mtmct_dir is None else True
        self.mtmct_dir = mtmct_dir

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        keep_idx = result['boxes'][:, 1] > self.threshold
        result['boxes'] = result['boxes'][keep_idx]
        np_boxes_num = [len(result['boxes'])]
        if np_boxes_num[0] <= 0:
            print('[WARNNING] No object detected.')
            result = {'boxes': np.zeros([0, 6]), 'boxes_num': [0]}
        result = {k: v for k, v in result.items() if v is not None}
        return result

    def reidprocess(self, det_results, repeats=1):
        pred_dets = det_results['boxes']  # cls_id, score, x0, y0, x1, y1
        pred_xyxys = pred_dets[:, 2:6]

        ori_image = det_results['ori_image']
        ori_image_shape = ori_image.shape[:2]
        pred_xyxys, keep_idx = clip_box(pred_xyxys, ori_image_shape)

        if len(keep_idx[0]) == 0:
            det_results['boxes'] = np.zeros((1, 6), dtype=np.float32)
            det_results['embeddings'] = None
            return det_results

        pred_dets = pred_dets[keep_idx[0]]
        pred_xyxys = pred_dets[:, 2:6]

        w, h = self.tracker.input_size
        crops = get_crops(pred_xyxys, ori_image, w, h)

        # to keep fast speed, only use topk crops
        crops = crops[:50]  # reid_batch_size
        det_results['crops'] = np.array(crops).astype('float32')
        det_results['boxes'] = pred_dets[:50]

        input_names = self.reid_predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.reid_predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(det_results[input_names[i]])

        # model prediction
        for i in range(repeats):
            self.reid_predictor.run()
            output_names = self.reid_predictor.get_output_names()
            feature_tensor = self.reid_predictor.get_output_handle(output_names[
                0])
            pred_embs = feature_tensor.copy_to_cpu()

        det_results['embeddings'] = pred_embs
        return det_results

    def tracking(self, det_results, img=None):
        pred_dets = det_results['boxes']  # cls_id, score, x0, y0, x1, y1
        pred_embs = det_results.get('embeddings', None)

        if self.use_deepsort_tracker:
            # use DeepSORTTracker, only support singe class
            self.tracker.predict()
            online_targets = self.tracker.update(pred_dets, pred_embs)
            online_tlwhs, online_scores, online_ids = [], [], []
            if self.do_mtmct:
                online_tlbrs, online_feats = [], []
            for t in online_targets:
                if not t.is_confirmed() or t.time_since_update > 1:
                    continue
                tlwh = t.to_tlwh()
                tscore = t.score
                tid = t.track_id
                if self.tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                        3] > self.tracker.vertical_ratio:
                    continue
                online_tlwhs.append(tlwh)
                online_scores.append(tscore)
                online_ids.append(tid)
                if self.do_mtmct:
                    online_tlbrs.append(t.to_tlbr())
                    online_feats.append(t.feat)

            tracking_outs = {
                'online_tlwhs': online_tlwhs,
                'online_scores': online_scores,
                'online_ids': online_ids,
            }
            if self.do_mtmct:
                seq_name = det_results['seq_name']
                frame_id = det_results['frame_id']

                tracking_outs['feat_data'] = {}
                for _tlbr, _id, _feat in zip(online_tlbrs, online_ids,
                                             online_feats):
                    feat_data = {}
                    feat_data['bbox'] = _tlbr
                    feat_data['frame'] = f"{frame_id:06d}"
                    feat_data['id'] = _id
                    _imgname = f'{seq_name}_{_id}_{frame_id}.jpg'
                    feat_data['imgname'] = _imgname
                    feat_data['feat'] = _feat
                    tracking_outs['feat_data'].update({_imgname: feat_data})
            return tracking_outs

        elif self.use_ocsort_tracker:
            # use OCSORTTracker, only support singe class
            online_targets = self.tracker.update(pred_dets, pred_embs)
            online_tlwhs = defaultdict(list)
            online_scores = defaultdict(list)
            online_ids = defaultdict(list)
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tscore = float(t[4])
                tid = int(t[5])
                if tlwh[2] * tlwh[3] <= self.tracker.min_box_area: continue
                if self.tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                        3] > self.tracker.vertical_ratio:
                    continue
                if tlwh[2] * tlwh[3] > 0:
                    online_tlwhs[0].append(tlwh)
                    online_ids[0].append(tid)
                    online_scores[0].append(tscore)
            tracking_outs = {
                'online_tlwhs': online_tlwhs,
                'online_scores': online_scores,
                'online_ids': online_ids,
            }
            return tracking_outs

        elif self.use_botsort_tracker:
            # use BOTSORTTracker, only support singe class
            online_targets = self.tracker.update(pred_dets, img)
            online_tlwhs = defaultdict(list)
            online_scores = defaultdict(list)
            online_ids = defaultdict(list)
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                tscore = t.score
                if tlwh[2] * tlwh[3] <= self.tracker.min_box_area:
                    continue
                online_tlwhs[0].append(tlwh)
                online_ids[0].append(tid)
                online_scores[0].append(tscore)

            tracking_outs = {
                'online_tlwhs': online_tlwhs,
                'online_scores': online_scores,
                'online_ids': online_ids,
            }
            return tracking_outs

        else:
            # use ByteTracker, support multiple class
            online_tlwhs = defaultdict(list)
            online_scores = defaultdict(list)
            online_ids = defaultdict(list)
            if self.do_mtmct:
                online_tlbrs, online_feats = defaultdict(list), defaultdict(
                    list)
            online_targets_dict = self.tracker.update(pred_dets, pred_embs)
            for cls_id in range(self.num_classes):
                online_targets = online_targets_dict[cls_id]
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    tscore = t.score
                    if tlwh[2] * tlwh[3] <= self.tracker.min_box_area:
                        continue
                    if self.tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                            3] > self.tracker.vertical_ratio:
                        continue
                    online_tlwhs[cls_id].append(tlwh)
                    online_ids[cls_id].append(tid)
                    online_scores[cls_id].append(tscore)
                    if self.do_mtmct:
                        online_tlbrs[cls_id].append(t.tlbr)
                        online_feats[cls_id].append(t.curr_feat)

            if self.do_mtmct:
                assert self.num_classes == 1, 'MTMCT only support single class.'
                tracking_outs = {
                    'online_tlwhs': online_tlwhs[0],
                    'online_scores': online_scores[0],
                    'online_ids': online_ids[0],
                }
                seq_name = det_results['seq_name']
                frame_id = det_results['frame_id']
                tracking_outs['feat_data'] = {}
                for _tlbr, _id, _feat in zip(online_tlbrs[0], online_ids[0],
                                             online_feats[0]):
                    feat_data = {}
                    feat_data['bbox'] = _tlbr
                    feat_data['frame'] = f"{frame_id:06d}"
                    feat_data['id'] = _id
                    _imgname = f'{seq_name}_{_id}_{frame_id}.jpg'
                    feat_data['imgname'] = _imgname
                    feat_data['feat'] = _feat
                    tracking_outs['feat_data'].update({_imgname: feat_data})
                return tracking_outs

            else:
                tracking_outs = {
                    'online_tlwhs': online_tlwhs,
                    'online_scores': online_scores,
                    'online_ids': online_ids,
                }
                return tracking_outs

    def predict_image(self,
                      image_list,
                      run_benchmark=False,
                      repeats=1,
                      visual=True,
                      seq_name=None,
                      reuse_det_result=False,
                      frame_count=0):
        num_classes = self.num_classes
        image_list.sort()
        ids2names = self.pred_config.labels
        if self.do_mtmct:
            mot_features_dict = {}  # cid_tid_fid feats
        else:
            mot_results = []
        for frame_id, img_file in enumerate(image_list):
            if self.do_mtmct:
                if frame_id % 10 == 0:
                    print('Tracking frame: %d' % (frame_id))
            batch_image_list = [img_file]  # bs=1 in MOT model
            frame, _ = decode_image(img_file, {})
            if run_benchmark:
                # preprocess
                inputs = self.preprocess(batch_image_list)  # warmup
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                result_warmup = self.predict(repeats=repeats)  # warmup
                self.det_times.inference_time_s.start()
                result = self.predict(repeats=repeats)
                self.det_times.inference_time_s.end(repeats=repeats)

                # postprocess
                result_warmup = self.postprocess(inputs, result)  # warmup
                self.det_times.postprocess_time_s.start()
                det_result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()

                # tracking
                if self.use_reid:
                    det_result['frame_id'] = frame_id
                    det_result['seq_name'] = seq_name
                    det_result['ori_image'] = frame
                    det_result = self.reidprocess(det_result)
                if self.use_botsort_tracker:
                    result_warmup = self.tracking(det_result, batch_image_list)
                else:
                    result_warmup = self.tracking(det_result)
                self.det_times.tracking_time_s.start()
                if self.use_reid:
                    det_result = self.reidprocess(det_result)
                tracking_outs = self.tracking(det_result)
                self.det_times.tracking_time_s.end()
                self.det_times.img_num += 1

                cm, gm, gu = get_current_memory_mb()
                self.cpu_mem += cm
                self.gpu_mem += gm
                self.gpu_util += gu

            else:
                if frame_count > self.warmup_frame:
                    self.det_times.preprocess_time_s.start()
                if not reuse_det_result:
                    inputs = self.preprocess(batch_image_list)
                if frame_count > self.warmup_frame:
                    self.det_times.preprocess_time_s.end()
                if frame_count > self.warmup_frame:
                    self.det_times.inference_time_s.start()
                if not reuse_det_result:
                    result = self.predict()
                if frame_count > self.warmup_frame:
                    self.det_times.inference_time_s.end()
                if frame_count > self.warmup_frame:
                    self.det_times.postprocess_time_s.start()
                if not reuse_det_result:
                    det_result = self.postprocess(inputs, result)
                    self.previous_det_result = det_result
                else:
                    assert self.previous_det_result is not None
                    det_result = self.previous_det_result
                if frame_count > self.warmup_frame:
                    self.det_times.postprocess_time_s.end()

                # tracking process
                if frame_count > self.warmup_frame:
                    self.det_times.tracking_time_s.start()
                if self.use_reid:
                    det_result['frame_id'] = frame_id
                    det_result['seq_name'] = seq_name
                    det_result['ori_image'] = frame
                    det_result = self.reidprocess(det_result)
                if self.use_botsort_tracker:
                    tracking_outs = self.tracking(det_result, batch_image_list)
                else:
                    tracking_outs = self.tracking(det_result)
                if frame_count > self.warmup_frame:
                    self.det_times.tracking_time_s.end()
                    self.det_times.img_num += 1

            online_tlwhs = tracking_outs['online_tlwhs']
            online_scores = tracking_outs['online_scores']
            online_ids = tracking_outs['online_ids']

            if self.do_mtmct:
                feat_data_dict = tracking_outs['feat_data']
                mot_features_dict = dict(mot_features_dict, **feat_data_dict)
            else:
                mot_results.append([online_tlwhs, online_scores, online_ids])

            if visual:
                if len(image_list) > 1 and frame_id % 10 == 0:
                    print('Tracking frame {}'.format(frame_id))
                frame, _ = decode_image(img_file, {})
                if isinstance(online_tlwhs, defaultdict):
                    im = plot_tracking_dict(
                        frame,
                        num_classes,
                        online_tlwhs,
                        online_ids,
                        online_scores,
                        frame_id=frame_id,
                        ids2names=ids2names)
                else:
                    im = plot_tracking(
                        frame,
                        online_tlwhs,
                        online_ids,
                        online_scores,
                        frame_id=frame_id,
                        ids2names=ids2names)
                save_dir = os.path.join(self.output_dir, seq_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(
                    os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), im)

        if self.do_mtmct:
            return mot_features_dict
        else:
            return mot_results

    def predict_video(self, video_file, camera_id):
        video_out_name = 'output.mp4'
        if camera_id != -1:
            capture = cv2.VideoCapture(camera_id)
        else:
            capture = cv2.VideoCapture(video_file)
            video_out_name = os.path.split(video_file)[-1]
        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("fps: %d, frame_count: %d" % (fps, frame_count))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        out_path = os.path.join(self.output_dir, video_out_name)
        video_format = 'mp4v'
        fourcc = cv2.VideoWriter_fourcc(*video_format)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_id = 0
        timer = MOTTimer()
        results = defaultdict(list)
        num_classes = self.num_classes
        data_type = 'mcmot' if num_classes > 1 else 'mot'
        ids2names = self.pred_config.labels

        center_traj = None
        entrance = None
        records = None
        if self.draw_center_traj:
            center_traj = [{} for i in range(num_classes)]
        if num_classes == 1:
            id_set = set()
            interval_id_set = set()
            in_id_list = list()
            out_id_list = list()
            prev_center = dict()
            records = list()
            if self.do_entrance_counting or self.do_break_in_counting:
                if self.region_type == 'horizontal':
                    entrance = [0, height / 2., width, height / 2.]
                elif self.region_type == 'vertical':
                    entrance = [width / 2, 0., width / 2, height]
                elif self.region_type == 'custom':
                    entrance = []
                    assert len(
                        self.region_polygon
                    ) % 2 == 0, "region_polygon should be pairs of coords points when do break_in counting."
                    for i in range(0, len(self.region_polygon), 2):
                        entrance.append([
                            self.region_polygon[i], self.region_polygon[i + 1]
                        ])
                    entrance.append([width, height])
                else:
                    raise ValueError("region_type:{} is not supported.".format(
                        self.region_type))

        video_fps = fps

        while (1):
            ret, frame = capture.read()
            if not ret:
                break
            if frame_id % 10 == 0:
                print('Tracking frame: %d' % (frame_id))

            timer.tic()
            mot_skip_frame_num = self.skip_frame_num
            reuse_det_result = False
            if mot_skip_frame_num > 1 and frame_id > 0 and frame_id % mot_skip_frame_num > 0:
                reuse_det_result = True
            seq_name = video_out_name.split('.')[0]
            mot_results = self.predict_image(
                [frame],
                visual=False,
                seq_name=seq_name,
                reuse_det_result=reuse_det_result,
                frame_count=frame_id)
            timer.toc()

            # bs=1 in MOT model
            online_tlwhs, online_scores, online_ids = mot_results[0]

            # flow statistic for one class, and only for bytetracker
            if num_classes == 1 and not self.use_deepsort_tracker and not self.use_ocsort_tracker:
                result = (frame_id + 1, online_tlwhs[0], online_scores[0],
                          online_ids[0])
                statistic = flow_statistic(
                    result,
                    self.secs_interval,
                    self.do_entrance_counting,
                    self.do_break_in_counting,
                    self.region_type,
                    video_fps,
                    entrance,
                    id_set,
                    interval_id_set,
                    in_id_list,
                    out_id_list,
                    prev_center,
                    records,
                    data_type,
                    ids2names=self.pred_config.labels)
                records = statistic['records']

            fps = 1. / timer.duration
            if self.use_deepsort_tracker or self.use_ocsort_tracker or self.use_botsort_tracker:
                # use DeepSORTTracker or OCSORTTracker, only support singe class
                if isinstance(online_tlwhs, defaultdict):
                    online_tlwhs = online_tlwhs[0]
                    online_scores = online_scores[0]
                    online_ids = online_ids[0]

                results[0].append(
                    (frame_id + 1, online_tlwhs, online_scores, online_ids))
                im = plot_tracking(
                    frame,
                    online_tlwhs,
                    online_ids,
                    online_scores,
                    frame_id=frame_id,
                    fps=fps,
                    ids2names=ids2names,
                    do_entrance_counting=self.do_entrance_counting,
                    entrance=entrance)
            else:
                # use ByteTracker, support multiple class
                for cls_id in range(num_classes):
                    results[cls_id].append(
                        (frame_id + 1, online_tlwhs[cls_id],
                         online_scores[cls_id], online_ids[cls_id]))
                im = plot_tracking_dict(
                    frame,
                    num_classes,
                    online_tlwhs,
                    online_ids,
                    online_scores,
                    frame_id=frame_id,
                    fps=fps,
                    ids2names=ids2names,
                    do_entrance_counting=self.do_entrance_counting,
                    entrance=entrance,
                    records=records,
                    center_traj=center_traj)

            writer.write(im)
            if camera_id != -1:
                cv2.imshow('Mask Detection', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_id += 1

        if self.save_mot_txts:
            result_filename = os.path.join(
                self.output_dir, video_out_name.split('.')[-2] + '.txt')
            write_mot_results(result_filename, results)

            result_filename = os.path.join(
                self.output_dir,
                video_out_name.split('.')[-2] + '_flow_statistic.txt')
            f = open(result_filename, 'w')
            for line in records:
                f.write(line)
            print('Flow statistic save in {}'.format(result_filename))
            f.close()

        writer.release()

    def predict_mtmct(self, mtmct_dir, mtmct_cfg):
        cameras_bias = mtmct_cfg['cameras_bias']
        cid_bias = parse_bias(cameras_bias)
        scene_cluster = list(cid_bias.keys())
        # 1.zone releated parameters
        use_zone = mtmct_cfg.get('use_zone', False)
        zone_path = mtmct_cfg.get('zone_path', None)

        # 2.tricks parameters, can be used for other mtmct dataset
        use_ff = mtmct_cfg.get('use_ff', False)
        use_rerank = mtmct_cfg.get('use_rerank', False)

        # 3.camera releated parameters
        use_camera = mtmct_cfg.get('use_camera', False)
        use_st_filter = mtmct_cfg.get('use_st_filter', False)

        # 4.zone releated parameters
        use_roi = mtmct_cfg.get('use_roi', False)
        roi_dir = mtmct_cfg.get('roi_dir', False)

        mot_list_breaks = []
        cid_tid_dict = dict()

        output_dir = self.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        seqs = os.listdir(mtmct_dir)
        for seq in sorted(seqs):
            fpath = os.path.join(mtmct_dir, seq)
            if os.path.isfile(fpath) and _is_valid_video(fpath):
                seq = seq.split('.')[-2]
                print('ffmpeg processing of video {}'.format(fpath))
                frames_path = video2frames(
                    video_path=fpath, outpath=mtmct_dir, frame_rate=25)
                fpath = os.path.join(mtmct_dir, seq)

            if os.path.isdir(fpath) == False:
                print('{} is not a image folder.'.format(fpath))
                continue
            if os.path.exists(os.path.join(fpath, 'img1')):
                fpath = os.path.join(fpath, 'img1')
            assert os.path.isdir(fpath), '{} should be a directory'.format(
                fpath)
            image_list = glob.glob(os.path.join(fpath, '*.jpg'))
            image_list.sort()
            assert len(image_list) > 0, '{} has no images.'.format(fpath)
            print('start tracking seq: {}'.format(seq))

            mot_features_dict = self.predict_image(
                image_list, visual=False, seq_name=seq)

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


def main():
    deploy_file = os.path.join(FLAGS.model_dir, 'infer_cfg.yml')
    with open(deploy_file) as f:
        yml_conf = yaml.safe_load(f)
    arch = yml_conf['arch']
    detector = SDE_Detector(
        FLAGS.model_dir,
        tracker_config=FLAGS.tracker_config,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        batch_size=1,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn,
        output_dir=FLAGS.output_dir,
        threshold=FLAGS.threshold,
        save_images=FLAGS.save_images,
        save_mot_txts=FLAGS.save_mot_txts,
        draw_center_traj=FLAGS.draw_center_traj,
        secs_interval=FLAGS.secs_interval,
        skip_frame_num=FLAGS.skip_frame_num,
        warmup_frame=FLAGS.warmup_frame,
        do_entrance_counting=FLAGS.do_entrance_counting,
        do_break_in_counting=FLAGS.do_break_in_counting,
        region_type=FLAGS.region_type,
        region_polygon=FLAGS.region_polygon,
        reid_model_dir=FLAGS.reid_model_dir,
        mtmct_dir=FLAGS.mtmct_dir, )

    # predict from video file or camera video stream
    if FLAGS.video_file is not None or FLAGS.camera_id != -1:
        detector.predict_video(FLAGS.video_file, FLAGS.camera_id)
        detector.det_times.info(average=True)
    elif FLAGS.mtmct_dir is not None:
        with open(FLAGS.mtmct_cfg) as f:
            mtmct_cfg = yaml.safe_load(f)
        detector.predict_mtmct(FLAGS.mtmct_dir, mtmct_cfg)
    else:
        # predict from image
        if FLAGS.image_dir is None and FLAGS.image_file is not None:
            assert FLAGS.batch_size == 1, "--batch_size should be 1 in MOT models."
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        seq_name = FLAGS.image_dir.split('/')[-1]
        detector.predict_image(
            img_list, FLAGS.run_benchmark, repeats=10, seq_name=seq_name)

        if not FLAGS.run_benchmark:
            detector.det_times.info(average=True)
        else:
            mode = FLAGS.run_mode
            model_dir = FLAGS.model_dir
            model_info = {
                'model_name': model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(detector, img_list, model_info, name='MOT')


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()
