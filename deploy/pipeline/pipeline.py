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
import yaml
import glob
import cv2
import numpy as np
import math
import paddle
import sys
import copy
import threading
import queue
import time
from collections import defaultdict
from datacollector import DataCollector, Result
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

# add deploy path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from cfg_utils import argsparser, print_arguments, merge_cfg
from pipe_utils import PipeTimer
from pipe_utils import get_test_images, crop_image_with_det, crop_image_with_mot, parse_mot_res, parse_mot_keypoint
from pipe_utils import PushStream

from python.infer import Detector, DetectorPicoDet
from python.keypoint_infer import KeyPointDetector
from python.keypoint_postprocess import translate_to_ori_images
from python.preprocess import decode_image, ShortSizeScale
from python.visualize import visualize_box_mask, visualize_attr, visualize_pose, visualize_action, visualize_vehicleplate, visualize_vehiclepress, visualize_lane, visualize_vehicle_retrograde

from pptracking.python.mot_sde_infer import SDE_Detector
from pptracking.python.mot.visualize import plot_tracking_dict
from pptracking.python.mot.utils import flow_statistic, update_object_info

from pphuman.attr_infer import AttrDetector
from pphuman.video_action_infer import VideoActionRecognizer
from pphuman.action_infer import SkeletonActionRecognizer, DetActionRecognizer, ClsActionRecognizer
from pphuman.action_utils import KeyPointBuff, ActionVisualHelper
from pphuman.reid import ReID
from pphuman.mtmct import mtmct_process

from ppvehicle.vehicle_plate import PlateRecognizer
from ppvehicle.vehicle_attr import VehicleAttr
from ppvehicle.vehicle_pressing import VehiclePressingRecognizer
from ppvehicle.vehicle_retrograde import VehicleRetrogradeRecognizer
from ppvehicle.lane_seg_infer import LaneSegPredictor

from download import auto_download_model


class Pipeline(object):
    """
    Pipeline

    Args:
        args (argparse.Namespace): arguments in pipeline, which contains environment and runtime settings
        cfg (dict): config of models in pipeline
    """

    def __init__(self, args, cfg):
        self.multi_camera = False
        reid_cfg = cfg.get('REID', False)
        self.enable_mtmct = reid_cfg['enable'] if reid_cfg else False
        self.is_video = False
        self.output_dir = args.output_dir
        self.vis_result = cfg['visual']
        self.input = self._parse_input(args.image_file, args.image_dir,
                                       args.video_file, args.video_dir,
                                       args.camera_id, args.rtsp)
        if self.multi_camera:
            self.predictor = []
            for name in self.input:
                predictor_item = PipePredictor(
                    args, cfg, is_video=True, multi_camera=True)
                predictor_item.set_file_name(name)
                self.predictor.append(predictor_item)

        else:
            self.predictor = PipePredictor(args, cfg, self.is_video)
            if self.is_video:
                self.predictor.set_file_name(self.input)

    def _parse_input(self, image_file, image_dir, video_file, video_dir,
                     camera_id, rtsp):

        # parse input as is_video and multi_camera

        if image_file is not None or image_dir is not None:
            input = get_test_images(image_dir, image_file)
            self.is_video = False
            self.multi_camera = False

        elif video_file is not None:
            assert os.path.exists(
                video_file
            ) or 'rtsp' in video_file, "video_file not exists and not an rtsp site."
            self.multi_camera = False
            input = video_file
            self.is_video = True

        elif video_dir is not None:
            videof = [os.path.join(video_dir, x) for x in os.listdir(video_dir)]
            if len(videof) > 1:
                self.multi_camera = True
                videof.sort()
                input = videof
            else:
                input = videof[0]
            self.is_video = True

        elif rtsp is not None:
            if len(rtsp) > 1:
                rtsp = [rtsp_item for rtsp_item in rtsp if 'rtsp' in rtsp_item]
                self.multi_camera = True
                input = rtsp
            else:
                self.multi_camera = False
                input = rtsp[0]
            self.is_video = True

        elif camera_id != -1:
            self.multi_camera = False
            input = camera_id
            self.is_video = True

        else:
            raise ValueError(
                "Illegal Input, please set one of ['video_file', 'camera_id', 'image_file', 'image_dir']"
            )

        return input

    def run_multithreads(self):
        if self.multi_camera:
            multi_res = []
            threads = []
            for idx, (predictor,
                      input) in enumerate(zip(self.predictor, self.input)):
                thread = threading.Thread(
                    name=str(idx).zfill(3),
                    target=predictor.run,
                    args=(input, idx))
                threads.append(thread)

            for thread in threads:
                thread.start()

            for predictor, thread in zip(self.predictor, threads):
                thread.join()
                collector_data = predictor.get_result()
                multi_res.append(collector_data)

            if self.enable_mtmct:
                mtmct_process(
                    multi_res,
                    self.input,
                    mtmct_vis=self.vis_result,
                    output_dir=self.output_dir)

        else:
            self.predictor.run(self.input)

    def run(self):
        if self.multi_camera:
            multi_res = []
            for predictor, input in zip(self.predictor, self.input):
                predictor.run(input)
                collector_data = predictor.get_result()
                multi_res.append(collector_data)
            if self.enable_mtmct:
                mtmct_process(
                    multi_res,
                    self.input,
                    mtmct_vis=self.vis_result,
                    output_dir=self.output_dir)

        else:
            self.predictor.run(self.input)


def get_model_dir(cfg):
    """ 
        Auto download inference model if the model_path is a url link. 
        Otherwise it will use the model_path directly.
    """
    for key in cfg.keys():
        if type(cfg[key]) ==  dict and \
            ("enable" in cfg[key].keys() and cfg[key]['enable']
                or "enable" not in cfg[key].keys()):

            if "model_dir" in cfg[key].keys():
                model_dir = cfg[key]["model_dir"]
                downloaded_model_dir = auto_download_model(model_dir)
                if downloaded_model_dir:
                    model_dir = downloaded_model_dir
                    cfg[key]["model_dir"] = model_dir
                print(key, " model dir: ", model_dir)
            elif key == "VEHICLE_PLATE":
                det_model_dir = cfg[key]["det_model_dir"]
                downloaded_det_model_dir = auto_download_model(det_model_dir)
                if downloaded_det_model_dir:
                    det_model_dir = downloaded_det_model_dir
                    cfg[key]["det_model_dir"] = det_model_dir
                print("det_model_dir model dir: ", det_model_dir)

                rec_model_dir = cfg[key]["rec_model_dir"]
                downloaded_rec_model_dir = auto_download_model(rec_model_dir)
                if downloaded_rec_model_dir:
                    rec_model_dir = downloaded_rec_model_dir
                    cfg[key]["rec_model_dir"] = rec_model_dir
                print("rec_model_dir model dir: ", rec_model_dir)

        elif key == "MOT":  # for idbased and skeletonbased actions
            model_dir = cfg[key]["model_dir"]
            downloaded_model_dir = auto_download_model(model_dir)
            if downloaded_model_dir:
                model_dir = downloaded_model_dir
                cfg[key]["model_dir"] = model_dir
            print("mot_model_dir model_dir: ", model_dir)


class PipePredictor(object):
    """
    Predictor in single camera
    
    The pipeline for image input: 

        1. Detection
        2. Detection -> Attribute

    The pipeline for video input: 

        1. Tracking
        2. Tracking -> Attribute
        3. Tracking -> KeyPoint -> SkeletonAction Recognition
        4. VideoAction Recognition

    Args:
        args (argparse.Namespace): arguments in pipeline, which contains environment and runtime settings
        cfg (dict): config of models in pipeline
        is_video (bool): whether the input is video, default as False
        multi_camera (bool): whether to use multi camera in pipeline, 
            default as False
    """

    def __init__(self, args, cfg, is_video=True, multi_camera=False):
        # general module for pphuman and ppvehicle
        self.with_mot = cfg.get('MOT', False)['enable'] if cfg.get(
            'MOT', False) else False
        self.with_human_attr = cfg.get('ATTR', False)['enable'] if cfg.get(
            'ATTR', False) else False
        if self.with_mot:
            print('Multi-Object Tracking enabled')
        if self.with_human_attr:
            print('Human Attribute Recognition enabled')

        # only for pphuman
        self.with_skeleton_action = cfg.get(
            'SKELETON_ACTION', False)['enable'] if cfg.get('SKELETON_ACTION',
                                                           False) else False
        self.with_video_action = cfg.get(
            'VIDEO_ACTION', False)['enable'] if cfg.get('VIDEO_ACTION',
                                                        False) else False
        self.with_idbased_detaction = cfg.get(
            'ID_BASED_DETACTION', False)['enable'] if cfg.get(
                'ID_BASED_DETACTION', False) else False
        self.with_idbased_clsaction = cfg.get(
            'ID_BASED_CLSACTION', False)['enable'] if cfg.get(
                'ID_BASED_CLSACTION', False) else False
        self.with_mtmct = cfg.get('REID', False)['enable'] if cfg.get(
            'REID', False) else False

        if self.with_skeleton_action:
            print('SkeletonAction Recognition enabled')
        if self.with_video_action:
            print('VideoAction Recognition enabled')
        if self.with_idbased_detaction:
            print('IDBASED Detection Action Recognition enabled')
        if self.with_idbased_clsaction:
            print('IDBASED Classification Action Recognition enabled')
        if self.with_mtmct:
            print("MTMCT enabled")

        # only for ppvehicle
        self.with_vehicleplate = cfg.get(
            'VEHICLE_PLATE', False)['enable'] if cfg.get('VEHICLE_PLATE',
                                                         False) else False
        if self.with_vehicleplate:
            print('Vehicle Plate Recognition enabled')

        self.with_vehicle_attr = cfg.get(
            'VEHICLE_ATTR', False)['enable'] if cfg.get('VEHICLE_ATTR',
                                                        False) else False
        if self.with_vehicle_attr:
            print('Vehicle Attribute Recognition enabled')

        self.with_vehicle_press = cfg.get(
            'VEHICLE_PRESSING', False)['enable'] if cfg.get('VEHICLE_PRESSING',
                                                            False) else False
        if self.with_vehicle_press:
            print('Vehicle Pressing Recognition enabled')

        self.with_vehicle_retrograde = cfg.get(
            'VEHICLE_RETROGRADE', False)['enable'] if cfg.get(
                'VEHICLE_RETROGRADE', False) else False
        if self.with_vehicle_retrograde:
            print('Vehicle Retrograde Recognition enabled')

        self.modebase = {
            "framebased": False,
            "videobased": False,
            "idbased": False,
            "skeletonbased": False
        }

        self.basemode = {
            "MOT": "idbased",
            "ATTR": "idbased",
            "VIDEO_ACTION": "videobased",
            "SKELETON_ACTION": "skeletonbased",
            "ID_BASED_DETACTION": "idbased",
            "ID_BASED_CLSACTION": "idbased",
            "REID": "idbased",
            "VEHICLE_PLATE": "idbased",
            "VEHICLE_ATTR": "idbased",
            "VEHICLE_PRESSING": "idbased",
            "VEHICLE_RETROGRADE": "idbased",
        }

        self.is_video = is_video
        self.multi_camera = multi_camera
        self.cfg = cfg

        self.output_dir = args.output_dir
        self.draw_center_traj = args.draw_center_traj
        self.secs_interval = args.secs_interval
        self.do_entrance_counting = args.do_entrance_counting
        self.do_break_in_counting = args.do_break_in_counting
        self.region_type = args.region_type
        self.region_polygon = args.region_polygon
        self.illegal_parking_time = args.illegal_parking_time

        self.warmup_frame = self.cfg['warmup_frame']
        self.pipeline_res = Result()
        self.pipe_timer = PipeTimer()
        self.file_name = None
        self.collector = DataCollector()

        self.pushurl = args.pushurl

        # auto download inference model
        get_model_dir(self.cfg)

        if self.with_vehicleplate:
            vehicleplate_cfg = self.cfg['VEHICLE_PLATE']
            self.vehicleplate_detector = PlateRecognizer(args, vehicleplate_cfg)
            basemode = self.basemode['VEHICLE_PLATE']
            self.modebase[basemode] = True

        if self.with_human_attr:
            attr_cfg = self.cfg['ATTR']
            basemode = self.basemode['ATTR']
            self.modebase[basemode] = True
            self.attr_predictor = AttrDetector.init_with_cfg(args, attr_cfg)

        if self.with_vehicle_attr:
            vehicleattr_cfg = self.cfg['VEHICLE_ATTR']
            basemode = self.basemode['VEHICLE_ATTR']
            self.modebase[basemode] = True
            self.vehicle_attr_predictor = VehicleAttr.init_with_cfg(
                args, vehicleattr_cfg)

        if self.with_vehicle_press:
            vehiclepress_cfg = self.cfg['VEHICLE_PRESSING']
            basemode = self.basemode['VEHICLE_PRESSING']
            self.modebase[basemode] = True
            self.vehicle_press_predictor = VehiclePressingRecognizer(
                vehiclepress_cfg)

        if self.with_vehicle_press or self.with_vehicle_retrograde:
            laneseg_cfg = self.cfg['LANE_SEG']
            self.laneseg_predictor = LaneSegPredictor(
                laneseg_cfg['lane_seg_config'], laneseg_cfg['model_dir'])

        if not is_video:

            det_cfg = self.cfg['DET']
            model_dir = det_cfg['model_dir']
            batch_size = det_cfg['batch_size']
            self.det_predictor = Detector(
                model_dir, args.device, args.run_mode, batch_size,
                args.trt_min_shape, args.trt_max_shape, args.trt_opt_shape,
                args.trt_calib_mode, args.cpu_threads, args.enable_mkldnn)
        else:
            if self.with_idbased_detaction:
                idbased_detaction_cfg = self.cfg['ID_BASED_DETACTION']
                basemode = self.basemode['ID_BASED_DETACTION']
                self.modebase[basemode] = True

                self.det_action_predictor = DetActionRecognizer.init_with_cfg(
                    args, idbased_detaction_cfg)
                self.det_action_visual_helper = ActionVisualHelper(1)

            if self.with_idbased_clsaction:
                idbased_clsaction_cfg = self.cfg['ID_BASED_CLSACTION']
                basemode = self.basemode['ID_BASED_CLSACTION']
                self.modebase[basemode] = True

                self.cls_action_predictor = ClsActionRecognizer.init_with_cfg(
                    args, idbased_clsaction_cfg)
                self.cls_action_visual_helper = ActionVisualHelper(1)

            if self.with_skeleton_action:
                skeleton_action_cfg = self.cfg['SKELETON_ACTION']
                display_frames = skeleton_action_cfg['display_frames']
                self.coord_size = skeleton_action_cfg['coord_size']
                basemode = self.basemode['SKELETON_ACTION']
                self.modebase[basemode] = True
                skeleton_action_frames = skeleton_action_cfg['max_frames']

                self.skeleton_action_predictor = SkeletonActionRecognizer.init_with_cfg(
                    args, skeleton_action_cfg)
                self.skeleton_action_visual_helper = ActionVisualHelper(
                    display_frames)

                kpt_cfg = self.cfg['KPT']
                kpt_model_dir = kpt_cfg['model_dir']
                kpt_batch_size = kpt_cfg['batch_size']
                self.kpt_predictor = KeyPointDetector(
                    kpt_model_dir,
                    args.device,
                    args.run_mode,
                    kpt_batch_size,
                    args.trt_min_shape,
                    args.trt_max_shape,
                    args.trt_opt_shape,
                    args.trt_calib_mode,
                    args.cpu_threads,
                    args.enable_mkldnn,
                    use_dark=False)
                self.kpt_buff = KeyPointBuff(skeleton_action_frames)

            if self.with_vehicleplate:
                vehicleplate_cfg = self.cfg['VEHICLE_PLATE']
                self.vehicleplate_detector = PlateRecognizer(args,
                                                             vehicleplate_cfg)
                basemode = self.basemode['VEHICLE_PLATE']
                self.modebase[basemode] = True

            if self.with_mtmct:
                reid_cfg = self.cfg['REID']
                basemode = self.basemode['REID']
                self.modebase[basemode] = True
                self.reid_predictor = ReID.init_with_cfg(args, reid_cfg)

            if self.with_vehicle_retrograde:
                vehicleretrograde_cfg = self.cfg['VEHICLE_RETROGRADE']
                basemode = self.basemode['VEHICLE_RETROGRADE']
                self.modebase[basemode] = True
                self.vehicle_retrograde_predictor = VehicleRetrogradeRecognizer(
                    vehicleretrograde_cfg)

            if self.with_mot or self.modebase["idbased"] or self.modebase[
                    "skeletonbased"]:
                mot_cfg = self.cfg['MOT']
                model_dir = mot_cfg['model_dir']
                tracker_config = mot_cfg['tracker_config']
                batch_size = mot_cfg['batch_size']
                skip_frame_num = mot_cfg.get('skip_frame_num', -1)
                basemode = self.basemode['MOT']
                self.modebase[basemode] = True
                self.mot_predictor = SDE_Detector(
                    model_dir,
                    tracker_config,
                    args.device,
                    args.run_mode,
                    batch_size,
                    args.trt_min_shape,
                    args.trt_max_shape,
                    args.trt_opt_shape,
                    args.trt_calib_mode,
                    args.cpu_threads,
                    args.enable_mkldnn,
                    skip_frame_num=skip_frame_num,
                    draw_center_traj=self.draw_center_traj,
                    secs_interval=self.secs_interval,
                    do_entrance_counting=self.do_entrance_counting,
                    do_break_in_counting=self.do_break_in_counting,
                    region_type=self.region_type,
                    region_polygon=self.region_polygon)

            if self.with_video_action:
                video_action_cfg = self.cfg['VIDEO_ACTION']
                basemode = self.basemode['VIDEO_ACTION']
                self.modebase[basemode] = True
                self.video_action_predictor = VideoActionRecognizer.init_with_cfg(
                    args, video_action_cfg)

    def set_file_name(self, path):
        if type(path) == int:
            self.file_name = path
        elif path is not None:
            self.file_name = os.path.split(path)[-1]
            if "." in self.file_name:
                self.file_name = self.file_name.split(".")[-2]
        else:
            # use camera id
            self.file_name = None

    def get_result(self):
        return self.collector.get_res()

    def run(self, input, thread_idx=0):
        if self.is_video:
            self.predict_video(input, thread_idx=thread_idx)
        else:
            self.predict_image(input)
        self.pipe_timer.info()
        if hasattr(self, 'mot_predictor'):
            self.mot_predictor.det_times.tracking_info(average=True)

    def predict_image(self, input):
        # det
        # det -> attr
        batch_loop_cnt = math.ceil(
            float(len(input)) / self.det_predictor.batch_size)
        self.warmup_frame = min(10, len(input) // 2) - 1
        for i in range(batch_loop_cnt):
            start_index = i * self.det_predictor.batch_size
            end_index = min((i + 1) * self.det_predictor.batch_size, len(input))
            batch_file = input[start_index:end_index]
            batch_input = [decode_image(f, {})[0] for f in batch_file]

            if i > self.warmup_frame:
                self.pipe_timer.total_time.start()
                self.pipe_timer.module_time['det'].start()
            # det output format: class, score, xmin, ymin, xmax, ymax
            det_res = self.det_predictor.predict_image(
                batch_input, visual=False)
            det_res = self.det_predictor.filter_box(det_res,
                                                    self.cfg['crop_thresh'])
            if i > self.warmup_frame:
                self.pipe_timer.module_time['det'].end()
                self.pipe_timer.track_num += len(det_res['boxes'])
            self.pipeline_res.update(det_res, 'det')

            if self.with_human_attr:
                crop_inputs = crop_image_with_det(batch_input, det_res)
                attr_res_list = []

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['attr'].start()

                for crop_input in crop_inputs:
                    attr_res = self.attr_predictor.predict_image(
                        crop_input, visual=False)
                    attr_res_list.extend(attr_res['output'])

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['attr'].end()

                attr_res = {'output': attr_res_list}
                self.pipeline_res.update(attr_res, 'attr')

            if self.with_vehicle_attr:
                crop_inputs = crop_image_with_det(batch_input, det_res)
                vehicle_attr_res_list = []

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicle_attr'].start()

                for crop_input in crop_inputs:
                    attr_res = self.vehicle_attr_predictor.predict_image(
                        crop_input, visual=False)
                    vehicle_attr_res_list.extend(attr_res['output'])

                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicle_attr'].end()

                attr_res = {'output': vehicle_attr_res_list}
                self.pipeline_res.update(attr_res, 'vehicle_attr')

            if self.with_vehicleplate:
                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicleplate'].start()
                crop_inputs = crop_image_with_det(batch_input, det_res)
                platelicenses = []
                for crop_input in crop_inputs:
                    platelicense = self.vehicleplate_detector.get_platelicense(
                        crop_input)
                    platelicenses.extend(platelicense['plate'])
                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicleplate'].end()
                vehicleplate_res = {'vehicleplate': platelicenses}
                self.pipeline_res.update(vehicleplate_res, 'vehicleplate')

            if self.with_vehicle_press:
                vehicle_press_res_list = []
                if i > self.warmup_frame:
                    self.pipe_timer.module_time['vehicle_press'].start()

                lanes, direction = self.laneseg_predictor.run(batch_input)
                if len(lanes) == 0:
                    print(" no lanes!")
                    continue

                lanes_res = {'output': lanes, 'direction': direction}
                self.pipeline_res.update(lanes_res, 'lanes')

                vehicle_press_res_list = self.vehicle_press_predictor.run(
                    lanes, det_res)
                vehiclepress_res = {'output': vehicle_press_res_list}
                self.pipeline_res.update(vehiclepress_res, 'vehicle_press')

            self.pipe_timer.img_num += len(batch_input)
            if i > self.warmup_frame:
                self.pipe_timer.total_time.end()

            if self.cfg['visual']:
                self.visualize_image(batch_file, batch_input, self.pipeline_res)

    def capturevideo(self, capture, queue):
        frame_id = 0
        while (1):
            if queue.full():
                time.sleep(0.1)
            else:
                ret, frame = capture.read()
                if not ret:
                    return
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                queue.put(frame_rgb)

    def predict_video(self, video_file, thread_idx=0):
        # mot
        # mot -> attr
        # mot -> pose -> action
        capture = cv2.VideoCapture(video_file)

        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("video fps: %d, frame_count: %d" % (fps, frame_count))

        if len(self.pushurl) > 0:
            video_out_name = 'output' if self.file_name is None else self.file_name
            pushurl = os.path.join(self.pushurl, video_out_name)
            print("the result will push stream to url:{}".format(pushurl))
            pushstream = PushStream(pushurl)
            pushstream.initcmd(fps, width, height)
        elif self.cfg['visual']:
            video_out_name = 'output' if (
                self.file_name is None or
                type(self.file_name) == int) else self.file_name
            if type(video_file) == str and "rtsp" in video_file:
                video_out_name = video_out_name + "_t" + str(thread_idx).zfill(
                    2) + "_rtsp"
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            out_path = os.path.join(self.output_dir, video_out_name + ".mp4")
            fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_id = 0

        entrance, records, center_traj = None, None, None
        if self.draw_center_traj:
            center_traj = [{}]
        id_set = set()
        interval_id_set = set()
        in_id_list = list()
        out_id_list = list()
        prev_center = dict()
        records = list()
        if self.do_entrance_counting or self.do_break_in_counting or self.illegal_parking_time != -1:
            if self.region_type == 'horizontal':
                entrance = [0, height / 2., width, height / 2.]
            elif self.region_type == 'vertical':
                entrance = [width / 2, 0., width / 2, height]
            elif self.region_type == 'custom':
                entrance = []
                assert len(
                    self.region_polygon
                ) % 2 == 0, "region_polygon should be pairs of coords points when do break_in counting."
                assert len(
                    self.region_polygon
                ) > 6, 'region_type is custom, region_polygon should be at least 3 pairs of point coords.'

                for i in range(0, len(self.region_polygon), 2):
                    entrance.append(
                        [self.region_polygon[i], self.region_polygon[i + 1]])
                entrance.append([width, height])
            else:
                raise ValueError("region_type:{} unsupported.".format(
                    self.region_type))

        video_fps = fps

        video_action_imgs = []

        if self.with_video_action:
            short_size = self.cfg["VIDEO_ACTION"]["short_size"]
            scale = ShortSizeScale(short_size)

        object_in_region_info = {
        }  # store info for vehicle parking in region       
        illegal_parking_dict = None
        cars_count = 0
        retrograde_traj_len = 0
        framequeue = queue.Queue(10)

        thread = threading.Thread(
            target=self.capturevideo, args=(capture, framequeue))
        thread.start()
        time.sleep(1)

        while (not framequeue.empty()):
            if frame_id % 10 == 0:
                print('Thread: {}; frame id: {}'.format(thread_idx, frame_id))

            frame_rgb = framequeue.get()
            if frame_id > self.warmup_frame:
                self.pipe_timer.total_time.start()

            if self.modebase["idbased"] or self.modebase["skeletonbased"]:
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['mot'].start()

                mot_skip_frame_num = self.mot_predictor.skip_frame_num
                reuse_det_result = False
                if mot_skip_frame_num > 1 and frame_id > 0 and frame_id % mot_skip_frame_num > 0:
                    reuse_det_result = True
                res = self.mot_predictor.predict_image(
                    [copy.deepcopy(frame_rgb)],
                    visual=False,
                    reuse_det_result=reuse_det_result,
                    frame_count=frame_id)

                # mot output format: id, class, score, xmin, ymin, xmax, ymax
                mot_res = parse_mot_res(res)
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['mot'].end()
                    self.pipe_timer.track_num += len(mot_res['boxes'])

                if frame_id % 10 == 0:
                    print("Thread: {}; trackid number: {}".format(
                        thread_idx, len(mot_res['boxes'])))

                # flow_statistic only support single class MOT
                boxes, scores, ids = res[0]  # batch size = 1 in MOT
                mot_result = (frame_id + 1, boxes[0], scores[0],
                              ids[0])  # single class
                statistic = flow_statistic(
                    mot_result,
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
                    ids2names=self.mot_predictor.pred_config.labels)
                records = statistic['records']

                if self.illegal_parking_time != -1:
                    object_in_region_info, illegal_parking_dict = update_object_info(
                        object_in_region_info, mot_result, self.region_type,
                        entrance, video_fps, self.illegal_parking_time)
                    if len(illegal_parking_dict) != 0:
                        # build relationship between id and plate
                        for key, value in illegal_parking_dict.items():
                            plate = self.collector.get_carlp(key)
                            illegal_parking_dict[key]['plate'] = plate

                # nothing detected
                if len(mot_res['boxes']) == 0:
                    frame_id += 1
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.img_num += 1
                        self.pipe_timer.total_time.end()
                    if self.cfg['visual']:
                        _, _, fps = self.pipe_timer.get_total_time()
                        im = self.visualize_video(
                            frame_rgb, mot_res, self.collector, frame_id, fps,
                            entrance, records, center_traj)  # visualize
                        if len(self.pushurl) > 0:
                            pushstream.pipe.stdin.write(im.tobytes())
                        else:
                            writer.write(im)
                            if self.file_name is None:  # use camera_id
                                cv2.imshow('Paddle-Pipeline', im)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                    continue

                self.pipeline_res.update(mot_res, 'mot')
                crop_input, new_bboxes, ori_bboxes = crop_image_with_mot(
                    frame_rgb, mot_res)

                if self.with_vehicleplate and frame_id % 10 == 0:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicleplate'].start()
                    plate_input, _, _ = crop_image_with_mot(
                        frame_rgb, mot_res, expand=False)
                    platelicense = self.vehicleplate_detector.get_platelicense(
                        plate_input)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicleplate'].end()
                    self.pipeline_res.update(platelicense, 'vehicleplate')
                else:
                    self.pipeline_res.clear('vehicleplate')

                if self.with_human_attr:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['attr'].start()
                    attr_res = self.attr_predictor.predict_image(
                        crop_input, visual=False)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['attr'].end()
                    self.pipeline_res.update(attr_res, 'attr')

                if self.with_vehicle_attr:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicle_attr'].start()
                    attr_res = self.vehicle_attr_predictor.predict_image(
                        crop_input, visual=False)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicle_attr'].end()
                    self.pipeline_res.update(attr_res, 'vehicle_attr')

                if self.with_vehicle_press or self.with_vehicle_retrograde:
                    if frame_id == 0 or cars_count == 0 or cars_count > len(
                            mot_res['boxes']):

                        if frame_id > self.warmup_frame:
                            self.pipe_timer.module_time['lanes'].start()
                        lanes, directions = self.laneseg_predictor.run(
                            [copy.deepcopy(frame_rgb)])
                        lanes_res = {'output': lanes, 'directions': directions}
                        if frame_id > self.warmup_frame:
                            self.pipe_timer.module_time['lanes'].end()

                        if frame_id == 0 or (len(lanes) > 0 and frame_id > 0):
                            self.pipeline_res.update(lanes_res, 'lanes')

                        cars_count = len(mot_res['boxes'])

                if self.with_vehicle_press:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicle_press'].start()
                    press_lane = copy.deepcopy(self.pipeline_res.get('lanes'))
                    if press_lane is None:
                        continue

                    vehicle_press_res_list = self.vehicle_press_predictor.mot_run(
                        press_lane, mot_res['boxes'])
                    vehiclepress_res = {'output': vehicle_press_res_list}

                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicle_press'].end()

                    self.pipeline_res.update(vehiclepress_res, 'vehicle_press')

                if self.with_idbased_detaction:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['det_action'].start()
                    det_action_res = self.det_action_predictor.predict(
                        crop_input, mot_res)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['det_action'].end()
                    self.pipeline_res.update(det_action_res, 'det_action')

                    if self.cfg['visual']:
                        self.det_action_visual_helper.update(det_action_res)

                if self.with_idbased_clsaction:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['cls_action'].start()
                    cls_action_res = self.cls_action_predictor.predict_with_mot(
                        crop_input, mot_res)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['cls_action'].end()
                    self.pipeline_res.update(cls_action_res, 'cls_action')

                    if self.cfg['visual']:
                        self.cls_action_visual_helper.update(cls_action_res)

                if self.with_skeleton_action:
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['kpt'].start()
                    kpt_pred = self.kpt_predictor.predict_image(
                        crop_input, visual=False)
                    keypoint_vector, score_vector = translate_to_ori_images(
                        kpt_pred, np.array(new_bboxes))
                    kpt_res = {}
                    kpt_res['keypoint'] = [
                        keypoint_vector.tolist(), score_vector.tolist()
                    ] if len(keypoint_vector) > 0 else [[], []]
                    kpt_res['bbox'] = ori_bboxes
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['kpt'].end()

                    self.pipeline_res.update(kpt_res, 'kpt')

                    self.kpt_buff.update(kpt_res, mot_res)  # collect kpt output
                    state = self.kpt_buff.get_state(
                    )  # whether frame num is enough or lost tracker

                    skeleton_action_res = {}
                    if state:
                        if frame_id > self.warmup_frame:
                            self.pipe_timer.module_time[
                                'skeleton_action'].start()
                        collected_keypoint = self.kpt_buff.get_collected_keypoint(
                        )  # reoragnize kpt output with ID
                        skeleton_action_input = parse_mot_keypoint(
                            collected_keypoint, self.coord_size)
                        skeleton_action_res = self.skeleton_action_predictor.predict_skeleton_with_mot(
                            skeleton_action_input)
                        if frame_id > self.warmup_frame:
                            self.pipe_timer.module_time['skeleton_action'].end()
                        self.pipeline_res.update(skeleton_action_res,
                                                 'skeleton_action')

                    if self.cfg['visual']:
                        self.skeleton_action_visual_helper.update(
                            skeleton_action_res)

                if self.with_mtmct and frame_id % 10 == 0:
                    crop_input, img_qualities, rects = self.reid_predictor.crop_image_with_mot(
                        frame_rgb, mot_res)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['reid'].start()
                    reid_res = self.reid_predictor.predict_batch(crop_input)

                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['reid'].end()

                    reid_res_dict = {
                        'features': reid_res,
                        "qualities": img_qualities,
                        "rects": rects
                    }
                    self.pipeline_res.update(reid_res_dict, 'reid')
                else:
                    self.pipeline_res.clear('reid')

            if self.with_video_action:
                # get the params
                frame_len = self.cfg["VIDEO_ACTION"]["frame_len"]
                sample_freq = self.cfg["VIDEO_ACTION"]["sample_freq"]

                if sample_freq * frame_len > frame_count:  # video is too short
                    sample_freq = int(frame_count / frame_len)

                # filter the warmup frames
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['video_action'].start()

                # collect frames
                if frame_id % sample_freq == 0:
                    # Scale image
                    scaled_img = scale(frame_rgb)
                    video_action_imgs.append(scaled_img)

                # the number of collected frames is enough to predict video action
                if len(video_action_imgs) == frame_len:
                    classes, scores = self.video_action_predictor.predict(
                        video_action_imgs)
                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['video_action'].end()

                    video_action_res = {"class": classes[0], "score": scores[0]}
                    self.pipeline_res.update(video_action_res, 'video_action')

                    print("video_action_res:", video_action_res)

                    video_action_imgs.clear()  # next clip

            if self.with_vehicle_retrograde:
                # get the params
                frame_len = self.cfg["VEHICLE_RETROGRADE"]["frame_len"]
                sample_freq = self.cfg["VEHICLE_RETROGRADE"]["sample_freq"]

                if sample_freq * frame_len > frame_count:  # video is too short
                    sample_freq = int(frame_count / frame_len)

                # filter the warmup frames
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['vehicle_retrograde'].start()

                if frame_id % sample_freq == 0:

                    frame_mot_res = copy.deepcopy(self.pipeline_res.get('mot'))
                    self.vehicle_retrograde_predictor.update_center_traj(
                        frame_mot_res, max_len=frame_len)
                    retrograde_traj_len = retrograde_traj_len + 1

                #the number of collected frames is enough to predict 
                if retrograde_traj_len == frame_len:
                    retrograde_mot_res = copy.deepcopy(
                        self.pipeline_res.get('mot'))
                    retrograde_lanes = copy.deepcopy(
                        self.pipeline_res.get('lanes'))
                    frame_shape = frame_rgb.shape

                    if retrograde_lanes is None:
                        continue
                    retrograde_res, fence_line = self.vehicle_retrograde_predictor.mot_run(
                        lanes_res=retrograde_lanes,
                        det_res=retrograde_mot_res,
                        frame_shape=frame_shape)

                    retrograde_res_update = self.pipeline_res.get(
                        'vehicle_retrograde')

                    if retrograde_res_update is not None:
                        retrograde_res_update = retrograde_res_update['output']
                        if retrograde_res is not None:
                            for retrograde_res_id in retrograde_res:
                                if retrograde_res_id not in retrograde_res_update:
                                    retrograde_res_update.append(
                                        retrograde_res_id)
                    else:
                        retrograde_res_update = []

                    retrograde_res_dict = {
                        'output': retrograde_res_update,
                        "fence_line": fence_line,
                    }

                    if retrograde_res is not None and len(retrograde_res) > 0:
                        print("retrograde res:", retrograde_res)

                    self.pipeline_res.update(retrograde_res_dict,
                                             'vehicle_retrograde')

                    if frame_id > self.warmup_frame:
                        self.pipe_timer.module_time['vehicle_retrograde'].end()

                    retrograde_traj_len = 0

            self.collector.append(frame_id, self.pipeline_res)

            if frame_id > self.warmup_frame:
                self.pipe_timer.img_num += 1
                self.pipe_timer.total_time.end()
            frame_id += 1

            if self.cfg['visual']:
                _, _, fps = self.pipe_timer.get_total_time()

                im = self.visualize_video(frame_rgb, self.pipeline_res,
                                          self.collector, frame_id, fps,
                                          entrance, records, center_traj,
                                          self.illegal_parking_time != -1,
                                          illegal_parking_dict)  # visualize
                if len(self.pushurl) > 0:
                    pushstream.pipe.stdin.write(im.tobytes())
                else:
                    writer.write(im)
                    if self.file_name is None:  # use camera_id
                        cv2.imshow('Paddle-Pipeline', im)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        if self.cfg['visual'] and len(self.pushurl) == 0:
            writer.release()
            print('save result to {}'.format(out_path))

    def visualize_video(self,
                        image_rgb,
                        result,
                        collector,
                        frame_id,
                        fps,
                        entrance=None,
                        records=None,
                        center_traj=None,
                        do_illegal_parking_recognition=False,
                        illegal_parking_dict=None):
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        mot_res = copy.deepcopy(result.get('mot'))

        if mot_res is not None:
            ids = mot_res['boxes'][:, 0]
            scores = mot_res['boxes'][:, 2]
            boxes = mot_res['boxes'][:, 3:]
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        else:
            boxes = np.zeros([0, 4])
            ids = np.zeros([0])
            scores = np.zeros([0])

        # single class, still need to be defaultdict type for ploting
        num_classes = 1
        online_tlwhs = defaultdict(list)
        online_scores = defaultdict(list)
        online_ids = defaultdict(list)
        online_tlwhs[0] = boxes
        online_scores[0] = scores
        online_ids[0] = ids

        if mot_res is not None:
            image = plot_tracking_dict(
                image,
                num_classes,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=frame_id,
                fps=fps,
                ids2names=self.mot_predictor.pred_config.labels,
                do_entrance_counting=self.do_entrance_counting,
                do_break_in_counting=self.do_break_in_counting,
                do_illegal_parking_recognition=do_illegal_parking_recognition,
                illegal_parking_dict=illegal_parking_dict,
                entrance=entrance,
                records=records,
                center_traj=center_traj)

        human_attr_res = result.get('attr')
        if human_attr_res is not None:
            boxes = mot_res['boxes'][:, 1:]
            human_attr_res = human_attr_res['output']
            image = visualize_attr(image, human_attr_res, boxes)
            image = np.array(image)

        vehicle_attr_res = result.get('vehicle_attr')
        if vehicle_attr_res is not None:
            boxes = mot_res['boxes'][:, 1:]
            vehicle_attr_res = vehicle_attr_res['output']
            image = visualize_attr(image, vehicle_attr_res, boxes)
            image = np.array(image)

        lanes_res = result.get('lanes')
        if lanes_res is not None:
            lanes = lanes_res['output'][0]
            image = visualize_lane(image, lanes)
            image = np.array(image)

        vehiclepress_res = result.get('vehicle_press')
        if vehiclepress_res is not None:
            press_vehicle = vehiclepress_res['output']
            if len(press_vehicle) > 0:
                image = visualize_vehiclepress(
                    image, press_vehicle, threshold=self.cfg['crop_thresh'])
                image = np.array(image)

        if mot_res is not None:
            vehicleplate = False
            plates = []
            for trackid in mot_res['boxes'][:, 0]:
                plate = collector.get_carlp(trackid)
                if plate != None:
                    vehicleplate = True
                    plates.append(plate)
                else:
                    plates.append("")
            if vehicleplate:
                boxes = mot_res['boxes'][:, 1:]
                image = visualize_vehicleplate(image, plates, boxes)
                image = np.array(image)

        kpt_res = result.get('kpt')
        if kpt_res is not None:
            image = visualize_pose(
                image,
                kpt_res,
                visual_thresh=self.cfg['kpt_thresh'],
                returnimg=True)

        video_action_res = result.get('video_action')
        if video_action_res is not None:
            video_action_score = None
            if video_action_res and video_action_res["class"] == 1:
                video_action_score = video_action_res["score"]
            mot_boxes = None
            if mot_res:
                mot_boxes = mot_res['boxes']
            image = visualize_action(
                image,
                mot_boxes,
                action_visual_collector=None,
                action_text="SkeletonAction",
                video_action_score=video_action_score,
                video_action_text="Fight")

        vehicle_retrograde_res = result.get('vehicle_retrograde')
        if vehicle_retrograde_res is not None:
            mot_retrograde_res = copy.deepcopy(result.get('mot'))
            image = visualize_vehicle_retrograde(image, mot_retrograde_res,
                                                 vehicle_retrograde_res)
            image = np.array(image)

        visual_helper_for_display = []
        action_to_display = []

        skeleton_action_res = result.get('skeleton_action')
        if skeleton_action_res is not None:
            visual_helper_for_display.append(self.skeleton_action_visual_helper)
            action_to_display.append("Falling")

        det_action_res = result.get('det_action')
        if det_action_res is not None:
            visual_helper_for_display.append(self.det_action_visual_helper)
            action_to_display.append("Smoking")

        cls_action_res = result.get('cls_action')
        if cls_action_res is not None:
            visual_helper_for_display.append(self.cls_action_visual_helper)
            action_to_display.append("Calling")

        if len(visual_helper_for_display) > 0:
            image = visualize_action(image, mot_res['boxes'],
                                     visual_helper_for_display,
                                     action_to_display)

        return image

    def visualize_image(self, im_files, images, result):
        start_idx, boxes_num_i = 0, 0
        det_res = result.get('det')
        human_attr_res = result.get('attr')
        vehicle_attr_res = result.get('vehicle_attr')
        vehicleplate_res = result.get('vehicleplate')
        lanes_res = result.get('lanes')
        vehiclepress_res = result.get('vehicle_press')

        for i, (im_file, im) in enumerate(zip(im_files, images)):
            if det_res is not None:
                det_res_i = {}
                boxes_num_i = det_res['boxes_num'][i]
                det_res_i['boxes'] = det_res['boxes'][start_idx:start_idx +
                                                      boxes_num_i, :]
                im = visualize_box_mask(
                    im,
                    det_res_i,
                    labels=['target'],
                    threshold=self.cfg['crop_thresh'])
                im = np.ascontiguousarray(np.copy(im))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            if human_attr_res is not None:
                human_attr_res_i = human_attr_res['output'][start_idx:start_idx
                                                            + boxes_num_i]
                im = visualize_attr(im, human_attr_res_i, det_res_i['boxes'])
            if vehicle_attr_res is not None:
                vehicle_attr_res_i = vehicle_attr_res['output'][
                    start_idx:start_idx + boxes_num_i]
                im = visualize_attr(im, vehicle_attr_res_i, det_res_i['boxes'])
            if vehicleplate_res is not None:
                plates = vehicleplate_res['vehicleplate']
                det_res_i['boxes'][:, 4:6] = det_res_i[
                    'boxes'][:, 4:6] - det_res_i['boxes'][:, 2:4]
                im = visualize_vehicleplate(im, plates, det_res_i['boxes'])
            if vehiclepress_res is not None:
                press_vehicle = vehiclepress_res['output'][i]
                if len(press_vehicle) > 0:
                    im = visualize_vehiclepress(
                        im, press_vehicle, threshold=self.cfg['crop_thresh'])
                    im = np.ascontiguousarray(np.copy(im))
            if lanes_res is not None:
                lanes = lanes_res['output'][i]
                im = visualize_lane(im, lanes)
                im = np.ascontiguousarray(np.copy(im))

            img_name = os.path.split(im_file)[-1]
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            out_path = os.path.join(self.output_dir, img_name)
            cv2.imwrite(out_path, im)
            print("save result to: " + out_path)
            start_idx += boxes_num_i


def main():
    cfg = merge_cfg(FLAGS)  # use command params to update config
    print_arguments(cfg)

    pipeline = Pipeline(FLAGS, cfg)
    # pipeline.run()
    pipeline.run_multithreads()


if __name__ == '__main__':
    paddle.enable_static()

    # parse params from command
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU', 'NPU'
                            ], "device should be CPU, GPU, XPU or NPU"

    main()
