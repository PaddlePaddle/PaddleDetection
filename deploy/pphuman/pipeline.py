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
from collections import Sequence

# add deploy path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from python.infer import Detector, DetectorPicoDet
from python.attr_infer import AttrDetector
from python.keypoint_infer import KeyPointDetector
from python.keypoint_postprocess import translate_to_ori_images
from python.action_infer import ActionRecognizer
from python.action_utils import KeyPointCollector, ActionVisualCollector

from pipe_utils import argsparser, print_arguments, merge_cfg, PipeTimer
from pipe_utils import get_test_images, crop_image_with_det, crop_image_with_mot, parse_mot_res, parse_mot_keypoint
from python.preprocess import decode_image
from python.visualize import visualize_box_mask, visualize_attr, visualize_pose, visualize_action

from pptracking.python.mot_sde_infer import SDE_Detector
from pptracking.python.visualize import plot_tracking


class Pipeline(object):
    """
    Pipeline

    Args:
        cfg (dict): config of models in pipeline
        image_file (string|None): the path of image file, default as None
        image_dir (string|None): the path of image directory, if not None, 
            then all the images in directory will be predicted, default as None
        video_file (string|None): the path of video file, default as None
        camera_id (int): the device id of camera to predict, default as -1
        enable_attr (bool): whether use attribute recognition, default as false
        enable_action (bool): whether use action recognition, default as false
        device (string): the device to predict, options are: CPU/GPU/XPU, 
            default as CPU
        run_mode (string): the mode of prediction, options are: 
            paddle/trt_fp32/trt_fp16, default as paddle
        trt_min_shape (int): min shape for dynamic shape in trt, default as 1
        trt_max_shape (int): max shape for dynamic shape in trt, default as 1280
        trt_opt_shape (int): opt shape for dynamic shape in trt, default as 640
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True. default as False
        cpu_threads (int): cpu threads, default as 1
        enable_mkldnn (bool): whether to open MKLDNN, default as False
        output_dir (string): The path of output, default as 'output'
    """

    def __init__(self,
                 cfg,
                 image_file=None,
                 image_dir=None,
                 video_file=None,
                 camera_id=-1,
                 enable_attr=False,
                 enable_action=True,
                 device='CPU',
                 run_mode='paddle',
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 output_dir='output'):
        self.multi_camera = False
        self.is_video = False
        self.input = self._parse_input(image_file, image_dir, video_file,
                                       camera_id)
        if self.multi_camera:
            self.predictor = [
                PipePredictor(
                    cfg,
                    is_video=True,
                    multi_camera=True,
                    enable_attr=enable_attr,
                    enable_action=enable_action,
                    device=device,
                    run_mode=run_mode,
                    trt_min_shape=trt_min_shape,
                    trt_max_shape=trt_max_shape,
                    trt_opt_shape=trt_opt_shape,
                    cpu_threads=cpu_threads,
                    enable_mkldnn=enable_mkldnn,
                    output_dir=output_dir) for i in self.input
            ]
        else:
            self.predictor = PipePredictor(
                cfg,
                self.is_video,
                enable_attr=enable_attr,
                enable_action=enable_action,
                device=device,
                run_mode=run_mode,
                trt_min_shape=trt_min_shape,
                trt_max_shape=trt_max_shape,
                trt_opt_shape=trt_opt_shape,
                trt_calib_mode=trt_calib_mode,
                cpu_threads=cpu_threads,
                enable_mkldnn=enable_mkldnn,
                output_dir=output_dir)
            if self.is_video:
                self.predictor.set_file_name(video_file)

    def _parse_input(self, image_file, image_dir, video_file, camera_id):

        # parse input as is_video and multi_camera

        if image_file is not None or image_dir is not None:
            input = get_test_images(image_dir, image_file)
            self.is_video = False
            self.multi_camera = False

        elif video_file is not None:
            if isinstance(video_file, list):
                self.multi_camera = True
                input = [cv2.VideoCapture(v) for v in video_file]
            else:
                input = cv2.VideoCapture(video_file)
            self.is_video = True

        elif camera_id != -1:
            if isinstance(camera_id, Sequence):
                self.multi_camera = True
                input = [cv2.VideoCapture(i) for i in camera_id]
            else:
                input = cv2.VideoCapture(camera_id)
            self.is_video = True

        else:
            raise ValueError(
                "Illegal Input, please set one of ['video_file'，'camera_id'，'image_file', 'image_dir']"
            )

        return input

    def run(self):
        if self.multi_camera:
            multi_res = []
            for predictor, input in zip(self.predictor, self.input):
                predictor.run(input)
                res = predictor.get_result()
                multi_res.append(res)

            mtmct_process(multi_res)

        else:
            self.predictor.run(self.input)


class Result(object):
    def __init__(self):
        self.res_dict = {
            'det': dict(),
            'mot': dict(),
            'attr': dict(),
            'kpt': dict(),
            'action': dict()
        }

    def update(self, res, name):
        self.res_dict[name].update(res)

    def get(self, name):
        if name in self.res_dict and len(self.res_dict[name]) > 0:
            return self.res_dict[name]
        return None


class PipePredictor(object):
    """
    Predictor in single camera
    
    The pipeline for image input: 

        1. Detection
        2. Detection -> Attribute

    The pipeline for video input: 

        1. Tracking
        2. Tracking -> Attribute
        3. Tracking -> KeyPoint -> Action Recognition

    Args:
        cfg (dict): config of models in pipeline
        is_video (bool): whether the input is video, default as False
        multi_camera (bool): whether to use multi camera in pipeline, 
            default as False
        camera_id (int): the device id of camera to predict, default as -1
        enable_attr (bool): whether use attribute recognition, default as false
        enable_action (bool): whether use action recognition, default as false
        device (string): the device to predict, options are: CPU/GPU/XPU, 
            default as CPU
        run_mode (string): the mode of prediction, options are: 
            paddle/trt_fp32/trt_fp16, default as paddle
        trt_min_shape (int): min shape for dynamic shape in trt, default as 1
        trt_max_shape (int): max shape for dynamic shape in trt, default as 1280
        trt_opt_shape (int): opt shape for dynamic shape in trt, default as 640
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True. default as False
        cpu_threads (int): cpu threads, default as 1
        enable_mkldnn (bool): whether to open MKLDNN, default as False
        output_dir (string): The path of output, default as 'output'
    """

    def __init__(self,
                 cfg,
                 is_video=True,
                 multi_camera=False,
                 enable_attr=False,
                 enable_action=False,
                 device='CPU',
                 run_mode='paddle',
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 output_dir='output'):

        if enable_attr and not cfg.get('ATTR', False):
            ValueError(
                'enable_attr is set to True, please set ATTR in config file')
        if enable_action and (not cfg.get('ACTION', False) or
                              not cfg.get('KPT', False)):
            ValueError(
                'enable_action is set to True, please set KPT and ACTION in config file'
            )

        self.with_attr = cfg.get('ATTR', False) and enable_attr
        self.with_action = cfg.get('ACTION', False) and enable_action
        if self.with_attr:
            print('Attribute Recognition enabled')
        if self.with_action:
            print('Action Recognition enabled')

        self.is_video = is_video
        self.multi_camera = multi_camera
        self.cfg = cfg
        self.output_dir = output_dir

        self.warmup_frame = 1
        self.pipeline_res = Result()
        self.pipe_timer = PipeTimer()
        self.file_name = None

        if not is_video:
            det_cfg = self.cfg['DET']
            model_dir = det_cfg['model_dir']
            batch_size = det_cfg['batch_size']
            self.det_predictor = Detector(
                model_dir, device, run_mode, batch_size, trt_min_shape,
                trt_max_shape, trt_opt_shape, trt_calib_mode, cpu_threads,
                enable_mkldnn)
            if self.with_attr:
                attr_cfg = self.cfg['ATTR']
                model_dir = attr_cfg['model_dir']
                batch_size = attr_cfg['batch_size']
                self.attr_predictor = AttrDetector(
                    model_dir, device, run_mode, batch_size, trt_min_shape,
                    trt_max_shape, trt_opt_shape, trt_calib_mode, cpu_threads,
                    enable_mkldnn)

        else:
            mot_cfg = self.cfg['MOT']
            model_dir = mot_cfg['model_dir']
            tracker_config = mot_cfg['tracker_config']
            batch_size = mot_cfg['batch_size']
            self.mot_predictor = SDE_Detector(
                model_dir, tracker_config, device, run_mode, batch_size,
                trt_min_shape, trt_max_shape, trt_opt_shape, trt_calib_mode,
                cpu_threads, enable_mkldnn)
            if self.with_attr:
                attr_cfg = self.cfg['ATTR']
                model_dir = attr_cfg['model_dir']
                batch_size = attr_cfg['batch_size']
                self.attr_predictor = AttrDetector(
                    model_dir, device, run_mode, batch_size, trt_min_shape,
                    trt_max_shape, trt_opt_shape, trt_calib_mode, cpu_threads,
                    enable_mkldnn)
            if self.with_action:
                kpt_cfg = self.cfg['KPT']
                kpt_model_dir = kpt_cfg['model_dir']
                kpt_batch_size = kpt_cfg['batch_size']
                action_cfg = self.cfg['ACTION']
                action_model_dir = action_cfg['model_dir']
                action_batch_size = action_cfg['batch_size']
                action_frames = action_cfg['max_frames']
                display_frames = action_cfg['display_frames']
                self.coord_size = action_cfg['coord_size']

                self.kpt_predictor = KeyPointDetector(
                    kpt_model_dir,
                    device,
                    run_mode,
                    kpt_batch_size,
                    trt_min_shape,
                    trt_max_shape,
                    trt_opt_shape,
                    trt_calib_mode,
                    cpu_threads,
                    enable_mkldnn,
                    use_dark=False)
                self.kpt_collector = KeyPointCollector(action_frames)

                self.action_predictor = ActionRecognizer(
                    action_model_dir,
                    device,
                    run_mode,
                    action_batch_size,
                    trt_min_shape,
                    trt_max_shape,
                    trt_opt_shape,
                    trt_calib_mode,
                    cpu_threads,
                    enable_mkldnn,
                    window_size=action_frames)

                self.action_visual_collector = ActionVisualCollector(
                    display_frames)

    def set_file_name(self, path):
        self.file_name = os.path.split(path)[-1]

    def get_result(self):
        return self.pipeline_res

    def run(self, input):
        if self.is_video:
            self.predict_video(input)
        else:
            self.predict_image(input)
        self.pipe_timer.info()

    def predict_image(self, input):
        # det
        # det -> attr
        batch_loop_cnt = math.ceil(
            float(len(input)) / self.det_predictor.batch_size)
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
            self.pipeline_res.update(det_res, 'det')

            if self.with_attr:
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

            self.pipe_timer.img_num += len(batch_input)
            if i > self.warmup_frame:
                self.pipe_timer.total_time.end()

            if self.cfg['visual']:
                self.visualize_image(batch_file, batch_input, self.pipeline_res)

    def predict_video(self, capture):
        # mot
        # mot -> attr
        # mot -> pose -> action
        video_out_name = 'output.mp4' if self.file_name is None else self.file_name

        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        out_path = os.path.join(self.output_dir, video_out_name)
        fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        frame_id = 0
        while (1):
            if frame_id % 10 == 0:
                print('frame id: ', frame_id)
            ret, frame = capture.read()
            if not ret:
                break

            if frame_id > self.warmup_frame:
                self.pipe_timer.total_time.start()
                self.pipe_timer.module_time['mot'].start()
            res = self.mot_predictor.predict_image([frame], visual=False)

            if frame_id > self.warmup_frame:
                self.pipe_timer.module_time['mot'].end()

            # mot output format: id, class, score, xmin, ymin, xmax, ymax
            mot_res = parse_mot_res(res)

            # nothing detected
            if len(mot_res['boxes']) == 0:
                frame_id += 1
                self.pipe_timer.img_num += 1
                self.pipe_timer.total_time.end()
                if self.cfg['visual']:
                    _, _, fps = self.pipe_timer.get_total_time()
                    im = self.visualize_video(frame, mot_res, frame_id,
                                              fps)  # visualize
                    writer.write(im)
                continue

            self.pipeline_res.update(mot_res, 'mot')
            if self.with_attr or self.with_action:
                crop_input, new_bboxes, ori_bboxes = crop_image_with_mot(
                    frame, mot_res)

            if self.with_attr:
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['attr'].start()
                attr_res = self.attr_predictor.predict_image(
                    crop_input, visual=False)
                if frame_id > self.warmup_frame:
                    self.pipe_timer.module_time['attr'].end()
                self.pipeline_res.update(attr_res, 'attr')

            if self.with_action:
                kpt_pred = self.kpt_predictor.predict_image(
                    crop_input, visual=False)
                keypoint_vector, score_vector = translate_to_ori_images(
                    kpt_pred, np.array(new_bboxes))
                kpt_res = {}
                kpt_res['keypoint'] = [
                    keypoint_vector.tolist(), score_vector.tolist()
                ] if len(keypoint_vector) > 0 else [[], []]
                kpt_res['bbox'] = ori_bboxes
                self.pipeline_res.update(kpt_res, 'kpt')

                self.kpt_collector.update(kpt_res,
                                          mot_res)  # collect kpt output
                state = self.kpt_collector.get_state(
                )  # whether frame num is enough or lost tracker

                action_res = {}
                if state:
                    collected_keypoint = self.kpt_collector.get_collected_keypoint(
                    )  # reoragnize kpt output with ID
                    action_input = parse_mot_keypoint(collected_keypoint,
                                                      self.coord_size)
                    action_res = self.action_predictor.predict_skeleton_with_mot(
                        action_input)
                    self.pipeline_res.update(action_res, 'action')

                if self.cfg['visual']:
                    self.action_visual_collector.update(action_res)

            if frame_id > self.warmup_frame:
                self.pipe_timer.img_num += 1
                self.pipe_timer.total_time.end()
            frame_id += 1

            if self.multi_camera:
                self.get_valid_instance(
                    frame,
                    self.pipeline_res)  # parse output result for multi-camera

            if self.cfg['visual']:
                _, _, fps = self.pipe_timer.get_total_time()
                im = self.visualize_video(frame, self.pipeline_res, frame_id,
                                          fps)  # visualize
                writer.write(im)

        writer.release()
        print('save result to {}'.format(out_path))

    def visualize_video(self, image, result, frame_id, fps):
        mot_res = result.get('mot')
        if mot_res is not None:
            ids = mot_res['boxes'][:, 0]
            boxes = mot_res['boxes'][:, 3:]
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        else:
            boxes = np.zeros([0, 4])
            ids = np.zeros([0])
        image = plot_tracking(image, boxes, ids, frame_id=frame_id, fps=fps)

        attr_res = result.get('attr')
        if attr_res is not None:
            boxes = mot_res['boxes'][:, 1:]
            attr_res = attr_res['output']
            image = visualize_attr(image, attr_res, boxes)
            image = np.array(image)

        kpt_res = result.get('kpt')
        if kpt_res is not None:
            image = visualize_pose(
                image,
                kpt_res,
                visual_thresh=self.cfg['kpt_thresh'],
                returnimg=True)

        action_res = result.get('action')
        if action_res is not None:
            image = visualize_action(image, mot_res['boxes'],
                                     self.action_visual_collector, "Falling")

        return image

    def visualize_image(self, im_files, images, result):
        start_idx, boxes_num_i = 0, 0
        det_res = result.get('det')
        attr_res = result.get('attr')
        for i, (im_file, im) in enumerate(zip(im_files, images)):
            if det_res is not None:
                det_res_i = {}
                boxes_num_i = det_res['boxes_num'][i]
                det_res_i['boxes'] = det_res['boxes'][start_idx:start_idx +
                                                      boxes_num_i, :]
                im = visualize_box_mask(
                    im,
                    det_res_i,
                    labels=['person'],
                    threshold=self.cfg['crop_thresh'])
                im = np.ascontiguousarray(np.copy(im))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            if attr_res is not None:
                attr_res_i = attr_res['output'][start_idx:start_idx +
                                                boxes_num_i]
                im = visualize_attr(im, attr_res_i, det_res_i['boxes'])
            img_name = os.path.split(im_file)[-1]
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            out_path = os.path.join(self.output_dir, img_name)
            cv2.imwrite(out_path, im)
            print("save result to: " + out_path)
            start_idx += boxes_num_i


def main():
    cfg = merge_cfg(FLAGS)
    print_arguments(cfg)
    pipeline = Pipeline(
        cfg, FLAGS.image_file, FLAGS.image_dir, FLAGS.video_file,
        FLAGS.camera_id, FLAGS.enable_attr, FLAGS.enable_action, FLAGS.device,
        FLAGS.run_mode, FLAGS.trt_min_shape, FLAGS.trt_max_shape,
        FLAGS.trt_opt_shape, FLAGS.trt_calib_mode, FLAGS.cpu_threads,
        FLAGS.enable_mkldnn, FLAGS.output_dir)

    pipeline.run()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()
