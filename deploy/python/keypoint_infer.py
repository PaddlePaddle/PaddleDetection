# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import glob
from functools import reduce

from PIL import Image
import cv2
import math
import numpy as np
import paddle
from preprocess import preprocess, NormalizeImage, Permute
from keypoint_preprocess import EvalAffine, TopDownEvalAffine, expand_crop
from keypoint_postprocess import HrHRNetPostProcess, HRNetPostProcess
from visualize import draw_pose
from paddle.inference import Config
from paddle.inference import create_predictor
from utils import argsparser, Timer, get_current_memory_mb
from benchmark_utils import PaddleInferBenchmark
from infer import Detector, get_test_images, print_arguments

# Global dictionary
KEYPOINT_SUPPORT_MODELS = {
    'HigherHRNet': 'keypoint_bottomup',
    'HRNet': 'keypoint_topdown'
}


class KeyPoint_Detector(Detector):
    """
    Args:
        config (object): config of model, defined by `Config(model_dir)`
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
        use_dark(bool): whether to use postprocess in DarkPose
    """

    def __init__(self,
                 pred_config,
                 model_dir,
                 device='CPU',
                 run_mode='fluid',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 use_dark=True):
        super(KeyPoint_Detector, self).__init__(
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
        self.use_dark = use_dark

    def get_person_from_rect(self, image, results, det_threshold=0.5):
        # crop the person result from image
        self.det_times.preprocess_time_s.start()
        det_results = results['boxes']
        mask = det_results[:, 1] > det_threshold
        valid_rects = det_results[mask]
        rect_images = []
        new_rects = []
        org_rects = []
        for rect in valid_rects:
            rect_image, new_rect, org_rect = expand_crop(image, rect)
            if rect_image is None or rect_image.size == 0:
                continue
            rect_images.append(rect_image)
            new_rects.append(new_rect)
            org_rects.append(org_rect)
        self.det_times.preprocess_time_s.end()
        return rect_images, new_rects, org_rects

    def preprocess(self, image_list):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []
        for im in image_list:
            im, im_info = preprocess(im, preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        return inputs

    def postprocess(self, np_boxes, np_masks, inputs, threshold=0.5):
        # postprocess output of predictor
        if KEYPOINT_SUPPORT_MODELS[
                self.pred_config.arch] == 'keypoint_bottomup':
            results = {}
            h, w = inputs['im_shape'][0]
            preds = [np_boxes]
            if np_masks is not None:
                preds += np_masks
            preds += [h, w]
            keypoint_postprocess = HrHRNetPostProcess()
            results['keypoint'] = keypoint_postprocess(*preds)
            return results
        elif KEYPOINT_SUPPORT_MODELS[
                self.pred_config.arch] == 'keypoint_topdown':
            results = {}
            imshape = inputs['im_shape'][:, ::-1]
            center = np.round(imshape / 2.)
            scale = imshape / 200.
            keypoint_postprocess = HRNetPostProcess(use_dark=self.use_dark)
            results['keypoint'] = keypoint_postprocess(np_boxes, center, scale)
            return results
        else:
            raise ValueError("Unsupported arch: {}, expect {}".format(
                self.pred_config.arch, KEYPOINT_SUPPORT_MODELS))

    def predict(self, image_list, threshold=0.5, warmup=0, repeats=1):
        '''
        Args:
            image_list (list): list of image 
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(image_list)
        np_boxes, np_masks = None, None
        input_names = self.predictor.get_input_names()

        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])
        self.det_times.preprocess_time_s.end()
        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if self.pred_config.tagmap:
                masks_tensor = self.predictor.get_output_handle(output_names[1])
                heat_k = self.predictor.get_output_handle(output_names[2])
                inds_k = self.predictor.get_output_handle(output_names[3])
                np_masks = [
                    masks_tensor.copy_to_cpu(), heat_k.copy_to_cpu(),
                    inds_k.copy_to_cpu()
                ]

        self.det_times.inference_time_s.start()
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if self.pred_config.tagmap:
                masks_tensor = self.predictor.get_output_handle(output_names[1])
                heat_k = self.predictor.get_output_handle(output_names[2])
                inds_k = self.predictor.get_output_handle(output_names[3])
                np_masks = [
                    masks_tensor.copy_to_cpu(), heat_k.copy_to_cpu(),
                    inds_k.copy_to_cpu()
                ]
        self.det_times.inference_time_s.end(repeats=repeats)

        self.det_times.postprocess_time_s.start()
        results = self.postprocess(
            np_boxes, np_masks, inputs, threshold=threshold)
        self.det_times.postprocess_time_s.end()
        self.det_times.img_num += len(image_list)
        return results


def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of image (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    inputs = {}
    inputs['image'] = np.stack(imgs, axis=0)
    im_shape = []
    for e in im_info:
        im_shape.append(np.array((e['im_shape'])).astype('float32'))
    inputs['im_shape'] = np.stack(im_shape, axis=0)
    return inputs


class PredictConfig_KeyPoint():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.archcls = KEYPOINT_SUPPORT_MODELS[yml_conf['arch']]
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.tagmap = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'keypoint_bottomup' == self.archcls:
            self.tagmap = True
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type 
        """
        for support_model in KEYPOINT_SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], KEYPOINT_SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


def predict_image(detector, image_list):
    for i, img_file in enumerate(image_list):
        if FLAGS.run_benchmark:
            detector.predict([img_file], FLAGS.threshold, warmup=10, repeats=10)
            cm, gm, gu = get_current_memory_mb()
            detector.cpu_mem += cm
            detector.gpu_mem += gm
            detector.gpu_util += gu
            print('Test iter {}, file name:{}'.format(i, img_file))
        else:
            results = detector.predict([img_file], FLAGS.threshold)
            if not os.path.exists(FLAGS.output_dir):
                os.makedirs(FLAGS.output_dir)
            draw_pose(
                img_file,
                results,
                visual_thread=FLAGS.threshold,
                save_dir=FLAGS.output_dir)


def predict_video(detector, camera_id):
    video_name = 'output.mp4'
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
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
    out_path = os.path.join(FLAGS.output_dir, video_name + '.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    index = 1
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        print('detect frame: %d' % (index))
        index += 1
        results = detector.predict([frame], FLAGS.threshold)
        im = draw_pose(
            frame, results, visual_thread=FLAGS.threshold, returnimg=True)
        writer.write(im)
        if camera_id != -1:
            cv2.imshow('Mask Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    writer.release()


def main():
    pred_config = PredictConfig_KeyPoint(FLAGS.model_dir)
    detector = KeyPoint_Detector(
        pred_config,
        FLAGS.model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn,
        use_dark=FLAGS.use_dark)

    # predict from video file or camera video stream
    if FLAGS.video_file is not None or FLAGS.camera_id != -1:
        predict_video(detector, FLAGS.camera_id)
    else:
        # predict from image
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        predict_image(detector, img_list)
        if not FLAGS.run_benchmark:
            detector.det_times.info(average=True)
        else:
            mems = {
                'cpu_rss_mb': detector.cpu_mem / len(img_list),
                'gpu_rss_mb': detector.gpu_mem / len(img_list),
                'gpu_util': detector.gpu_util * 100 / len(img_list)
            }
            perf_info = detector.det_times.report(average=True)
            model_dir = FLAGS.model_dir
            mode = FLAGS.run_mode
            model_info = {
                'model_name': model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            data_info = {
                'batch_size': 1,
                'shape': "dynamic_shape",
                'data_num': perf_info['img_num']
            }
            det_log = PaddleInferBenchmark(detector.config, model_info,
                                           data_info, perf_info, mems)
            det_log('KeyPoint')


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"
    assert not FLAGS.use_gpu, "use_gpu has been deprecated, please use --device"

    main()
