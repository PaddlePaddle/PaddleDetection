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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import argparse
import time
import yaml
import ast
from functools import reduce

from PIL import Image
import cv2
import numpy as np
import paddle
import paddle.fluid as fluid
from preprocess import preprocess, Resize, Normalize, Permute, PadStride
from visualize import visualize_box_mask, lmk2out

# Global dictionary
SUPPORT_MODELS = {
    'YOLO',
    'SSD',
    'RetinaNet',
    'EfficientDet',
    'RCNN',
    'Face',
    'TTF',
    'FCOS',
    'SOLOv2',
}


class Detector(object):
    """
    Args:
        config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of __model__, __params__ and infer_cfg.yml
        use_gpu (bool): whether use gpu
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16)
        threshold (float): threshold to reserve the result for output.
    """

    def __init__(self,
                 config,
                 model_dir,
                 use_gpu=False,
                 run_mode='fluid',
                 threshold=0.5):
        self.config = config
        if self.config.use_python_inference:
            self.executor, self.program, self.fecth_targets = load_executor(
                model_dir, use_gpu=use_gpu)
        else:
            self.predictor = load_predictor(
                model_dir,
                run_mode=run_mode,
                min_subgraph_size=self.config.min_subgraph_size,
                use_gpu=use_gpu)

    def preprocess(self, im):
        preprocess_ops = []
        for op_info in self.config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            if op_type == 'Resize':
                new_op_info['arch'] = self.config.arch
            preprocess_ops.append(eval(op_type)(**new_op_info))
        im, im_info = preprocess(im, preprocess_ops)
        inputs = create_inputs(im, im_info, self.config.arch)
        return inputs, im_info

    def postprocess(self, np_boxes, np_masks, np_lmk, im_info, threshold=0.5):
        # postprocess output of predictor
        results = {}
        if np_lmk is not None:
            results['landmark'] = lmk2out(np_boxes, np_lmk, im_info, threshold)

        if self.config.arch in ['SSD', 'Face']:
            w, h = im_info['origin_shape']
            np_boxes[:, 2] *= h
            np_boxes[:, 3] *= w
            np_boxes[:, 4] *= h
            np_boxes[:, 5] *= w
        expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[expect_boxes, :]
        for box in np_boxes:
            print('class_id:{:d}, confidence:{:.4f},'
                  'left_top:[{:.2f},{:.2f}],'
                  ' right_bottom:[{:.2f},{:.2f}]'.format(
                      int(box[0]), box[1], box[2], box[3], box[4], box[5]))
        results['boxes'] = np_boxes
        if np_masks is not None:
            np_masks = np_masks[expect_boxes, :, :, :]
            results['masks'] = np_masks
        return results

    def predict(self,
                image,
                threshold=0.5,
                warmup=0,
                repeats=1,
                run_benchmark=False):
        '''
        Args:
            image (str/np.ndarray): path of image/ np.ndarray read by cv2
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape:[N, class_num, mask_resolution, mask_resolution]
        '''
        inputs, im_info = self.preprocess(image)
        np_boxes, np_masks, np_lmk = None, None, None
        if self.config.use_python_inference:
            for i in range(warmup):
                outs = self.executor.run(self.program,
                                         feed=inputs,
                                         fetch_list=self.fecth_targets,
                                         return_numpy=False)
            t1 = time.time()
            for i in range(repeats):
                outs = self.executor.run(self.program,
                                         feed=inputs,
                                         fetch_list=self.fecth_targets,
                                         return_numpy=False)
            t2 = time.time()
            ms = (t2 - t1) * 1000.0 / repeats
            print("Inference: {} ms per batch image".format(ms))
            np_boxes = np.array(outs[0])
            if self.config.mask_resolution is not None:
                np_masks = np.array(outs[1])
        else:
            input_names = self.predictor.get_input_names()
            for i in range(len(input_names)):
                input_tensor = self.predictor.get_input_tensor(input_names[i])
                input_tensor.copy_from_cpu(inputs[input_names[i]])

            for i in range(warmup):
                self.predictor.zero_copy_run()
                output_names = self.predictor.get_output_names()
                boxes_tensor = self.predictor.get_output_tensor(output_names[0])
                np_boxes = boxes_tensor.copy_to_cpu()
                if self.config.mask_resolution is not None:
                    masks_tensor = self.predictor.get_output_tensor(
                        output_names[1])
                    np_masks = masks_tensor.copy_to_cpu()

                if self.config.with_lmk is not None and self.config.with_lmk == True:
                    face_index = self.predictor.get_output_tensor(output_names[
                        1])
                    landmark = self.predictor.get_output_tensor(output_names[2])
                    prior_boxes = self.predictor.get_output_tensor(output_names[
                        3])
                    np_face_index = face_index.copy_to_cpu()
                    np_prior_boxes = prior_boxes.copy_to_cpu()
                    np_landmark = landmark.copy_to_cpu()
                    np_lmk = [np_face_index, np_landmark, np_prior_boxes]

            t1 = time.time()
            for i in range(repeats):
                self.predictor.zero_copy_run()
                output_names = self.predictor.get_output_names()
                boxes_tensor = self.predictor.get_output_tensor(output_names[0])
                np_boxes = boxes_tensor.copy_to_cpu()
                if self.config.mask_resolution is not None:
                    masks_tensor = self.predictor.get_output_tensor(
                        output_names[1])
                    np_masks = masks_tensor.copy_to_cpu()

                if self.config.with_lmk is not None and self.config.with_lmk == True:
                    face_index = self.predictor.get_output_tensor(output_names[
                        1])
                    landmark = self.predictor.get_output_tensor(output_names[2])
                    prior_boxes = self.predictor.get_output_tensor(output_names[
                        3])
                    np_face_index = face_index.copy_to_cpu()
                    np_prior_boxes = prior_boxes.copy_to_cpu()
                    np_landmark = landmark.copy_to_cpu()
                    np_lmk = [np_face_index, np_landmark, np_prior_boxes]
            t2 = time.time()
            ms = (t2 - t1) * 1000.0 / repeats
            print("Inference: {} ms per batch image".format(ms))

        # do not perform postprocess in benchmark mode
        results = []
        if not run_benchmark:
            if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
                print('[WARNNING] No object detected.')
                results = {'boxes': np.array([])}
            else:
                results = self.postprocess(
                    np_boxes, np_masks, np_lmk, im_info, threshold=threshold)

        return results


class DetectorSOLOv2(Detector):
    def __init__(self,
                 config,
                 model_dir,
                 use_gpu=False,
                 run_mode='fluid',
                 threshold=0.5):
        super(DetectorSOLOv2, self).__init__(
            config=config,
            model_dir=model_dir,
            use_gpu=use_gpu,
            run_mode=run_mode,
            threshold=threshold)

    def predict(self,
                image,
                threshold=0.5,
                warmup=0,
                repeats=1,
                run_benchmark=False):
        inputs, im_info = self.preprocess(image)
        np_label, np_score, np_segms = None, None, None
        if self.config.use_python_inference:
            for i in range(warmup):
                outs = self.executor.run(self.program,
                                         feed=inputs,
                                         fetch_list=self.fecth_targets,
                                         return_numpy=False)
            t1 = time.time()
            for i in range(repeats):
                outs = self.executor.run(self.program,
                                         feed=inputs,
                                         fetch_list=self.fecth_targets,
                                         return_numpy=False)
            t2 = time.time()
            ms = (t2 - t1) * 1000.0 / repeats
            print("Inference: {} ms per batch image".format(ms))
            np_label, np_score, np_segms = np.array(outs[0]), np.array(outs[
                1]), np.array(outs[2])
        else:
            input_names = self.predictor.get_input_names()
            for i in range(len(input_names)):
                input_tensor = self.predictor.get_input_tensor(input_names[i])
                input_tensor.copy_from_cpu(inputs[input_names[i]])
            for i in range(warmup):
                self.predictor.zero_copy_run()
                output_names = self.predictor.get_output_names()
                np_label = self.predictor.get_output_tensor(output_names[
                    0]).copy_to_cpu()
                np_score = self.predictor.get_output_tensor(output_names[
                    1]).copy_to_cpu()
                np_segms = self.predictor.get_output_tensor(output_names[
                    2]).copy_to_cpu()

            t1 = time.time()
            for i in range(repeats):
                self.predictor.zero_copy_run()
                output_names = self.predictor.get_output_names()
                np_label = self.predictor.get_output_tensor(output_names[
                    0]).copy_to_cpu()
                np_score = self.predictor.get_output_tensor(output_names[
                    1]).copy_to_cpu()
                np_segms = self.predictor.get_output_tensor(output_names[
                    2]).copy_to_cpu()
            t2 = time.time()
            ms = (t2 - t1) * 1000.0 / repeats
            print("Inference: {} ms per batch image".format(ms))

        # do not perform postprocess in benchmark mode
        results = []
        if not run_benchmark:
            return dict(segm=np_segms, label=np_label, score=np_score)
        return results


def create_inputs(im, im_info, model_arch='YOLO'):
    """generate input for different model type
    Args:
        im (np.ndarray): image (np.ndarray)
        im_info (dict): info of image
        model_arch (str): model type
    Returns:
        inputs (dict): input of model
    """
    inputs = {}
    inputs['image'] = im
    origin_shape = list(im_info['origin_shape'])
    resize_shape = list(im_info['resize_shape'])
    pad_shape = list(im_info['pad_shape']) if im_info[
        'pad_shape'] is not None else list(im_info['resize_shape'])
    scale_x, scale_y = im_info['scale']
    if 'YOLO' in model_arch:
        im_size = np.array([origin_shape]).astype('int32')
        inputs['im_size'] = im_size
    elif 'RetinaNet' in model_arch or 'EfficientDet' in model_arch:
        scale = scale_x
        im_info = np.array([pad_shape + [scale]]).astype('float32')
        inputs['im_info'] = im_info
    elif ('RCNN' in model_arch) or ('FCOS' in model_arch):
        scale = scale_x
        im_info = np.array([pad_shape + [scale]]).astype('float32')
        im_shape = np.array([origin_shape + [1.]]).astype('float32')
        inputs['im_info'] = im_info
        inputs['im_shape'] = im_shape
    elif 'TTF' in model_arch:
        scale_factor = np.array([scale_x, scale_y] * 2).astype('float32')
        inputs['scale_factor'] = scale_factor
    elif 'SOLOv2' in model_arch:
        scale = scale_x
        im_info = np.array([resize_shape + [scale]]).astype('float32')
        inputs['im_info'] = im_info
    return inputs


class Config():
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
        self.preprocess_infos = yml_conf['Preprocess']
        self.use_python_inference = yml_conf['use_python_inference']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask_resolution = None
        if 'mask_resolution' in yml_conf:
            self.mask_resolution = yml_conf['mask_resolution']
        self.with_lmk = None
        if 'with_lmk' in yml_conf:
            self.with_lmk = yml_conf['with_lmk']
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type 
        """
        for support_model in SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: %s' % ('Use Paddle Executor', self.use_python_inference))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


def load_predictor(model_dir,
                   run_mode='fluid',
                   batch_size=1,
                   use_gpu=False,
                   min_subgraph_size=3):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        use_gpu (bool): whether use gpu
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need use_gpu == True.
    """
    if not use_gpu and not run_mode == 'fluid':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect use_gpu==True, but use_gpu == {}"
            .format(run_mode, use_gpu))
    if run_mode == 'trt_int8':
        raise ValueError("TensorRT int8 mode is not supported now, "
                         "please use trt_fp32 or trt_fp16 instead.")
    precision_map = {
        'trt_int8': fluid.core.AnalysisConfig.Precision.Int8,
        'trt_fp32': fluid.core.AnalysisConfig.Precision.Float32,
        'trt_fp16': fluid.core.AnalysisConfig.Precision.Half
    }
    config = fluid.core.AnalysisConfig(
        os.path.join(model_dir, '__model__'),
        os.path.join(model_dir, '__params__'))
    if use_gpu:
        # initial GPU memory(M), device ID
        config.enable_use_gpu(100, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()

    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=1 << 10,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=False)

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = fluid.core.create_paddle_predictor(config)
    return predictor


def load_executor(model_dir, use_gpu=False):
    if use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    program, feed_names, fetch_targets = fluid.io.load_inference_model(
        dirname=model_dir,
        executor=exe,
        model_filename='__model__',
        params_filename='__params__')
    return exe, program, fetch_targets


def visualize(image_file,
              results,
              labels,
              mask_resolution=14,
              output_dir='output/',
              threshold=0.5):
    # visualize the predict result
    im = visualize_box_mask(
        image_file,
        results,
        labels,
        mask_resolution=mask_resolution,
        threshold=threshold)
    img_name = os.path.split(image_file)[-1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_path = os.path.join(output_dir, img_name)
    im.save(out_path, quality=95)
    print("save result to: " + out_path)


def print_arguments(args):
    print('-----------  Running Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------')


def predict_image(detector):
    if FLAGS.run_benchmark:
        detector.predict(
            FLAGS.image_file,
            FLAGS.threshold,
            warmup=100,
            repeats=100,
            run_benchmark=True)
    else:
        results = detector.predict(FLAGS.image_file, FLAGS.threshold)
        visualize(
            FLAGS.image_file,
            results,
            detector.config.labels,
            mask_resolution=detector.config.mask_resolution,
            output_dir=FLAGS.output_dir,
            threshold=FLAGS.threshold)


def predict_video(detector, camera_id):
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
        video_name = 'output.mp4'
    else:
        capture = cv2.VideoCapture(FLAGS.video_file)
        video_name = os.path.split(FLAGS.video_file)[-1]
    fps = 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    out_path = os.path.join(FLAGS.output_dir, video_name)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    index = 1
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        print('detect frame:%d' % (index))
        index += 1
        results = detector.predict(frame, FLAGS.threshold)
        im = visualize_box_mask(
            frame,
            results,
            detector.config.labels,
            mask_resolution=detector.config.mask_resolution,
            threshold=FLAGS.threshold)
        im = np.array(im)
        writer.write(im)
        if camera_id != -1:
            cv2.imshow('Mask Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    writer.release()


def main():
    config = Config(FLAGS.model_dir)
    detector = Detector(
        config, FLAGS.model_dir, use_gpu=FLAGS.use_gpu, run_mode=FLAGS.run_mode)
    if config.arch == 'SOLOv2':
        detector = DetectorSOLOv2(
            config,
            FLAGS.model_dir,
            use_gpu=FLAGS.use_gpu,
            run_mode=FLAGS.run_mode)
    # predict from image
    if FLAGS.image_file != '':
        predict_image(detector)
    # predict from video file or camera video stream
    if FLAGS.video_file != '' or FLAGS.camera_id != -1:
        predict_video(detector, FLAGS.camera_id)


if __name__ == '__main__':
    try:
        paddle.enable_static()
    except:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help=("Directory include:'__model__', '__params__', "
              "'infer_cfg.yml', created by tools/export_model.py."),
        required=True)
    parser.add_argument(
        "--image_file", type=str, default='', help="Path of image file.")
    parser.add_argument(
        "--video_file", type=str, default='', help="Path of video file.")
    parser.add_argument(
        "--camera_id",
        type=int,
        default=-1,
        help="device id of camera to predict.")
    parser.add_argument(
        "--run_mode",
        type=str,
        default='fluid',
        help="mode of running(fluid/trt_fp32/trt_fp16)")
    parser.add_argument(
        "--use_gpu",
        type=ast.literal_eval,
        default=False,
        help="Whether to predict with GPU.")
    parser.add_argument(
        "--run_benchmark",
        type=ast.literal_eval,
        default=False,
        help="Whether to predict a image_file repeatedly for benchmark")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold of score.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory of output visualization files.")

    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    if FLAGS.image_file != '' and FLAGS.video_file != '':
        assert "Cannot predict image and video at the same time"

    main()
