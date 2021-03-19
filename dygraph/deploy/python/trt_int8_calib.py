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
import argparse
import time
import yaml
import ast
from functools import reduce

from PIL import Image
import cv2
import numpy as np
import glob
import paddle
from preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride
from visualize import visualize_box_mask
from paddle.inference import Config
from paddle.inference import create_predictor

# Global dictionary
SUPPORT_MODELS = {
    'YOLO',
    'RCNN',
    'SSD',
    'FCOS',
    'SOLOv2',
    'TTFNet',
}


class Detector(object):
    """
    Args:
        config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        use_gpu (bool): whether use gpu
    """

    def __init__(self, pred_config, model_dir, use_gpu=False):
        self.pred_config = pred_config
        self.predictor = load_predictor(
            model_dir,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            use_gpu=use_gpu)

    def preprocess(self, im):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        im, im_info = preprocess(im, preprocess_ops,
                                 self.pred_config.input_shape)
        inputs = create_inputs(im, im_info)
        return inputs

    def postprocess(self, np_boxes, np_masks, inputs, threshold=0.5):
        # postprocess output of predictor
        results = {}
        if self.pred_config.arch in ['Face']:
            h, w = inputs['im_shape']
            scale_y, scale_x = inputs['scale_factor']
            w, h = float(h) / scale_y, float(w) / scale_x
            np_boxes[:, 2] *= h
            np_boxes[:, 3] *= w
            np_boxes[:, 4] *= h
            np_boxes[:, 5] *= w
        results['boxes'] = np_boxes
        if np_masks is not None:
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
                            shape: [N, im_h, im_w]
        '''
        inputs = self.preprocess(image)
        np_boxes, np_masks = None, None
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()

        t1 = time.time()
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()
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
                    np_boxes, np_masks, inputs, threshold=threshold)

        return results


def create_inputs(im, im_info):
    """generate input for different model type
    Args:
        im (np.ndarray): image (np.ndarray)
        im_info (dict): info of image
        model_arch (str): model type
    Returns:
        inputs (dict): input of model
    """
    inputs = {}
    inputs['image'] = np.array((im, )).astype('float32')
    inputs['im_shape'] = np.array((im_info['im_shape'], )).astype('float32')
    inputs['scale_factor'] = np.array(
        (im_info['scale_factor'], )).astype('float32')

    return inputs


class PredictConfig():
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
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        if 'mask' in yml_conf:
            self.mask = yml_conf['mask']
        self.input_shape = yml_conf['image_shape']
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
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


def load_predictor(model_dir, batch_size=1, use_gpu=False, min_subgraph_size=3):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        use_gpu (bool): whether use gpu
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need use_gpu == True.
    """
    run_mode = 'trt_int8'
    if not use_gpu and not run_mode == 'fluid':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect use_gpu==True, but use_gpu == {}"
            .format(run_mode, use_gpu))
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams'))
    precision_map = {
        'trt_int8': Config.Precision.Int8,
        'trt_fp32': Config.Precision.Float32,
        'trt_fp16': Config.Precision.Half
    }
    if use_gpu:
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
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
            use_calib_mode=True)

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor


def print_arguments(args):
    print('-----------  Running Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------')


def predict_image_dir(detector):
    for image_file in glob.glob(FLAGS.image_dir + '/*.jpg'):
        print('image_file is', image_file)
        results = detector.predict(image_file, threshold=0.5)


def main():
    pred_config = PredictConfig(FLAGS.model_dir)
    detector = Detector(pred_config, FLAGS.model_dir, use_gpu=FLAGS.use_gpu)
    # predict from image
    if FLAGS.image_dir != '':
        predict_image_dir(detector)


if __name__ == '__main__':
    paddle.enable_static()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help=("Directory include:'model.pdiparams', 'model.pdmodel', "
              "'infer_cfg.yml', created by tools/export_model.py."),
        required=True)
    parser.add_argument(
        "--image_dir", type=str, default='', help="Directory of image file.")
    parser.add_argument(
        "--use_gpu",
        type=ast.literal_eval,
        default=False,
        help="Whether to predict with GPU.")
    print('err?')
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory of output visualization files.")
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)

    main()
