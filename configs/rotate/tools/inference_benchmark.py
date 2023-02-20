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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import six
import glob
import time
import yaml
import argparse
import cv2
import numpy as np

import paddle
import paddle.version as paddle_version
from paddle.inference import Config, create_predictor, PrecisionType, get_trt_runtime_version

TUNED_TRT_DYNAMIC_MODELS = {'DETR'}


def check_version(version='2.2'):
    err = "PaddlePaddle version {} or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code.".format(version)

    version_installed = [
        paddle_version.major, paddle_version.minor, paddle_version.patch,
        paddle_version.rc
    ]

    if version_installed == ['0', '0', '0', '0']:
        return

    if version == 'develop':
        raise Exception("PaddlePaddle develop version is required!")

    version_split = version.split('.')

    length = min(len(version_installed), len(version_split))
    for i in six.moves.range(length):
        if version_installed[i] > version_split[i]:
            return
        if version_installed[i] < version_split[i]:
            raise Exception(err)


def check_trt_version(version='8.2'):
    err = "TensorRT version {} or higher is required," \
          "Please make sure the version is good with your code.".format(version)
    version_split = list(map(int, version.split('.')))
    version_installed = get_trt_runtime_version()
    length = min(len(version_installed), len(version_split))
    for i in six.moves.range(length):
        if version_installed[i] > version_split[i]:
            return
        if version_installed[i] < version_split[i]:
            raise Exception(err)


# preprocess ops
def decode_image(im_file, im_info):
    if isinstance(im_file, str):
        with open(im_file, 'rb') as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im = im_file
    im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
    im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
    return im, im_info


class Resize(object):
    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_LINEAR):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp

    def __call__(self, im, im_info):
        assert len(self.target_size) == 2
        assert self.target_size[0] > 0 and self.target_size[1] > 0
        im_channel = im.shape[2]
        im_scale_y, im_scale_x = self.generate_scale(im)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)
        im_info['im_shape'] = np.array(im.shape[:2]).astype('float32')
        im_info['scale_factor'] = np.array(
            [im_scale_y, im_scale_x]).astype('float32')
        return im, im_info

    def generate_scale(self, im):
        origin_shape = im.shape[:2]
        im_c = im.shape[2]
        if self.keep_ratio:
            im_size_min = np.min(origin_shape)
            im_size_max = np.max(origin_shape)
            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)
            im_scale = float(target_size_min) / float(im_size_min)
            if np.round(im_scale * im_size_max) > target_size_max:
                im_scale = float(target_size_max) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / float(origin_shape[0])
            im_scale_x = resize_w / float(origin_shape[1])
        return im_scale_y, im_scale_x


class Permute(object):
    def __init__(self, ):
        super(Permute, self).__init__()

    def __call__(self, im, im_info):
        im = im.transpose((2, 0, 1))
        return im, im_info


class NormalizeImage(object):
    def __init__(self, mean, std, is_scale=True, norm_type='mean_std'):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type

    def __call__(self, im, im_info):
        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            scale = 1.0 / 255.0
            im *= scale

        if self.norm_type == 'mean_std':
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im -= mean
            im /= std
        return im, im_info


class PadStride(object):
    def __init__(self, stride=0):
        self.coarsest_stride = stride

    def __call__(self, im, im_info):
        coarsest_stride = self.coarsest_stride
        if coarsest_stride <= 0:
            return im, im_info
        im_c, im_h, im_w = im.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        return padding_im, im_info


def preprocess(im, preprocess_ops):
    # process image by preprocess_ops
    im_info = {
        'scale_factor': np.array(
            [1., 1.], dtype=np.float32),
        'im_shape': None,
    }
    im, im_info = decode_image(im, im_info)
    for operator in preprocess_ops:
        im, im_info = operator(im, im_info)
    return im, im_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir', type=str, help='directory of inference model')
    parser.add_argument(
        '--run_mode', type=str, default='paddle', help='running mode')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument(
        '--image_dir',
        type=str,
        default='/paddle/data/DOTA_1024_ss/test1024/images',
        help='directory of test images')
    parser.add_argument(
        '--warmup_iter', type=int, default=5, help='num of warmup iters')
    parser.add_argument(
        '--total_iter', type=int, default=2000, help='num of total iters')
    parser.add_argument(
        '--log_iter', type=int, default=50, help='num of log interval')
    parser.add_argument(
        '--tuned_trt_shape_file',
        type=str,
        default='shape_range_info.pbtxt',
        help='dynamic shape range info')
    args = parser.parse_args()
    return args


def init_predictor(FLAGS):
    model_dir, run_mode, batch_size = FLAGS.model_dir, FLAGS.run_mode, FLAGS.batch_size
    yaml_file = os.path.join(model_dir, 'infer_cfg.yml')
    with open(yaml_file) as f:
        yml_conf = yaml.safe_load(f)

    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams'))

    # initial GPU memory(M), device ID
    config.enable_use_gpu(200, 0)
    # optimize graph and fuse op
    config.switch_ir_optim(True)

    precision_map = {
        'trt_int8': Config.Precision.Int8,
        'trt_fp32': Config.Precision.Float32,
        'trt_fp16': Config.Precision.Half
    }

    arch = yml_conf['arch']
    tuned_trt_shape_file = os.path.join(model_dir, FLAGS.tuned_trt_shape_file)

    if run_mode in precision_map.keys():
        if arch in TUNED_TRT_DYNAMIC_MODELS and not os.path.exists(
                tuned_trt_shape_file):
            print(
                'dynamic shape range info is saved in {}. After that, rerun the code'.
                format(tuned_trt_shape_file))
            config.collect_shape_range_info(tuned_trt_shape_file)
        config.enable_tensorrt_engine(
            workspace_size=(1 << 25) * batch_size,
            max_batch_size=batch_size,
            min_subgraph_size=yml_conf['min_subgraph_size'],
            precision_mode=precision_map[run_mode],
            use_static=True,
            use_calib_mode=False)

        if yml_conf['use_dynamic_shape']:
            if arch in TUNED_TRT_DYNAMIC_MODELS and os.path.exists(
                    tuned_trt_shape_file):
                config.enable_tuned_tensorrt_dynamic_shape(tuned_trt_shape_file,
                                                           True)
            else:
                min_input_shape = {
                    'image': [batch_size, 3, 640, 640],
                    'scale_factor': [batch_size, 2]
                }
                max_input_shape = {
                    'image': [batch_size, 3, 1280, 1280],
                    'scale_factor': [batch_size, 2]
                }
                opt_input_shape = {
                    'image': [batch_size, 3, 1024, 1024],
                    'scale_factor': [batch_size, 2]
                }
                config.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape)

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor, yml_conf


def create_preprocess_ops(yml_conf):
    preprocess_ops = []
    for op_info in yml_conf['Preprocess']:
        new_op_info = op_info.copy()
        op_type = new_op_info.pop('type')
        preprocess_ops.append(eval(op_type)(**new_op_info))
    return preprocess_ops


def get_test_images(image_dir):
    images = set()
    infer_dir = os.path.abspath(image_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)
    return images


def create_inputs(image_files, preprocess_ops):
    inputs = dict()
    im_list, im_info_list = [], []
    for im_path in image_files:
        im, im_info = preprocess(im_path, preprocess_ops)
        im_list.append(im)
        im_info_list.append(im_info)

    inputs['im_shape'] = np.stack(
        [e['im_shape'] for e in im_info_list], axis=0).astype('float32')
    inputs['scale_factor'] = np.stack(
        [e['scale_factor'] for e in im_info_list], axis=0).astype('float32')
    inputs['image'] = np.stack(im_list, axis=0).astype('float32')
    return inputs


def measure_speed(FLAGS):
    predictor, yml_conf = init_predictor(FLAGS)
    input_names = predictor.get_input_names()
    preprocess_ops = create_preprocess_ops(yml_conf)

    image_files = get_test_images(FLAGS.image_dir)

    batch_size = FLAGS.batch_size
    warmup_iter, log_iter, total_iter = FLAGS.warmup_iter, FLAGS.log_iter, FLAGS.total_iter

    total_time = 0
    fps = 0
    for i in range(0, total_iter, batch_size):
        # make data ready
        inputs = create_inputs(image_files[i:i + batch_size], preprocess_ops)
        for name in input_names:
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(inputs[name])

        paddle.device.cuda.synchronize()
        # start running
        start_time = time.perf_counter()
        predictor.run()
        paddle.device.cuda.synchronize()

        if i >= warmup_iter:
            total_time += time.perf_counter() - start_time
            if (i + 1) % log_iter == 0:
                fps = (i + 1 - warmup_iter) / total_time
                print(
                    f'Done image [{i + 1:<3}/ {total_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == total_iter:
            fps = (i + 1 - warmup_iter) / total_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break


if __name__ == '__main__':
    FLAGS = parse_args()
    if 'trt' in FLAGS.run_mode:
        check_version('develop')
        check_trt_version('8.2')
    else:
        check_version('2.4')
    measure_speed(FLAGS)
