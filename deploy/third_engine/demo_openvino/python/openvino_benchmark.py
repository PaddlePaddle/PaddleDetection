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

import cv2
import numpy as np
import time
import argparse
from openvino.runtime import Core


def image_preprocess(img_path, re_shape):
    img = cv2.imread(img_path)
    img = cv2.resize(
        img, (re_shape, re_shape), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, [2, 0, 1]) / 255
    img = np.expand_dims(img, 0)
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    return img.astype(np.float32)


def benchmark(img_file, onnx_file, re_shape):

    ie = Core()
    net = ie.read_model(onnx_file)

    test_image = image_preprocess(img_file, re_shape)

    compiled_model = ie.compile_model(net, 'CPU')

    # benchmark       
    loop_num = 100
    warm_up = 8
    timeall = 0
    time_min = float("inf")
    time_max = float('-inf')

    for i in range(loop_num + warm_up):
        time0 = time.time()
        #perform the inference step

        output = compiled_model.infer_new_request({0: test_image})
        time1 = time.time()
        timed = time1 - time0

        if i >= warm_up:
            timeall = timeall + timed
            time_min = min(time_min, timed)
            time_max = max(time_max, timed)

    time_avg = timeall / loop_num

    print('inference_time(ms): min={}, max={}, avg={}'.format(
        round(time_min * 1000, 2),
        round(time_max * 1000, 1), round(time_avg * 1000, 1)))


if __name__ == '__main__':

    onnx_path = "out_onnx"
    onnx_file = onnx_path + "/picodet_s_320_coco.onnx"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_path',
        type=str,
        default='demo/000000570688.jpg',
        help="image path")
    parser.add_argument(
        '--onnx_path',
        type=str,
        default='out_onnxsim/picodet_xs_320_coco_lcnet.onnx',
        help="onnx filepath")
    parser.add_argument('--in_shape', type=int, default=320, help="input_size")

    args = parser.parse_args()
    benchmark(args.img_path, args.onnx_path, args.in_shape)
