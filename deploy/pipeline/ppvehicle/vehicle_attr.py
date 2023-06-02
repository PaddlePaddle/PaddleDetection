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
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

# add deploy path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)

from paddle.inference import Config, create_predictor
from python.utils import argsparser, Timer, get_current_memory_mb
from python.benchmark_utils import PaddleInferBenchmark
from python.infer import Detector, print_arguments
from pipeline.pphuman.attr_infer import AttrDetector


class VehicleAttr(AttrDetector):
    """
    Args:
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU/NPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
        type_threshold (float): The threshold of score for vehicle type recognition.
        color_threshold (float): The threshold of score for vehicle color recognition.
    """

    def __init__(self,
                 model_dir,
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
                 color_threshold=0.5,
                 type_threshold=0.5):
        super(VehicleAttr, self).__init__(
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
            output_dir=output_dir)
        self.color_threshold = color_threshold
        self.type_threshold = type_threshold
        self.result_history = {}
        self.color_list = [
            "yellow", "orange", "green", "gray", "red", "blue", "white",
            "golden", "brown", "black"
        ]
        self.type_list = [
            "sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus", "truck",
            "estate"
        ]

    @classmethod
    def init_with_cfg(cls, args, cfg):
        return cls(model_dir=cfg['model_dir'],
                   batch_size=cfg['batch_size'],
                   color_threshold=cfg['color_threshold'],
                   type_threshold=cfg['type_threshold'],
                   device=args.device,
                   run_mode=args.run_mode,
                   trt_min_shape=args.trt_min_shape,
                   trt_max_shape=args.trt_max_shape,
                   trt_opt_shape=args.trt_opt_shape,
                   trt_calib_mode=args.trt_calib_mode,
                   cpu_threads=args.cpu_threads,
                   enable_mkldnn=args.enable_mkldnn)

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        im_results = result['output']
        batch_res = []
        for res in im_results:
            res = res.tolist()
            attr_res = []
            color_res_str = "Color: "
            type_res_str = "Type: "
            color_idx = np.argmax(res[:10])
            type_idx = np.argmax(res[10:])

            if res[color_idx] >= self.color_threshold:
                color_res_str += self.color_list[color_idx]
            else:
                color_res_str += "Unknown"
            attr_res.append(color_res_str)

            if res[type_idx + 10] >= self.type_threshold:
                type_res_str += self.type_list[type_idx]
            else:
                type_res_str += "Unknown"
            attr_res.append(type_res_str)

            batch_res.append(attr_res)
        result = {'output': batch_res}
        return result


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU', 'NPU'
                            ], "device should be CPU, GPU, NPU or XPU"
    assert not FLAGS.use_gpu, "use_gpu has been deprecated, please use --device"

    main()
