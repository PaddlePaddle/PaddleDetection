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
import paddle.nn.functional as F
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

# add deploy path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from paddle.inference import Config, create_predictor
from python.utils import argsparser, Timer, get_current_memory_mb
from python.benchmark_utils import PaddleInferBenchmark
from python.infer import Detector, print_arguments
from video_action_preprocess import VideoDecoder, Sampler, Scale, CenterCrop, Normalization, Image2Array


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


class VideoActionRecognizer(object):
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
    """

    def __init__(self,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 num_seg=8,
                 seg_len=1,
                 short_size=256,
                 target_size=224,
                 top_k=1,
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 ir_optim=True):

        self.num_seg = num_seg
        self.seg_len = seg_len
        self.short_size = short_size
        self.target_size = target_size
        self.top_k = top_k

        assert batch_size == 1, "VideoActionRecognizer only support batch_size=1 now."

        self.model_dir = model_dir
        self.device = device
        self.run_mode = run_mode
        self.batch_size = batch_size
        self.trt_min_shape = trt_min_shape
        self.trt_max_shape = trt_max_shape
        self.trt_opt_shape = trt_opt_shape
        self.trt_calib_mode = trt_calib_mode
        self.cpu_threads = cpu_threads
        self.enable_mkldnn = enable_mkldnn
        self.ir_optim = ir_optim

        self.recognize_times = Timer()

        model_file_path = glob.glob(os.path.join(model_dir, "*.pdmodel"))[0]
        params_file_path = glob.glob(os.path.join(model_dir, "*.pdiparams"))[0]
        self.config = Config(model_file_path, params_file_path)

        if device == "GPU" or device == "gpu":
            self.config.enable_use_gpu(8000, 0)
        elif device == "XPU" or device == "xpu":
            self.config.enable_xpu(10 * 1024 * 1024)
        elif device == "NPU" or device == "npu":
            self.config.enable_custom_device('npu')
        else:
            self.config.disable_gpu()
        if self.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            self.config.set_mkldnn_cache_capacity(10)
            self.config.enable_mkldnn()

        self.config.switch_ir_optim(self.ir_optim)  # default true

        precision_map = {
            'trt_int8': Config.Precision.Int8,
            'trt_fp32': Config.Precision.Float32,
            'trt_fp16': Config.Precision.Half
        }
        if run_mode in precision_map.keys():
            self.config.enable_tensorrt_engine(
                max_batch_size=8, precision_mode=precision_map[run_mode])

        self.config.enable_memory_optim()
        # use zero copy
        self.config.switch_use_feed_fetch_ops(False)

        self.predictor = create_predictor(self.config)

    @classmethod
    def init_with_cfg(cls, args, cfg):
        return cls(model_dir=cfg['model_dir'],
                   short_size=cfg['short_size'],
                   target_size=cfg['target_size'],
                   batch_size=cfg['batch_size'],
                   device=args.device,
                   run_mode=args.run_mode,
                   trt_min_shape=args.trt_min_shape,
                   trt_max_shape=args.trt_max_shape,
                   trt_opt_shape=args.trt_opt_shape,
                   trt_calib_mode=args.trt_calib_mode,
                   cpu_threads=args.cpu_threads,
                   enable_mkldnn=args.enable_mkldnn)

    def preprocess_batch(self, file_list):
        batched_inputs = []
        for file in file_list:
            inputs = self.preprocess(file)
            batched_inputs.append(inputs)
        batched_inputs = [
            np.concatenate([item[i] for item in batched_inputs])
            for i in range(len(batched_inputs[0]))
        ]
        self.input_file = file_list
        return batched_inputs

    def get_timer(self):
        return self.recognize_times

    def predict(self, input):
        '''
        Args:
            input (str) or (list): video file path or image data list
        Returns:
            results (dict): 
        '''

        input_names = self.predictor.get_input_names()
        input_tensor = self.predictor.get_input_handle(input_names[0])

        output_names = self.predictor.get_output_names()
        output_tensor = self.predictor.get_output_handle(output_names[0])

        # preprocess
        self.recognize_times.preprocess_time_s.start()
        if type(input) == str:
            inputs = self.preprocess_video(input)
        else:
            inputs = self.preprocess_frames(input)
        self.recognize_times.preprocess_time_s.end()

        inputs = np.expand_dims(
            inputs, axis=0).repeat(
                self.batch_size, axis=0).copy()

        input_tensor.copy_from_cpu(inputs)

        # model prediction
        self.recognize_times.inference_time_s.start()
        self.predictor.run()
        self.recognize_times.inference_time_s.end()

        output = output_tensor.copy_to_cpu()

        # postprocess
        self.recognize_times.postprocess_time_s.start()
        classes, scores = self.postprocess(output)
        self.recognize_times.postprocess_time_s.end()

        return classes, scores

    def preprocess_frames(self, frame_list):
        """
        frame_list: list, frame list
        return: list
        """

        results = {}
        results['frames_len'] = len(frame_list)
        results["imgs"] = frame_list

        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        ops = [
            CenterCrop(self.target_size), Image2Array(),
            Normalization(img_mean, img_std)
        ]
        for op in ops:
            results = op(results)

        res = np.expand_dims(results['imgs'], axis=0).copy()
        return [res]

    def preprocess_video(self, input_file):
        """
        input_file: str, file path
        return: list
        """
        assert os.path.isfile(input_file) is not None, "{0} not exists".format(
            input_file)

        results = {'filename': input_file}
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        ops = [
            VideoDecoder(), Sampler(
                self.num_seg, self.seg_len, valid_mode=True),
            Scale(self.short_size), CenterCrop(self.target_size), Image2Array(),
            Normalization(img_mean, img_std)
        ]
        for op in ops:
            results = op(results)

        res = np.expand_dims(results['imgs'], axis=0).copy()
        return [res]

    def postprocess(self, output):
        output = output.flatten()  # numpy.ndarray
        output = softmax(output)
        classes = np.argpartition(output, -self.top_k)[-self.top_k:]
        classes = classes[np.argsort(-output[classes])]
        scores = output[classes]
        return classes, scores


def main():
    if not FLAGS.run_benchmark:
        assert FLAGS.batch_size == 1
        assert FLAGS.use_fp16 is False
    else:
        assert FLAGS.use_gpu is True

    recognizer = VideoActionRecognizer(
        FLAGS.model_dir,
        short_size=FLAGS.short_size,
        target_size=FLAGS.target_size,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        batch_size=FLAGS.batch_size,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn, )

    if not FLAGS.run_benchmark:
        classes, scores = recognizer.predict(FLAGS.video_file)
        print("Current video file: {}".format(FLAGS.video_file))
        print("\ttop-1 class: {0}".format(classes[0]))
        print("\ttop-1 score: {0}".format(scores[0]))
    else:
        cm, gm, gu = get_current_memory_mb()
        mems = {'cpu_rss_mb': cm, 'gpu_rss_mb': gm, 'gpu_util': gu * 100}

        perf_info = recognizer.recognize_times.report()
        model_dir = FLAGS.model_dir
        mode = FLAGS.run_mode
        model_info = {
            'model_name': model_dir.strip('/').split('/')[-1],
            'precision': mode.split('_')[-1]
        }
        data_info = {
            'batch_size': FLAGS.batch_size,
            'shape': "dynamic_shape",
            'data_num': perf_info['img_num']
        }
        recognize_log = PaddleInferBenchmark(recognizer.config, model_info,
                                             data_info, perf_info, mems)
        recognize_log('Fight')


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU', 'NPU'
                            ], "device should be CPU, GPU, XPU or NPU"

    main()
