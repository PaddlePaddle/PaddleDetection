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

from paddle.inference import Config, create_predictor
from utils import argsparser, Timer, get_current_memory_mb
from benchmark_utils import PaddleInferBenchmark
from infer import Detector, print_arguments

SUPPORT_MODELS = []


class ActionRecognizer(Detector):
    """
    Args:
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
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
        threshold (float): The threshold of score for visualization
        window_size(int): Temporal size of skeleton feature.
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
                 threshold=0.5,
                 window_size=100,
                 random_pad=False):
        super(ActionRecognizer, self).__init__(
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
            threshold=threshold)

    def set_config(self, model_dir):
        return PredictConfig_Action(model_dir)

    def predict(self, repeats=1):
        '''
        Args:
            repeats (int): repeat number for prediction
        Returns:
            results (dict): 
        '''
        # model prediction
        output_names = self.predictor.get_output_names()
        for i in range(repeats):
            self.predictor.run()
            output_tensor = self.predictor.get_output_handle(output_names[0])
            np_output = output_tensor.copy_to_cpu()
        result = dict(output=np_output)
        return result

    def predict_skeleton(self,
                         skeleton_list,
                         run_benchmark=False,
                         repeats=1,
                         visual=True):
        results = []
        for i, skeleton in enumerate(skeleton_list):
            if run_benchmark:
                # preprocess
                inputs = self.preprocess(skeleton)  # warmup
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(skeleton)
                self.det_times.preprocess_time_s.end()

                # model prediction
                result = self.predict(repeats=repeats)  # warmup
                self.det_times.inference_time_s.start()
                result = self.predict(repeats=repeats)
                self.det_times.inference_time_s.end(repeats=repeats)

                # postprocess
                result_warmup = self.postprocess(inputs, result)  # warmup
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(skeleton)

                cm, gm, gu = get_current_memory_mb()
                self.cpu_mem += cm
                self.gpu_mem += gm
                self.gpu_util += gu
            else:
                # preprocess
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(skeleton)
                self.det_times.preprocess_time_s.end()

                # model prediction
                self.det_times.inference_time_s.start()
                result = self.predict()
                self.det_times.inference_time_s.end()

                # postprocess
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(skeleton)

                if visual:
                    pass
                    #visualize_action(skeleton, result, output_dir=self.output_dir)
            results.append(result)
            if visual:
                print('Test iter {}'.format(i))

        #results = self.merge_batch_result(results)
        return results

    def predict_skeleton_with_mot(self,
                                  skeleton_with_mot,
                                  run_benchmark=False,
                                  visual=True):
        """
            skeleton_with_mot (dict): includes individual skeleton sequences, which shape is [C, T, K, 1]
                                      and its corresponding track id.
        """

        skeleton_list = skeleton_with_mot["skeleton"]
        mot_id = skeleton_with_mot["mot_id"]
        act_res = self.predict_skeleton(
            skeleton_list, run_benchmark, repeats=1, visual=False)
        if visual:
            pass
        results = list(zip(mot_id, act_res))
        return results

    def preprocess(self, data):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_lst = []
        data = action_preprocess(data, preprocess_ops)
        input_lst.append(data)
        input_names = self.predictor.get_input_names()
        inputs = {}
        inputs['data_batch_0'] = np.stack(input_lst, axis=0).astype('float32')

        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        return inputs

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        output_logit = result['output'][0]
        classes = np.argpartition(output_logit, -1)[-1:]
        classes = classes[np.argsort(-output_logit[classes])]
        scores = output_logit[classes]
        result = {'class': classes, 'score': scores}
        return result


def action_preprocess(input, preprocess_ops):
    """
    input (str | numpy.array): if input is str, it should be a legal file path with numpy array saved.
                               Otherwise it should be numpy.array as direct input.
    return (numpy.array)
    """
    if isinstance(input, str):
        assert os.path.isfile(input) is not None, "{0} not exists".format(input)
        data = np.load(input)
    else:
        data = input
    #data = self.pad_op(data)
    for operator in preprocess_ops:
        data = operator(data)
    return data


class AutoPadding(object):
    """
    Sample or Padding frame skeleton feature.
    Args:
        window_size: int, temporal size of skeleton feature.
        random_pad: bool, whether do random padding when frame length < window size. Default: False.
    """

    def __init__(self, window_size=100, random_pad=False):
        self.window_size = window_size
        self.random_pad = random_pad

    def get_frame_num(self, data):
        C, T, V, M = data.shape
        for i in range(T - 1, -1, -1):
            tmp = np.sum(data[:, i, :, :])
            if tmp > 0:
                T = i + 1
                break
        return T

    def __call__(self, results):
        data = results

        C, T, V, M = data.shape
        T = self.get_frame_num(data)
        if T == self.window_size:
            data_pad = data[:, :self.window_size, :, :]
        elif T < self.window_size:
            begin = random.randint(
                0, self.window_size - T) if self.random_pad else 0
            data_pad = np.zeros((C, self.window_size, V, M))
            data_pad[:, begin:begin + T, :, :] = data[:, :T, :, :]
        else:
            if self.random_pad:
                index = np.random.choice(
                    T, self.window_size, replace=False).astype('int64')
            else:
                index = np.linspace(0, T, self.window_size).astype("int64")
            data_pad = data[:, index, :, :]

        return data_pad


class PredictConfig_Action(object):
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        #self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        #self.archcls = SUPPORT_MODELS[yml_conf['arch']]
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
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
            'arch'], KEYPOINT_SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


class KeyPointSequence(object):
    def __init__(self, max_size=100):
        self.frames = 0
        self.kpts = []
        self.bboxes = []
        self.max_size = max_size

    def save(self, kpt, bbox):
        self.kpts.append(kpt)
        self.bboxes.append(bbox)
        self.frames += 1
        if self.frames == self.max_size:
            return True
        return False


class KeyPointCollector(object):
    def __init__(self, max_size=100):
        self.flag_track_interrupt = False
        self.keypoint_saver = dict()
        self.max_size = max_size
        self.id_to_pop = set()
        self.flag_to_pop = False

    def get_state(self):
        return self.flag_to_pop

    def update(self, kpt_res, mot_res):
        kpts = kpt_res.get('keypoint')[0]
        bboxes = kpt_res.get('bbox')
        mot_bboxes = mot_res.get('boxes')
        updated_id = set()

        for idx in range(len(kpts)):
            tracker_id = mot_bboxes[idx, 0]
            updated_id.add(tracker_id)

            kpt_seq = self.keypoint_saver.get(tracker_id,
                                              KeyPointSequence(self.max_size))
            is_full = kpt_seq.save(kpts[idx], bboxes[idx])
            self.keypoint_saver[tracker_id] = kpt_seq

            #Scene1: result should be popped when frames meet max size
            if is_full:
                self.id_to_pop.add(tracker_id)
                self.flag_to_pop = True

        #Scene2: result of a lost tracker should be popped
        interrupted_id = set(self.keypoint_saver.keys()) - updated_id
        if len(interrupted_id) > 0:
            self.flag_to_pop = True
            self.id_to_pop.update(interrupted_id)

    def get_collected_keypoint(self):
        """
            Output (List): List of keypoint results for Action Recognition task, where 
                           the format of each element is [tracker_id, KeyPointSequence of tracker_id]
        """
        output = []
        for tracker_id in self.id_to_pop:
            output.append([tracker_id, self.keypoint_saver[tracker_id]])
            del (self.keypoint_saver[tracker_id])
        self.flag_to_pop = False
        self.id_to_pop.clear()
        return output


def get_test_skeletons(input_file):
    assert input_file is not None, "--action_file can not be None"
    input_data = np.load(input_file)
    if input_data.ndim == 4:
        return [input_data]
    elif input_data.ndim == 5:
        output = list(
            map(lambda x: np.squeeze(x, 0),
                np.split(input_data, input_data.shape[0], 0)))
        return output
    else:
        raise ValueError(
            "Now only support input with shape: (N, C, T, K, M) or (C, T, K, M)")


def main():
    detector = ActionRecognizer(
        FLAGS.model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        batch_size=FLAGS.batch_size,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn,
        threshold=FLAGS.threshold,
        output_dir=FLAGS.output_dir,
        window_size=FLAGS.window_size,
        random_pad=FLAGS.random_pad)
    # predict from numpy array
    input_list = get_test_skeletons(FLAGS.action_file)
    detector.predict_skeleton(input_list, FLAGS.run_benchmark, repeats=10)
    if not FLAGS.run_benchmark:
        detector.det_times.info(average=True)
    else:
        mems = {
            'cpu_rss_mb': detector.cpu_mem / len(input_list),
            'gpu_rss_mb': detector.gpu_mem / len(input_list),
            'gpu_util': detector.gpu_util * 100 / len(input_list)
        }

        perf_info = detector.det_times.report(average=True)
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
        det_log = PaddleInferBenchmark(detector.config, model_info, data_info,
                                       perf_info, mems)
        det_log('Action')


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
