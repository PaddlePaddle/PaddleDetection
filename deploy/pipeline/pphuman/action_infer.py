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
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from paddle.inference import Config, create_predictor
from python.utils import argsparser, Timer, get_current_memory_mb
from python.benchmark_utils import PaddleInferBenchmark
from python.infer import Detector, print_arguments
from attr_infer import AttrDetector


class SkeletonActionRecognizer(Detector):
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
        threshold (float): The threshold of score for visualization
        window_size(int): Temporal size of skeleton feature.
        random_pad (bool): Whether do random padding when frame length < window_size.
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
        assert batch_size == 1, "SkeletonActionRecognizer only support batch_size=1 now."
        super(SkeletonActionRecognizer, self).__init__(
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
            threshold=threshold,
            delete_shuffle_pass=True)

    @classmethod
    def init_with_cfg(cls, args, cfg):
        return cls(model_dir=cfg['model_dir'],
                   batch_size=cfg['batch_size'],
                   window_size=cfg['max_frames'],
                   device=args.device,
                   run_mode=args.run_mode,
                   trt_min_shape=args.trt_min_shape,
                   trt_max_shape=args.trt_max_shape,
                   trt_opt_shape=args.trt_opt_shape,
                   trt_calib_mode=args.trt_calib_mode,
                   cpu_threads=args.cpu_threads,
                   enable_mkldnn=args.enable_mkldnn)

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

    def predict_skeleton(self, skeleton_list, run_benchmark=False, repeats=1):
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

            results.append(result)
        return results

    def predict_skeleton_with_mot(self, skeleton_with_mot, run_benchmark=False):
        """
            skeleton_with_mot (dict): includes individual skeleton sequences, which shape is [C, T, K, 1]
                                      and its corresponding track id.
        """

        skeleton_list = skeleton_with_mot["skeleton"]
        mot_id = skeleton_with_mot["mot_id"]
        act_res = self.predict_skeleton(skeleton_list, run_benchmark, repeats=1)
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
    for operator in preprocess_ops:
        data = operator(data)
    return data


class AutoPadding(object):
    """
    Sample or Padding frame skeleton feature.
    Args:
        window_size (int): Temporal size of skeleton feature.
        random_pad (bool): Whether do random padding when frame length < window size. Default: False.
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


class DetActionRecognizer(object):
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
        threshold (float): The threshold of score for action feature object detection.
        display_frames (int): The duration for corresponding detected action.
        skip_frame_num (int): The number of frames for interval prediction. A skipped frame will 
            reuse the result of its last frame. If it is set to 0, no frame will be skipped. Default
            is 0.

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
                 display_frames=20,
                 skip_frame_num=0):
        super(DetActionRecognizer, self).__init__()
        self.detector = Detector(
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
        self.threshold = threshold
        self.frame_life = display_frames
        self.result_history = {}
        self.skip_frame_num = skip_frame_num
        self.skip_frame_cnt = 0
        self.id_in_last_frame = []

    @classmethod
    def init_with_cfg(cls, args, cfg):
        return cls(model_dir=cfg['model_dir'],
                   batch_size=cfg['batch_size'],
                   threshold=cfg['threshold'],
                   display_frames=cfg['display_frames'],
                   skip_frame_num=cfg['skip_frame_num'],
                   device=args.device,
                   run_mode=args.run_mode,
                   trt_min_shape=args.trt_min_shape,
                   trt_max_shape=args.trt_max_shape,
                   trt_opt_shape=args.trt_opt_shape,
                   trt_calib_mode=args.trt_calib_mode,
                   cpu_threads=args.cpu_threads,
                   enable_mkldnn=args.enable_mkldnn)

    def predict(self, images, mot_result):
        if self.skip_frame_cnt == 0 or (not self.check_id_is_same(mot_result)):
            det_result = self.detector.predict_image(images, visual=False)
            result = self.postprocess(det_result, mot_result)
        else:
            result = self.reuse_result(mot_result)

        self.skip_frame_cnt += 1
        if self.skip_frame_cnt >= self.skip_frame_num:
            self.skip_frame_cnt = 0

        return result

    def postprocess(self, det_result, mot_result):
        np_boxes_num = det_result['boxes_num']
        if np_boxes_num[0] <= 0:
            return [[], []]

        mot_bboxes = mot_result.get('boxes')

        cur_box_idx = 0
        mot_id = []
        act_res = []
        for idx in range(len(mot_bboxes)):
            tracker_id = mot_bboxes[idx, 0]

            # Current now,  class 0 is positive, class 1 is negative.
            action_ret = {'class': 1.0, 'score': -1.0}
            box_num = np_boxes_num[idx]
            boxes = det_result['boxes'][cur_box_idx:cur_box_idx + box_num]
            cur_box_idx += box_num
            isvalid = (boxes[:, 1] > self.threshold) & (boxes[:, 0] == 0)
            valid_boxes = boxes[isvalid, :]

            if valid_boxes.shape[0] >= 1:
                action_ret['class'] = valid_boxes[0, 0]
                action_ret['score'] = valid_boxes[0, 1]
                self.result_history[
                    tracker_id] = [0, self.frame_life, valid_boxes[0, 1]]
            else:
                history_det, life_remain, history_score = self.result_history.get(
                    tracker_id, [1, self.frame_life, -1.0])
                action_ret['class'] = history_det
                action_ret['score'] = -1.0
                life_remain -= 1
                if life_remain <= 0 and tracker_id in self.result_history:
                    del (self.result_history[tracker_id])
                elif tracker_id in self.result_history:
                    self.result_history[tracker_id][1] = life_remain
                else:
                    self.result_history[tracker_id] = [
                        history_det, life_remain, history_score
                    ]

            mot_id.append(tracker_id)
            act_res.append(action_ret)
        result = list(zip(mot_id, act_res))
        self.id_in_last_frame = mot_id

        return result

    def check_id_is_same(self, mot_result):
        mot_bboxes = mot_result.get('boxes')
        for idx in range(len(mot_bboxes)):
            tracker_id = mot_bboxes[idx, 0]
            if tracker_id not in self.id_in_last_frame:
                return False
        return True

    def reuse_result(self, mot_result):
        # This function reusing previous results of the same ID directly.
        mot_bboxes = mot_result.get('boxes')

        mot_id = []
        act_res = []

        for idx in range(len(mot_bboxes)):
            tracker_id = mot_bboxes[idx, 0]
            history_cls, life_remain, history_score = self.result_history.get(
                tracker_id, [1, 0, -1.0])

            life_remain -= 1
            if tracker_id in self.result_history:
                self.result_history[tracker_id][1] = life_remain

            action_ret = {'class': history_cls, 'score': history_score}
            mot_id.append(tracker_id)
            act_res.append(action_ret)

        result = list(zip(mot_id, act_res))
        self.id_in_last_frame = mot_id

        return result


class ClsActionRecognizer(AttrDetector):
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
        threshold (float): The threshold of score for action feature object detection.
        display_frames (int): The duration for corresponding detected action. 
        skip_frame_num (int): The number of frames for interval prediction. A skipped frame will 
            reuse the result of its last frame. If it is set to 0, no frame will be skipped. Default
            is 0.
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
                 display_frames=80,
                 skip_frame_num=0):
        super(ClsActionRecognizer, self).__init__(
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
        self.threshold = threshold
        self.frame_life = display_frames
        self.result_history = {}
        self.skip_frame_num = skip_frame_num
        self.skip_frame_cnt = 0
        self.id_in_last_frame = []

    @classmethod
    def init_with_cfg(cls, args, cfg):
        return cls(model_dir=cfg['model_dir'],
                   batch_size=cfg['batch_size'],
                   threshold=cfg['threshold'],
                   display_frames=cfg['display_frames'],
                   skip_frame_num=cfg['skip_frame_num'],
                   device=args.device,
                   run_mode=args.run_mode,
                   trt_min_shape=args.trt_min_shape,
                   trt_max_shape=args.trt_max_shape,
                   trt_opt_shape=args.trt_opt_shape,
                   trt_calib_mode=args.trt_calib_mode,
                   cpu_threads=args.cpu_threads,
                   enable_mkldnn=args.enable_mkldnn)

    def predict_with_mot(self, images, mot_result):
        if self.skip_frame_cnt == 0 or (not self.check_id_is_same(mot_result)):
            images = self.crop_half_body(images)
            cls_result = self.predict_image(images, visual=False)["output"]
            result = self.match_action_with_id(cls_result, mot_result)
        else:
            result = self.reuse_result(mot_result)

        self.skip_frame_cnt += 1
        if self.skip_frame_cnt >= self.skip_frame_num:
            self.skip_frame_cnt = 0

        return result

    def crop_half_body(self, images):
        crop_images = []
        for image in images:
            h = image.shape[0]
            crop_images.append(image[:h // 2 + 1, :, :])
        return crop_images

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        im_results = result['output']
        batch_res = []
        for res in im_results:
            action_res = res.tolist()
            for cid, score in enumerate(action_res):
                action_res[cid] = score
            batch_res.append(action_res)
        result = {'output': batch_res}
        return result

    def match_action_with_id(self, cls_result, mot_result):
        mot_bboxes = mot_result.get('boxes')

        mot_id = []
        act_res = []

        for idx in range(len(mot_bboxes)):
            tracker_id = mot_bboxes[idx, 0]

            cls_id_res = 1
            cls_score_res = -1.0
            for cls_id in range(len(cls_result[idx])):
                score = cls_result[idx][cls_id]
                if score > cls_score_res:
                    cls_id_res = cls_id
                    cls_score_res = score

            # Current now,  class 0 is positive, class 1 is negative.
            if cls_id_res == 1 or (cls_id_res == 0 and
                                   cls_score_res < self.threshold):
                history_cls, life_remain, history_score = self.result_history.get(
                    tracker_id, [1, self.frame_life, -1.0])
                cls_id_res = history_cls
                cls_score_res = 1 - cls_score_res
                life_remain -= 1
                if life_remain <= 0 and tracker_id in self.result_history:
                    del (self.result_history[tracker_id])
                elif tracker_id in self.result_history:
                    self.result_history[tracker_id][1] = life_remain
                else:
                    self.result_history[
                        tracker_id] = [cls_id_res, life_remain, cls_score_res]
            else:
                self.result_history[
                    tracker_id] = [cls_id_res, self.frame_life, cls_score_res]

            action_ret = {'class': cls_id_res, 'score': cls_score_res}
            mot_id.append(tracker_id)
            act_res.append(action_ret)
        result = list(zip(mot_id, act_res))
        self.id_in_last_frame = mot_id

        return result

    def check_id_is_same(self, mot_result):
        mot_bboxes = mot_result.get('boxes')
        for idx in range(len(mot_bboxes)):
            tracker_id = mot_bboxes[idx, 0]
            if tracker_id not in self.id_in_last_frame:
                return False
        return True

    def reuse_result(self, mot_result):
        # This function reusing previous results of the same ID directly.
        mot_bboxes = mot_result.get('boxes')

        mot_id = []
        act_res = []

        for idx in range(len(mot_bboxes)):
            tracker_id = mot_bboxes[idx, 0]
            history_cls, life_remain, history_score = self.result_history.get(
                tracker_id, [1, 0, -1.0])

            life_remain -= 1
            if tracker_id in self.result_history:
                self.result_history[tracker_id][1] = life_remain

            action_ret = {'class': history_cls, 'score': history_score}
            mot_id.append(tracker_id)
            act_res.append(action_ret)

        result = list(zip(mot_id, act_res))
        self.id_in_last_frame = mot_id

        return result


def main():
    detector = SkeletonActionRecognizer(
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
        det_log('SkeletonAction')


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
