# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import cv2
import numpy as np
import paddle
from benchmark_utils import PaddleInferBenchmark
from preprocess import preprocess

from tracker import JDETracker
from ppdet.modeling.mot import visualization as mot_vis
from ppdet.modeling.mot.utils import Timer as MOTTimer

from paddle.inference import Config
from paddle.inference import create_predictor
from utils import argsparser, Timer, get_current_memory_mb
from infer import Detector, get_test_images, print_arguments, PredictConfig

# Global dictionary
MOT_SUPPORT_MODELS = {
    'JDE',
    'FairMOT',
}


class JDE_Detector(Detector):
    """
    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16)
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
                 pred_config,
                 model_dir,
                 device='CPU',
                 run_mode='fluid',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1088,
                 trt_opt_shape=608,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False):
        super(JDE_Detector, self).__init__(
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
        assert batch_size == 1, "The JDE Detector only supports batch size=1 now"
        assert pred_config.tracker, "Tracking model should have tracker"
        tp = pred_config.tracker
        min_box_area = tp['min_box_area'] if 'min_box_area' in tp else 200
        vertical_ratio = tp['vertical_ratio'] if 'vertical_ratio' in tp else 1.6
        conf_thres = tp['conf_thres'] if 'conf_thres' in tp else 0.
        tracked_thresh = tp['tracked_thresh'] if 'tracked_thresh' in tp else 0.7
        metric_type = tp['metric_type'] if 'metric_type' in tp else 'euclidean'
        self.tracker = JDETracker(
            min_box_area=min_box_area,
            vertical_ratio=vertical_ratio,
            conf_thres=conf_thres,
            tracked_thresh=tracked_thresh,
            metric_type=metric_type)

    def postprocess(self, pred_dets, pred_embs, threshold):
        online_targets = self.tracker.update(pred_dets, pred_embs)
        if online_targets == []:
            return [pred_dets[0][:4]], [pred_dets[0][4]], [1]
        online_tlwhs, online_ids = [], []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            tscore = t.score
            if tscore < threshold: continue
            if tlwh[2] * tlwh[3] <= self.tracker.min_box_area: continue
            if self.tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                    3] > self.tracker.vertical_ratio:
                continue
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(tscore)
        return online_tlwhs, online_scores, online_ids

    def predict(self, image_list, threshold=0.5, warmup=0, repeats=1):
        '''
        Args:
            image_list (list): list of image
            threshold (float): threshold of predicted box' score
        Returns:
            online_tlwhs, online_scores, online_ids (np.ndarray)
        '''
        self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(image_list)
        self.det_times.preprocess_time_s.end()

        pred_dets, pred_embs = None, None
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            pred_dets = boxes_tensor.copy_to_cpu()

        self.det_times.inference_time_s.start()
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            pred_dets = boxes_tensor.copy_to_cpu()
            embs_tensor = self.predictor.get_output_handle(output_names[1])
            pred_embs = embs_tensor.copy_to_cpu()
        self.det_times.inference_time_s.end(repeats=repeats)

        self.det_times.postprocess_time_s.start()
        online_tlwhs, online_scores, online_ids = self.postprocess(
            pred_dets, pred_embs, threshold)
        self.det_times.postprocess_time_s.end()
        self.det_times.img_num += 1
        return online_tlwhs, online_scores, online_ids


def write_mot_results(filename, results, data_type='mot'):
    if data_type in ['mot', 'mcmot', 'lab']:
        save_format = '{frame},{id},{x1},{y1},{w},{h},{score},-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, tscores, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, score, track_id in zip(tlwhs, tscores, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    w=w,
                    h=h,
                    score=score)
                f.write(line)


def predict_image(detector, image_list):
    results = []
    image_list.sort()
    for i, img_file in enumerate(image_list):
        frame = cv2.imread(img_file)
        if FLAGS.run_benchmark:
            detector.predict([frame], FLAGS.threshold, warmup=10, repeats=10)
            cm, gm, gu = get_current_memory_mb()
            detector.cpu_mem += cm
            detector.gpu_mem += gm
            detector.gpu_util += gu
            print('Test iter {}, file name:{}'.format(i, img_file))
        else:
            online_tlwhs, online_scores, online_ids = detector.predict(
                [frame], FLAGS.threshold)
            online_im = mot_vis.plot_tracking(
                frame, online_tlwhs, online_ids, online_scores, frame_id=i)
            if FLAGS.save_images:
                if not os.path.exists(FLAGS.output_dir):
                    os.makedirs(FLAGS.output_dir)
                img_name = os.path.split(img_file)[-1]
                out_path = os.path.join(FLAGS.output_dir, img_name)
                cv2.imwrite(out_path, online_im)
                print("save result to: " + out_path)


def predict_video(detector, camera_id):
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
        video_name = 'mot_output.mp4'
    else:
        capture = cv2.VideoCapture(FLAGS.video_file)
        video_name = os.path.split(FLAGS.video_file)[-1]
    fps = 30
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frame_count', frame_count)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # yapf: disable
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # yapf: enable
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    out_path = os.path.join(FLAGS.output_dir, video_name)
    if not FLAGS.save_images:
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    frame_id = 0
    timer = MOTTimer()
    results = []
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        timer.tic()
        online_tlwhs, online_scores, online_ids = detector.predict(
            [frame], FLAGS.threshold)
        timer.toc()

        results.append((frame_id + 1, online_tlwhs, online_scores, online_ids))
        fps = 1. / timer.average_time
        im = mot_vis.plot_tracking(
            frame,
            online_tlwhs,
            online_ids,
            online_scores,
            frame_id=frame_id,
            fps=fps)
        if FLAGS.save_images:
            save_dir = os.path.join(FLAGS.output_dir, video_name.split('.')[-2])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(
                os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), im)
        else:
            writer.write(im)
        frame_id += 1
        print('detect frame:%d' % (frame_id))
        if camera_id != -1:
            cv2.imshow('Tracking Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if FLAGS.save_mot_txts:
        result_filename = os.path.join(FLAGS.output_dir,
                                       video_name.split('.')[-2] + '.txt')
        write_mot_results(result_filename, results)

    if FLAGS.save_images:
        save_dir = os.path.join(FLAGS.output_dir, video_name.split('.')[-2])
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg {}'.format(
            save_dir, out_path)
        os.system(cmd_str)
        print('Save video in {}.'.format(out_path))
    else:
        writer.release()


def main():
    pred_config = PredictConfig(FLAGS.model_dir)
    detector = JDE_Detector(
        pred_config,
        FLAGS.model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn)

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
            det_log('MOT')


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()
