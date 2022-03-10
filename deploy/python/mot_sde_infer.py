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
import time
import yaml
import cv2
import numpy as np
from collections import defaultdict
import paddle

from benchmark_utils import PaddleInferBenchmark
from preprocess import decode_image
from utils import argsparser, Timer, get_current_memory_mb
from infer import Detector, get_test_images, print_arguments, bench_log, PredictConfig

# add python path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from pptracking.python.mot import JDETracker
from pptracking.python.mot.utils import MOTTimer, write_mot_results
from pptracking.python.visualize import plot_tracking, plot_tracking_dict

# Global dictionary
MOT_SDE_SUPPORT_MODELS = {
    'DeepSORT',
    'ByteTrack',
    'YOLO',
}


class SDE_Detector(Detector):
    """
    Args:
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        tracker_config (str): tracker config path
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
        use_dark(bool): whether to use postprocess in DarkPose
    """

    def __init__(self,
                 model_dir,
                 tracker_config,
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
                 threshold=0.5):
        super(SDE_Detector, self).__init__(
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
            threshold=threshold, )
        assert batch_size == 1, "MOT model only supports batch_size=1."
        self.det_times = Timer(with_tracker=True)
        self.num_classes = len(self.pred_config.labels)

        # tracker config
        self.tracker_config = tracker_config
        cfg = yaml.safe_load(open(self.tracker_config))['tracker']
        min_box_area = cfg.get('min_box_area', 200)
        vertical_ratio = cfg.get('vertical_ratio', 1.6)
        use_byte = cfg.get('use_byte', True)
        match_thres = cfg.get('match_thres', 0.9)
        conf_thres = cfg.get('conf_thres', 0.6)
        low_conf_thres = cfg.get('low_conf_thres', 0.1)

        self.tracker = JDETracker(
            use_byte=use_byte,
            num_classes=self.num_classes,
            min_box_area=min_box_area,
            vertical_ratio=vertical_ratio,
            match_thres=match_thres,
            conf_thres=conf_thres,
            low_conf_thres=low_conf_thres)

    def tracking(self, det_results):
        pred_dets = det_results['boxes']  # 'cls_id, score, x0, y0, x1, y1'
        pred_embs = None

        online_targets_dict = self.tracker.update(pred_dets, pred_embs)
        online_tlwhs = defaultdict(list)
        online_scores = defaultdict(list)
        online_ids = defaultdict(list)
        for cls_id in range(self.num_classes):
            online_targets = online_targets_dict[cls_id]
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                tscore = t.score
                if tlwh[2] * tlwh[3] <= self.tracker.min_box_area:
                    continue
                if self.tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                        3] > self.tracker.vertical_ratio:
                    continue
                online_tlwhs[cls_id].append(tlwh)
                online_ids[cls_id].append(tid)
                online_scores[cls_id].append(tscore)

        return online_tlwhs, online_scores, online_ids

    def predict_image(self,
                      image_list,
                      run_benchmark=False,
                      repeats=1,
                      visual=True):
        mot_results = []
        num_classes = self.num_classes
        image_list.sort()
        ids2names = self.pred_config.labels
        for frame_id, img_file in enumerate(image_list):
            batch_image_list = [img_file]  # bs=1 in MOT model
            if run_benchmark:
                # preprocess
                inputs = self.preprocess(batch_image_list)  # warmup
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                result_warmup = self.predict(repeats=repeats)  # warmup
                self.det_times.inference_time_s.start()
                result = self.predict(repeats=repeats)
                self.det_times.inference_time_s.end(repeats=repeats)

                # postprocess
                result_warmup = self.postprocess(inputs, result)  # warmup
                self.det_times.postprocess_time_s.start()
                det_result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()

                # tracking
                result_warmup = self.tracking(det_result)
                self.det_times.tracking_time_s.start()
                online_tlwhs, online_scores, online_ids = self.tracking(
                    det_result)
                self.det_times.tracking_time_s.end()
                self.det_times.img_num += 1

                cm, gm, gu = get_current_memory_mb()
                self.cpu_mem += cm
                self.gpu_mem += gm
                self.gpu_util += gu

            else:
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                self.det_times.inference_time_s.start()
                result = self.predict()
                self.det_times.inference_time_s.end()

                self.det_times.postprocess_time_s.start()
                det_result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()

                # tracking process
                self.det_times.tracking_time_s.start()
                online_tlwhs, online_scores, online_ids = self.tracking(
                    det_result)
                self.det_times.tracking_time_s.end()
                self.det_times.img_num += 1

            if visual:
                if frame_id % 10 == 0:
                    print('Tracking frame {}'.format(frame_id))
                frame, _ = decode_image(img_file, {})

                im = plot_tracking_dict(
                    frame,
                    num_classes,
                    online_tlwhs,
                    online_ids,
                    online_scores,
                    frame_id=frame_id,
                    ids2names=[])
                seq_name = image_list[0].split('/')[-2]
                save_dir = os.path.join(self.output_dir, seq_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(
                    os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), im)

            mot_results.append([online_tlwhs, online_scores, online_ids])
        return mot_results

    def predict_video(self, video_file, camera_id):
        video_out_name = 'output.mp4'
        if camera_id != -1:
            capture = cv2.VideoCapture(camera_id)
        else:
            capture = cv2.VideoCapture(video_file)
            video_out_name = os.path.split(video_file)[-1]
        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("fps: %d, frame_count: %d" % (fps, frame_count))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        out_path = os.path.join(self.output_dir, video_out_name)
        fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_id = 1
        timer = MOTTimer()
        results = defaultdict(list)  # support single class and multi classes
        num_classes = self.num_classes
        while (1):
            ret, frame = capture.read()
            if not ret:
                break
            if frame_id % 10 == 0:
                print('Tracking frame: %d' % (frame_id))
            frame_id += 1

            timer.tic()
            mot_results = self.predict_image([frame], visual=False)
            timer.toc()

            online_tlwhs, online_scores, online_ids = mot_results[0]
            for cls_id in range(num_classes):
                results[cls_id].append(
                    (frame_id + 1, online_tlwhs[cls_id], online_scores[cls_id],
                     online_ids[cls_id]))

            fps = 1. / timer.duration
            im = plot_tracking_dict(
                frame,
                num_classes,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=frame_id,
                fps=fps,
                ids2names=[])

            writer.write(im)
            if camera_id != -1:
                cv2.imshow('Mask Detection', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        writer.release()


def main():
    deploy_file = os.path.join(FLAGS.model_dir, 'infer_cfg.yml')
    with open(deploy_file) as f:
        yml_conf = yaml.safe_load(f)
    arch = yml_conf['arch']
    assert arch in MOT_SDE_SUPPORT_MODELS, '{} is not supported.'.format(arch)
    detector = SDE_Detector(
        FLAGS.model_dir,
        FLAGS.tracker_config,
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
        output_dir=FLAGS.output_dir)

    # predict from video file or camera video stream
    if FLAGS.video_file is not None or FLAGS.camera_id != -1:
        detector.predict_video(FLAGS.video_file, FLAGS.camera_id)
    else:
        # predict from image
        if FLAGS.image_dir is None and FLAGS.image_file is not None:
            assert FLAGS.batch_size == 1, "--batch_size should be 1 in MOT models."
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        detector.predict_image(img_list, FLAGS.run_benchmark, repeats=10)

        if not FLAGS.run_benchmark:
            detector.det_times.info(average=True)
        else:
            mode = FLAGS.run_mode
            model_dir = FLAGS.model_dir
            model_info = {
                'model_name': model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(detector, img_list, model_info, name='MOT')


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()
