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
from collections import defaultdict

import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

from utils import argsparser, Timer, get_current_memory_mb
from det_infer import Detector, get_test_images, print_arguments, PredictConfig
from benchmark_utils import PaddleInferBenchmark
from visualize import plot_tracking_dict

from mot.tracker import JDETracker
from mot.utils import MOTTimer, write_mot_results, flow_statistic

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
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of per batch in inference, default is 1 in tracking models
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
                 run_mode='paddle',
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
        self.num_classes = len(pred_config.labels)

        tp = pred_config.tracker
        min_box_area = tp['min_box_area'] if 'min_box_area' in tp else 200
        vertical_ratio = tp['vertical_ratio'] if 'vertical_ratio' in tp else 1.6
        conf_thres = tp['conf_thres'] if 'conf_thres' in tp else 0.
        tracked_thresh = tp['tracked_thresh'] if 'tracked_thresh' in tp else 0.7
        metric_type = tp['metric_type'] if 'metric_type' in tp else 'euclidean'

        self.tracker = JDETracker(
            num_classes=self.num_classes,
            min_box_area=min_box_area,
            vertical_ratio=vertical_ratio,
            conf_thres=conf_thres,
            tracked_thresh=tracked_thresh,
            metric_type=metric_type)

    def postprocess(self, pred_dets, pred_embs, threshold):
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
                if tscore < threshold: continue
                if tlwh[2] * tlwh[3] <= self.tracker.min_box_area:
                    continue
                if self.tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                        3] > self.tracker.vertical_ratio:
                    continue
                online_tlwhs[cls_id].append(tlwh)
                online_ids[cls_id].append(tid)
                online_scores[cls_id].append(tscore)
        return online_tlwhs, online_scores, online_ids

    def predict(self, image_list, threshold=0.5, repeats=1, add_timer=True):
        '''
        Args:
            image_list (list[str]): path of images, only support one image path
                (batch_size=1) in tracking model
            threshold (float): threshold of predicted box' score
            repeats (int): repeat number for prediction
            add_timer (bool): whether add timer during prediction
        Returns:
            online_tlwhs, online_scores, online_ids (dict[np.array])
        '''
        # preprocess
        if add_timer:
            self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(image_list)

        pred_dets, pred_embs = None, None
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])
        if add_timer:
            self.det_times.preprocess_time_s.end()
            self.det_times.inference_time_s.start()

        # model prediction
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            pred_dets = boxes_tensor.copy_to_cpu()
            embs_tensor = self.predictor.get_output_handle(output_names[1])
            pred_embs = embs_tensor.copy_to_cpu()
        if add_timer:
            self.det_times.inference_time_s.end(repeats=repeats)
            self.det_times.postprocess_time_s.start()

        # postprocess
        online_tlwhs, online_scores, online_ids = self.postprocess(
            pred_dets, pred_embs, threshold)
        if add_timer:
            self.det_times.postprocess_time_s.end()
            self.det_times.img_num += 1
        return online_tlwhs, online_scores, online_ids


def predict_image(detector,
                  image_list,
                  threshold,
                  output_dir,
                  save_images=True,
                  run_benchmark=False):
    results = []
    num_classes = detector.num_classes
    data_type = 'mcmot' if num_classes > 1 else 'mot'
    ids2names = detector.pred_config.labels

    image_list.sort()
    for frame_id, img_file in enumerate(image_list):
        frame = cv2.imread(img_file)
        if run_benchmark:
            # warmup
            detector.predict([img_file], threshold, repeats=10, add_timer=False)
            # run benchmark
            detector.predict([img_file], threshold, repeats=10, add_timer=True)
            cm, gm, gu = get_current_memory_mb()
            detector.cpu_mem += cm
            detector.gpu_mem += gm
            detector.gpu_util += gu
            print('Test iter {}, file name:{}'.format(frame_id, img_file))
        else:
            online_tlwhs, online_scores, online_ids = detector.predict(
                [img_file], threshold)
            online_im = plot_tracking_dict(
                frame,
                num_classes,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=frame_id,
                ids2names=ids2names)
            if save_images:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                img_name = os.path.split(img_file)[-1]
                out_path = os.path.join(output_dir, img_name)
                cv2.imwrite(out_path, online_im)
                print("save result to: " + out_path)


def predict_video(detector,
                  video_file,
                  threshold,
                  output_dir,
                  save_images=True,
                  save_mot_txts=True,
                  draw_center_traj=False,
                  secs_interval=10,
                  do_entrance_counting=False,
                  camera_id=-1):
    video_name = 'mot_output.mp4'
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
    else:
        capture = cv2.VideoCapture(video_file)
        video_name = os.path.split(video_file)[-1]

    # Get Video info : resolution, fps, frame count
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps: %d, frame_count: %d" % (fps, frame_count))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_path = os.path.join(output_dir, video_name)
    if not save_images:
        video_format = 'mp4v'
        fourcc = cv2.VideoWriter_fourcc(*video_format)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    frame_id = 0
    timer = MOTTimer()
    results = defaultdict(list)  # support single class and multi classes
    num_classes = detector.num_classes
    data_type = 'mcmot' if num_classes > 1 else 'mot'
    ids2names = detector.pred_config.labels
    center_traj = None
    entrance = None
    records = None
    if draw_center_traj:
        center_traj = [{} for i in range(num_classes)]

    if num_classes == 1:
        id_set = set()
        interval_id_set = set()
        in_id_list = list()
        out_id_list = list()
        prev_center = dict()
        records = list()
        entrance = [0, height / 2., width, height / 2.]

    video_fps = fps

    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        timer.tic()
        online_tlwhs, online_scores, online_ids = detector.predict([frame],
                                                                   threshold)
        timer.toc()

        for cls_id in range(num_classes):
            results[cls_id].append((frame_id + 1, online_tlwhs[cls_id],
                                    online_scores[cls_id], online_ids[cls_id]))

        fps = 1. / timer.duration
        # NOTE: just implement flow statistic for one class
        if num_classes == 1:
            result = (frame_id + 1, online_tlwhs[0], online_scores[0],
                      online_ids[0])
            statistic = flow_statistic(
                result, secs_interval, do_entrance_counting, video_fps,
                entrance, id_set, interval_id_set, in_id_list, out_id_list,
                prev_center, records, data_type, num_classes)
            id_set = statistic['id_set']
            interval_id_set = statistic['interval_id_set']
            in_id_list = statistic['in_id_list']
            out_id_list = statistic['out_id_list']
            prev_center = statistic['prev_center']
            records = statistic['records']

        elif num_classes > 1 and do_entrance_counting:
            raise NotImplementedError(
                'Multi-class flow counting is not implemented now!')
        im = plot_tracking_dict(
            frame,
            num_classes,
            online_tlwhs,
            online_ids,
            online_scores,
            frame_id=frame_id,
            fps=fps,
            ids2names=ids2names,
            do_entrance_counting=do_entrance_counting,
            entrance=entrance,
            records=records,
            center_traj=center_traj)

        if save_images:
            save_dir = os.path.join(output_dir, video_name.split('.')[-2])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(
                os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), im)
        else:
            writer.write(im)

        frame_id += 1
        print('detect frame: %d, fps: %f' % (frame_id, fps))
        if camera_id != -1:
            cv2.imshow('Tracking Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if save_mot_txts:
        result_filename = os.path.join(output_dir,
                                       video_name.split('.')[-2] + '.txt')

        write_mot_results(result_filename, results, data_type, num_classes)

        if num_classes == 1:
            result_filename = os.path.join(
                output_dir, video_name.split('.')[-2] + '_flow_statistic.txt')
            f = open(result_filename, 'w')
            for line in records:
                f.write(line)
            print('Flow statistic save in {}'.format(result_filename))
            f.close()

    if save_images:
        save_dir = os.path.join(output_dir, video_name.split('.')[-2])
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg {}'.format(save_dir,
                                                              out_path)
        os.system(cmd_str)
        print('Save video in {}.'.format(out_path))
    else:
        writer.release()


def predict_naive(model_dir,
                  video_file,
                  image_dir,
                  device='gpu',
                  threshold=0.5,
                  output_dir='output'):
    pred_config = PredictConfig(model_dir)
    detector = JDE_Detector(pred_config, model_dir, device=device.upper())

    if video_file is not None:
        predict_video(
            detector,
            video_file,
            threshold=threshold,
            output_dir=output_dir,
            save_images=True,
            save_mot_txts=True,
            draw_center_traj=False,
            secs_interval=10,
            do_entrance_counting=False)
    else:
        img_list = get_test_images(image_dir, infer_img=None)
        predict_image(
            detector,
            img_list,
            threshold=threshold,
            output_dir=output_dir,
            save_images=True)


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
        predict_video(
            detector,
            FLAGS.video_file,
            threshold=FLAGS.threshold,
            output_dir=FLAGS.output_dir,
            save_images=FLAGS.save_images,
            save_mot_txts=FLAGS.save_mot_txts,
            draw_center_traj=FLAGS.draw_center_traj,
            secs_interval=FLAGS.secs_interval,
            do_entrance_counting=FLAGS.do_entrance_counting,
            camera_id=FLAGS.camera_id)
    else:
        # predict from image
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        predict_image(
            detector,
            img_list,
            threshold=FLAGS.threshold,
            output_dir=FLAGS.output_dir,
            save_images=FLAGS.save_images,
            run_benchmark=FLAGS.run_benchmark)
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
