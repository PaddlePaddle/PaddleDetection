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
import cv2
import math
import numpy as np
import paddle

from mot_keypoint_unite_utils import argsparser
from keypoint_infer import KeyPoint_Detector, PredictConfig_KeyPoint
from keypoint_det_unite_infer import bench_log
from keypoint_visualize import draw_pose
from benchmark_utils import PaddleInferBenchmark
from utils import Timer

from tracker import JDETracker
from preprocess import LetterBoxResize
from mot_infer import MOT_Detector, write_mot_results
from infer import Detector, PredictConfig, print_arguments, get_test_images
from ppdet.modeling.mot import visualization as mot_vis
from ppdet.modeling.mot.utils import Timer as FPSTimer
from utils import get_current_memory_mb


def mot_keypoint_unite_predict_image(mot_model, keypoint_model, image_list):
    for i, img_file in enumerate(image_list):
        frame = cv2.imread(img_file)

        if FLAGS.run_benchmark:
            mot_model.predict(frame, FLAGS.mot_threshold, warmup=10, repeats=10)
            cm, gm, gu = get_current_memory_mb()
            mot_model.cpu_mem += cm
            mot_model.gpu_mem += gm
            mot_model.gpu_util += gu

            keypoint_model.predict(
                [frame], FLAGS.keypoint_threshold, warmup=10, repeats=10)
            cm, gm, gu = get_current_memory_mb()
            keypoint_model.cpu_mem += cm
            keypoint_model.gpu_mem += gm
            keypoint_model.gpu_util += gu
        else:
            online_tlwhs, online_scores, online_ids = mot_model.predict(
                frame, FLAGS.mot_threshold)
            keypoint_results = keypoint_model.predict([frame],
                                                      FLAGS.keypoint_threshold)

            im = draw_pose(
                frame,
                keypoint_results,
                visual_thread=FLAGS.keypoint_threshold,
                returnimg=True)

            online_im = mot_vis.plot_tracking(
                im, online_tlwhs, online_ids, online_scores, frame_id=i)
            if FLAGS.save_images:
                if not os.path.exists(FLAGS.output_dir):
                    os.makedirs(FLAGS.output_dir)
                cv2.imwrite(os.path.join(FLAGS.output_dir, img_file), online_im)


def mot_keypoint_unite_predict_video(mot_model, keypoint_model, camera_id):
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
        video_name = 'output.mp4'
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
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    frame_id = 0
    timer_mot = FPSTimer()
    timer_kp = FPSTimer()
    timer_mot_kp = FPSTimer()
    mot_results = []
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        timer_mot_kp.tic()
        timer_mot.tic()
        online_tlwhs, online_scores, online_ids = mot_model.predict(
            frame, FLAGS.mot_threshold)
        timer_mot.toc()

        mot_results.append(
            (frame_id + 1, online_tlwhs, online_scores, online_ids))
        mot_fps = 1. / timer_mot.average_time

        timer_kp.tic()
        keypoint_results = keypoint_model.predict([frame],
                                                  FLAGS.keypoint_threshold)
        timer_kp.toc()
        timer_mot_kp.toc()
        kp_fps = 1. / timer_kp.average_time
        mot_kp_fps = 1. / timer_mot_kp.average_time

        im = draw_pose(
            frame,
            keypoint_results,
            visual_thread=FLAGS.keypoint_threshold,
            returnimg=True)

        online_im = mot_vis.plot_tracking(
            im,
            online_tlwhs,
            online_ids,
            online_scores,
            frame_id=frame_id,
            fps=mot_kp_fps)

        im = np.array(online_im)

        frame_id += 1
        print('detect frame:%d' % (frame_id))

        if FLAGS.save_images:
            save_dir = os.path.join(FLAGS.output_dir, video_name.split('.')[-2])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(
                os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), im)

        writer.write(im)
        if camera_id != -1:
            cv2.imshow('Tracking and keypoint results', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if FLAGS.save_mot_txts:
        result_filename = os.path.join(FLAGS.output_dir,
                                       video_name.split('.')[-2] + '.txt')
        write_mot_results(result_filename, mot_results)
    writer.release()


def main():
    pred_config = PredictConfig(FLAGS.mot_model_dir)
    mot_model = MOT_Detector(
        pred_config,
        FLAGS.mot_model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn)

    pred_config = PredictConfig_KeyPoint(FLAGS.keypoint_model_dir)
    keypoint_model = KeyPoint_Detector(
        pred_config,
        FLAGS.keypoint_model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn,
        use_dark=FLAGS.use_dark)

    # predict from video file or camera video stream
    if FLAGS.video_file is not None or FLAGS.camera_id != -1:
        mot_keypoint_unite_predict_video(mot_model, keypoint_model,
                                         FLAGS.camera_id)
    else:
        # predict from image
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        mot_keypoint_unite_predict_image(mot_model, keypoint_model, img_list)

        if not FLAGS.run_benchmark:
            mot_model.det_times.info(average=True)
            keypoint_model.det_times.info(average=True)
        else:
            mode = FLAGS.run_mode
            mot_model_dir = FLAGS.mot_model_dir
            mot_model_info = {
                'model_name': mot_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(mot_model, img_list, mot_model_info, name='MOT')

            keypoint_model_dir = FLAGS.keypoint_model_dir
            keypoint_model_info = {
                'model_name': keypoint_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(keypoint_model, img_list, keypoint_model_info, 'KeyPoint')


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()
