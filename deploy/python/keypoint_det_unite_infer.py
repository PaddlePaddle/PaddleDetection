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
from PIL import Image
import cv2
import math
import numpy as np
import paddle

from topdown_unite_utils import argsparser
from preprocess import decode_image
from infer import Detector, PredictConfig, print_arguments, get_test_images
from keypoint_infer import KeyPoint_Detector, PredictConfig_KeyPoint
from keypoint_visualize import draw_pose
from benchmark_utils import PaddleInferBenchmark
from utils import get_current_memory_mb


def bench_log(detector, img_list, model_info, batch_size=1, name=None):
    mems = {
        'cpu_rss_mb': detector.cpu_mem / len(img_list),
        'gpu_rss_mb': detector.gpu_mem / len(img_list),
        'gpu_util': detector.gpu_util * 100 / len(img_list)
    }
    perf_info = detector.det_times.report(average=True)
    data_info = {
        'batch_size': batch_size,
        'shape': "dynamic_shape",
        'data_num': perf_info['img_num']
    }

    log = PaddleInferBenchmark(detector.config, model_info, data_info,
                               perf_info, mems)
    log(name)


def affine_backto_orgimages(keypoint_result, batch_records):
    kpts, scores = keypoint_result['keypoint']
    kpts[..., 0] += batch_records[:, 0:1]
    kpts[..., 1] += batch_records[:, 1:2]
    return kpts, scores


def topdown_unite_predict(detector,
                          topdown_keypoint_detector,
                          image_list,
                          keypoint_batch_size=1):
    det_timer = detector.get_timer()
    for i, img_file in enumerate(image_list):
        # Decode image in advance in det + pose prediction
        det_timer.preprocess_time_s.start()
        image, _ = decode_image(img_file, {})
        det_timer.preprocess_time_s.end()

        if FLAGS.run_benchmark:
            results = detector.predict(
                [image], FLAGS.det_threshold, warmup=10, repeats=10)
            cm, gm, gu = get_current_memory_mb()
            detector.cpu_mem += cm
            detector.gpu_mem += gm
            detector.gpu_util += gu
        else:
            results = detector.predict([image], FLAGS.det_threshold)

        if results['boxes_num'] == 0:
            continue
        rec_images, records, det_rects = topdown_keypoint_detector.get_person_from_rect(
            image, results, FLAGS.det_threshold)
        keypoint_vector = []
        score_vector = []
        rect_vector = det_rects
        batch_loop_cnt = math.ceil(float(len(rec_images)) / keypoint_batch_size)

        for i in range(batch_loop_cnt):
            start_index = i * keypoint_batch_size
            end_index = min((i + 1) * keypoint_batch_size, len(rec_images))
            batch_images = rec_images[start_index:end_index]
            batch_records = np.array(records[start_index:end_index])
            if FLAGS.run_benchmark:
                keypoint_result = topdown_keypoint_detector.predict(
                    batch_images,
                    FLAGS.keypoint_threshold,
                    warmup=10,
                    repeats=10)
            else:
                keypoint_result = topdown_keypoint_detector.predict(
                    batch_images, FLAGS.keypoint_threshold)
            orgkeypoints, scores = affine_backto_orgimages(keypoint_result,
                                                           batch_records)
            keypoint_vector.append(orgkeypoints)
            score_vector.append(scores)
        if FLAGS.run_benchmark:
            cm, gm, gu = get_current_memory_mb()
            topdown_keypoint_detector.cpu_mem += cm
            topdown_keypoint_detector.gpu_mem += gm
            topdown_keypoint_detector.gpu_util += gu
        else:
            keypoint_res = {}
            keypoint_res['keypoint'] = [
                np.vstack(keypoint_vector), np.vstack(score_vector)
            ]
            keypoint_res['bbox'] = rect_vector
            if not os.path.exists(FLAGS.output_dir):
                os.makedirs(FLAGS.output_dir)
            draw_pose(
                img_file,
                keypoint_res,
                visual_thread=FLAGS.keypoint_threshold,
                save_dir=FLAGS.output_dir)


def topdown_unite_predict_video(detector,
                                topdown_keypoint_detector,
                                camera_id,
                                keypoint_batch_size=1):
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
        video_name = 'output.mp4'
    else:
        capture = cv2.VideoCapture(FLAGS.video_file)
        video_name = os.path.splitext(os.path.basename(FLAGS.video_file))[
            0] + '.mp4'
    fps = 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # yapf: disable
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # yapf: enable
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    out_path = os.path.join(FLAGS.output_dir, video_name)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    index = 0
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        index += 1
        print('detect frame:%d' % (index))

        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.predict([frame2], FLAGS.det_threshold)
        rec_images, records, rect_vector = topdown_keypoint_detector.get_person_from_rect(
            frame2, results)
        keypoint_vector = []
        score_vector = []
        batch_loop_cnt = math.ceil(float(len(rec_images)) / keypoint_batch_size)
        for i in range(batch_loop_cnt):
            start_index = i * keypoint_batch_size
            end_index = min((i + 1) * keypoint_batch_size, len(rec_images))
            batch_images = rec_images[start_index:end_index]
            batch_records = np.array(records[start_index:end_index])
            keypoint_result = topdown_keypoint_detector.predict(
                batch_images, FLAGS.keypoint_threshold)
            orgkeypoints, scores = affine_backto_orgimages(keypoint_result,
                                                           batch_records)
            keypoint_vector.append(orgkeypoints)
            score_vector.append(scores)
        keypoint_res = {}
        keypoint_res['keypoint'] = [
            np.vstack(keypoint_vector), np.vstack(score_vector)
        ] if len(keypoint_vector) > 0 else [[], []]
        keypoint_res['bbox'] = rect_vector
        im = draw_pose(
            frame,
            keypoint_res,
            visual_thread=FLAGS.keypoint_threshold,
            returnimg=True)

        writer.write(im)
        if camera_id != -1:
            cv2.imshow('Mask Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    writer.release()


def main():
    pred_config = PredictConfig(FLAGS.det_model_dir)
    detector = Detector(
        pred_config,
        FLAGS.det_model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn)

    pred_config = PredictConfig_KeyPoint(FLAGS.keypoint_model_dir)
    topdown_keypoint_detector = KeyPoint_Detector(
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
        topdown_unite_predict_video(detector, topdown_keypoint_detector,
                                    FLAGS.camera_id, FLAGS.keypoint_batch_size)
    else:
        # predict from image
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        topdown_unite_predict(detector, topdown_keypoint_detector, img_list,
                              FLAGS.keypoint_batch_size)
        if not FLAGS.run_benchmark:
            detector.det_times.info(average=True)
            topdown_keypoint_detector.det_times.info(average=True)
        else:
            mode = FLAGS.run_mode
            det_model_dir = FLAGS.det_model_dir
            det_model_info = {
                'model_name': det_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(detector, img_list, det_model_info, name='Det')
            keypoint_model_dir = FLAGS.keypoint_model_dir
            keypoint_model_info = {
                'model_name': keypoint_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(topdown_keypoint_detector, img_list, keypoint_model_info,
                      FLAGS.keypoint_batch_size, 'KeyPoint')


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()
