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
import copy
import numpy as np
from collections import defaultdict
import paddle

from utils import get_current_memory_mb
from infer import Detector, PredictConfig, print_arguments, get_test_images
from visualize import draw_pose

from mot_keypoint_unite_utils import argsparser
from keypoint_infer import KeyPoint_Detector, PredictConfig_KeyPoint
from det_keypoint_unite_infer import predict_with_given_det, bench_log
from mot_jde_infer import JDE_Detector

from ppdet.modeling.mot.visualization import plot_tracking_dict
from ppdet.modeling.mot.utils import MOTTimer as FPSTimer
from ppdet.modeling.mot.utils import write_mot_results

# Global dictionary
KEYPOINT_SUPPORT_MODELS = {
    'HigherHRNet': 'keypoint_bottomup',
    'HRNet': 'keypoint_topdown'
}


def convert_mot_to_det(tlwhs, scores):
    results = {}
    num_mot = len(tlwhs)
    xyxys = copy.deepcopy(tlwhs)
    for xyxy in xyxys.copy():
        xyxy[2:] = xyxy[2:] + xyxy[:2]
    # support single class now
    results['boxes'] = np.vstack(
        [np.hstack([0, scores[i], xyxys[i]]) for i in range(num_mot)])
    return results


def mot_keypoint_unite_predict_image(mot_model,
                                     keypoint_model,
                                     image_list,
                                     keypoint_batch_size=1):
    num_classes = mot_model.num_classes
    assert num_classes == 1, 'Only one category mot model supported for uniting keypoint deploy.'
    data_type = 'mot'
    image_list.sort()
    for i, img_file in enumerate(image_list):
        frame = cv2.imread(img_file)

        if FLAGS.run_benchmark:
            online_tlwhs, online_scores, online_ids = mot_model.predict(
                [frame], FLAGS.mot_threshold, warmup=10, repeats=10)
            cm, gm, gu = get_current_memory_mb()
            mot_model.cpu_mem += cm
            mot_model.gpu_mem += gm
            mot_model.gpu_util += gu

        else:
            online_tlwhs, online_scores, online_ids = mot_model.predict(
                [frame], FLAGS.mot_threshold)

        keypoint_arch = keypoint_model.pred_config.arch
        if KEYPOINT_SUPPORT_MODELS[keypoint_arch] == 'keypoint_topdown':
            results = convert_mot_to_det(online_tlwhs, online_scores)
            keypoint_results = predict_with_given_det(
                frame, results, keypoint_model, keypoint_batch_size,
                FLAGS.mot_threshold, FLAGS.keypoint_threshold,
                FLAGS.run_benchmark)

        else:
            warmup = 10 if FLAGS.run_benchmark else 0
            repeats = 10 if FLAGS.run_benchmark else 1
            keypoint_results = keypoint_model.predict(
                [frame],
                FLAGS.keypoint_threshold,
                warmup=warmup,
                repeats=repeats)

        if FLAGS.run_benchmark:
            cm, gm, gu = get_current_memory_mb()
            keypoint_model.cpu_mem += cm
            keypoint_model.gpu_mem += gm
            keypoint_model.gpu_util += gu
        else:
            im = draw_pose(
                frame,
                keypoint_results,
                visual_thread=FLAGS.keypoint_threshold,
                returnimg=True,
                ids=online_ids
                if KEYPOINT_SUPPORT_MODELS[keypoint_arch] == 'keypoint_topdown'
                else None)

            online_im = plot_tracking_dict(
                im,
                num_classes,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=i)
            if FLAGS.save_images:
                if not os.path.exists(FLAGS.output_dir):
                    os.makedirs(FLAGS.output_dir)
                img_name = os.path.split(img_file)[-1]
                out_path = os.path.join(FLAGS.output_dir, img_name)
                cv2.imwrite(out_path, online_im)
                print("save result to: " + out_path)


def mot_keypoint_unite_predict_video(mot_model,
                                     keypoint_model,
                                     camera_id,
                                     keypoint_batch_size=1):
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
        video_name = 'output.mp4'
    else:
        capture = cv2.VideoCapture(FLAGS.video_file)
        video_name = os.path.split(FLAGS.video_file)[-1]
    # Get Video info : resolution, fps, frame count
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps: %d, frame_count: %d" % (fps, frame_count))

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    out_path = os.path.join(FLAGS.output_dir, video_name)
    if not FLAGS.save_images:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    frame_id = 0
    timer_mot = FPSTimer()
    timer_kp = FPSTimer()
    timer_mot_kp = FPSTimer()

    # support single class and multi classes, but should be single class here
    mot_results = defaultdict(list)
    num_classes = mot_model.num_classes
    assert num_classes == 1, 'Only one category mot model supported for uniting keypoint deploy.'
    data_type = 'mot'

    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        timer_mot_kp.tic()
        timer_mot.tic()
        online_tlwhs, online_scores, online_ids = mot_model.predict(
            [frame], FLAGS.mot_threshold)
        timer_mot.toc()
        mot_results[0].append(
            (frame_id + 1, online_tlwhs[0], online_scores[0], online_ids[0]))
        mot_fps = 1. / timer_mot.average_time

        timer_kp.tic()

        keypoint_arch = keypoint_model.pred_config.arch
        if KEYPOINT_SUPPORT_MODELS[keypoint_arch] == 'keypoint_topdown':
            results = convert_mot_to_det(online_tlwhs[0], online_scores[0])
            keypoint_results = predict_with_given_det(
                frame, results, keypoint_model, keypoint_batch_size,
                FLAGS.mot_threshold, FLAGS.keypoint_threshold,
                FLAGS.run_benchmark)

        else:
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
            returnimg=True,
            ids=online_ids
            if KEYPOINT_SUPPORT_MODELS[keypoint_arch] == 'keypoint_topdown' else
            None)

        online_im = plot_tracking_dict(
            im,
            num_classes,
            online_tlwhs,
            online_ids,
            online_scores,
            frame_id=frame_id,
            fps=mot_kp_fps)

        im = np.array(online_im)

        frame_id += 1
        print('detect frame: %d' % (frame_id))

        if FLAGS.save_images:
            save_dir = os.path.join(FLAGS.output_dir, video_name.split('.')[-2])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(
                os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), im)
        else:
            writer.write(im)
        if camera_id != -1:
            cv2.imshow('Tracking and keypoint results', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if FLAGS.save_mot_txts:
        result_filename = os.path.join(FLAGS.output_dir,
                                       video_name.split('.')[-2] + '.txt')
        write_mot_results(result_filename, mot_results, data_type, num_classes)

    if FLAGS.save_images:
        save_dir = os.path.join(FLAGS.output_dir, video_name.split('.')[-2])
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg {}'.format(save_dir,
                                                              out_path)
        os.system(cmd_str)
        print('Save video in {}.'.format(out_path))
    else:
        writer.release()


def main():
    pred_config = PredictConfig(FLAGS.mot_model_dir)
    mot_model = JDE_Detector(
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
        batch_size=FLAGS.keypoint_batch_size,
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
                                         FLAGS.camera_id,
                                         FLAGS.keypoint_batch_size)
    else:
        # predict from image
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        mot_keypoint_unite_predict_image(mot_model, keypoint_model, img_list,
                                         FLAGS.keypoint_batch_size)

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
