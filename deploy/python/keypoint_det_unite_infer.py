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
import numpy as np
import paddle

from topdown_unite_utils import argsparser
from preprocess import decode_image
from infer import Detector, PredictConfig, print_arguments, get_test_images
from keypoint_infer import KeyPoint_Detector, PredictConfig_KeyPoint
from keypoint_visualize import draw_pose


def expand_crop(images, rect, expand_ratio=0.3):
    imgh, imgw, c = images.shape
    label, conf, xmin, ymin, xmax, ymax = [int(x) for x in rect.tolist()]
    if label != 0:
        return None, None, None
    org_rect = [xmin, ymin, xmax, ymax]
    h_half = (ymax - ymin) * (1 + expand_ratio) / 2.
    w_half = (xmax - xmin) * (1 + expand_ratio) / 2.
    if h_half > w_half * 4 / 3:
        w_half = h_half * 0.75
    center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]
    ymin = max(0, int(center[0] - h_half))
    ymax = min(imgh - 1, int(center[0] + h_half))
    xmin = max(0, int(center[1] - w_half))
    xmax = min(imgw - 1, int(center[1] + w_half))
    return images[ymin:ymax, xmin:xmax, :], [xmin, ymin, xmax, ymax], org_rect


def get_person_from_rect(images, results):
    det_results = results['boxes']
    mask = det_results[:, 1] > FLAGS.det_threshold
    valid_rects = det_results[mask]
    image_buff = []
    org_rects = []
    for rect in valid_rects:
        rect_image, new_rect, org_rect = expand_crop(images, rect)
        if rect_image is None or rect_image.size == 0:
            continue
        image_buff.append([rect_image, new_rect])
        org_rects.append(org_rect)
    return image_buff, org_rects


def affine_backto_orgimages(keypoint_result, batch_records):
    kpts, scores = keypoint_result['keypoint']
    kpts[..., 0] += batch_records[0]
    kpts[..., 1] += batch_records[1]
    return kpts, scores


def topdown_unite_predict(detector, topdown_keypoint_detector, image_list):
    for i, img_file in enumerate(image_list):
        image, _ = decode_image(img_file, {})
        results = detector.predict(image, FLAGS.det_threshold)
        batchs_images, det_rects = get_person_from_rect(image, results)
        keypoint_vector = []
        score_vector = []
        rect_vecotr = det_rects
        for batch_images, batch_records in batchs_images:
            keypoint_result = topdown_keypoint_detector.predict(
                batch_images, FLAGS.keypoint_threshold)
            orgkeypoints, scores = affine_backto_orgimages(keypoint_result,
                                                           batch_records)
            keypoint_vector.append(orgkeypoints)
            score_vector.append(scores)
        keypoint_res = {}
        keypoint_res['keypoint'] = [
            np.vstack(keypoint_vector), np.vstack(score_vector)
        ]
        keypoint_res['bbox'] = rect_vecotr
        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        draw_pose(
            img_file,
            keypoint_res,
            visual_thread=FLAGS.keypoint_threshold,
            save_dir=FLAGS.output_dir)


def topdown_unite_predict_video(detector, topdown_keypoint_detector, camera_id):
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
        results = detector.predict(frame2, FLAGS.det_threshold)
        batchs_images, rect_vecotr = get_person_from_rect(frame2, results)
        keypoint_vector = []
        score_vector = []
        for batch_images, batch_records in batchs_images:
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
        keypoint_res['bbox'] = rect_vecotr
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
        use_gpu=FLAGS.use_gpu,
        run_mode=FLAGS.run_mode,
        use_dynamic_shape=FLAGS.use_dynamic_shape,
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
        use_gpu=FLAGS.use_gpu,
        run_mode=FLAGS.run_mode,
        use_dynamic_shape=FLAGS.use_dynamic_shape,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn)

    # predict from video file or camera video stream
    if FLAGS.video_file is not None or FLAGS.camera_id != -1:
        topdown_unite_predict_video(detector, topdown_keypoint_detector,
                                    FLAGS.camera_id)
    else:
        # predict from image
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        topdown_unite_predict(detector, topdown_keypoint_detector, img_list)


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)

    main()
