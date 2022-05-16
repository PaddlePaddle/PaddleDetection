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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')

import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Tracker
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_version, check_config
from ppdet.utils.cli import ArgsParser


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        '--video_file', type=str, default=None, help='Video name for tracking.')
    parser.add_argument(
        '--frame_rate',
        type=int,
        default=-1,
        help='Video frame rate for tracking.')
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--det_results_dir",
        type=str,
        default='',
        help="Directory name for detection results.")
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Directory name for output tracking results.')
    parser.add_argument(
        '--save_images',
        action='store_true',
        help='Save tracking results (image).')
    parser.add_argument(
        '--save_videos',
        action='store_true',
        help='Save tracking results (video).')
    parser.add_argument(
        '--show_image',
        action='store_true',
        help='Show tracking results (image).')
    parser.add_argument(
        '--scaled',
        type=bool,
        default=False,
        help="Whether coords after detector outputs are scaled, False in JDE YOLOv3 "
        "True in general detector.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    # build Tracker
    tracker = Tracker(cfg, mode='test')

    # load weights
    if cfg.architecture in ['DeepSORT', 'ByteTrack']:
        tracker.load_weights_sde(cfg.det_weights, cfg.reid_weights)
    else:
        tracker.load_weights_jde(cfg.weights)

    # inference
    tracker.mot_predict_seq(
        video_file=FLAGS.video_file,
        frame_rate=FLAGS.frame_rate,
        image_dir=FLAGS.image_dir,
        data_type=cfg.metric.lower(),
        model_type=cfg.architecture,
        output_dir=FLAGS.output_dir,
        save_images=FLAGS.save_images,
        save_videos=FLAGS.save_videos,
        show_image=FLAGS.show_image,
        scaled=FLAGS.scaled,
        det_results_dir=FLAGS.det_results_dir,
        draw_threshold=FLAGS.draw_threshold)


def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)

    # disable npu in config by default
    if 'use_npu' not in cfg:
        cfg.use_npu = False

    # disable xpu in config by default
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False

    if cfg.use_gpu:
        place = paddle.set_device('gpu')
    elif cfg.use_npu:
        place = paddle.set_device('npu')
    elif cfg.use_xpu:
        place = paddle.set_device('xpu')
    else:
        place = paddle.set_device('cpu')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_npu(cfg.use_npu)
    check_xpu(cfg.use_xpu)
    check_version()

    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
