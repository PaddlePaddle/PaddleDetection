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
import os, sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import glob

import paddle
from paddle.distributed import ParallelEnv
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Tracker
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser

from ppdet.utils.logger import setup_logger
logger = setup_logger('train')


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        '--video_file',
        type=str,
        default='test.mp4',
        help='Video name for tracking.')
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
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    # build Tracker
    tracker = Tracker(cfg, mode='test')

    # load weights
    tracker.load_weights(cfg.weights, 'resume')

    # inference
    tracker.mot_predict(
        video_file=FLAGS.video_file,
        output_dir=FLAGS.output_dir,
        save_images=FLAGS.save_images,
        save_videos=FLAGS.save_videos,
        show_image=FLAGS.show_image)


def main():
    FLAGS = parse_args()

    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    if FLAGS.slim_config:
        slim_cfg = load_config(FLAGS.slim_config)
        merge_config(slim_cfg)
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    place = 'gpu:{}'.format(ParallelEnv().dev_id) if cfg.use_gpu else 'cpu'
    place = paddle.set_device(place)
    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
