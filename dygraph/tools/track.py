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
        '--benchmark',
        default='MOT16_train',
        type=str,
        help='Benchmark name for tracking')
    parser.add_argument(
        '--data_root',
        type=str,
        default='./dataset/MOT',
        help='Directory for tracking dataset')
    parser.add_argument(
        '--exp_name', type=str, default='demo', help='exp_name of tracking dir')

    parser.add_argument(
        '--save_images',
        action='store_true',
        help='Save tracking results (image)')
    parser.add_argument(
        '--save_videos',
        action='store_true',
        help='Save tracking results (video)')
    parser.add_argument(
        '--show_image',
        action='store_true',
        help='Show tracking results (image)')
    parser.add_argument(
        '--save_dir',
        default=None,
        type=str,
        help='Directory for saved tracking results (image)')
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    if FLAGS.benchmark == 'MOT17_train':
        data_root = '{}/MOT17/images/train'.format(FLAGS.data_root)
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP
                    '''
    elif FLAGS.benchmark == 'MOT16_train':
        data_root = '{}/MOT16/images/train'.format(FLAGS.data_root)
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13
                   '''
    elif FLAGS.benchmark == 'MOT16_test':
        data_root = '{}/MOT16/images/test'.format(FLAGS.data_root)
        seqs_str = '''MOT16-01
                      MOT16-02
                      MOT16-03
                      MOT16-04
                      MOT16-05
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-12
                      MOT16-13
                      MOT16-14
                   '''
    elif FLAGS.benchmark == 'MOT16_debug':
        data_root = '{}/MOT16/images/train'.format(FLAGS.data_root)
        seqs_str = '''MOT16-02
                   '''
    seqs = [seq.strip() for seq in seqs_str.split()]

    # build Tracker
    tracker = Tracker(cfg, mode='track')

    # load weights
    tracker.load_weights(cfg.weights, 'resume')

    # inference
    tracker.track(
        data_root=data_root,
        seqs=seqs,
        exp_name=FLAGS.exp_name,
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
