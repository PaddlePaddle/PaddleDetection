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
logger = setup_logger('eval')


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        '--benchmark',
        default='MOT16_train',
        type=str,
        help='Benchmark name for tracking dataset.')
    parser.add_argument(
        '--data_root',
        type=str,
        default='./dataset/MOT',
        help='Directory for tracking dataset.')
    parser.add_argument(
        "--data_type",
        default='mot',
        type=str,
        help='Data type of tracking dataset, should be in ["mot", "kitti"]')
    parser.add_argument(
        "--model_type",
        default='jde',
        type=str,
        help='Model type of tracking, should be in ["jde", "deepsort", "fairmot"]'
    )
    parser.add_argument(
        "--det_dir",
        default='output/mot_results/',
        type=str,
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
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    if FLAGS.benchmark == 'MOT15_train':
        data_root = '{}/MOT15/images/train'.format(FLAGS.data_root)
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2'''
    elif FLAGS.benchmark == 'MOT15_test':
        data_root = '{}/MOT15/images/test'.format(FLAGS.data_root)
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
    elif FLAGS.benchmark == 'MOT16_train':
        data_root = '{}/MOT16/images/train'.format(FLAGS.data_root)
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
    elif FLAGS.benchmark == 'MOT16_test':
        data_root = '{}/MOT16/images/test'.format(FLAGS.data_root)
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
    elif FLAGS.benchmark == 'MOT17_train':
        data_root = '{}/MOT17/images/train'.format(FLAGS.data_root)
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
    elif FLAGS.benchmark == 'MOT17_test':
        data_root = '{}/MOT17/images/test'.format(FLAGS.data_root)
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
    elif FLAGS.benchmark == 'MOT20_train':
        data_root = '{}/MOT20/images/train'.format(FLAGS.data_root)
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05'''
    elif FLAGS.benchmark == 'MOT20_test':
        data_root = '{}/MOT20/images/test'.format(FLAGS.data_root)
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08'''
    else:
        data_root = '{}/MOT16/images/train'.format(FLAGS.data_root)
        seqs_str = '''MOT16-02
                   '''
    seqs = [seq.strip() for seq in seqs_str.split()]

    # build Tracker
    tracker = Tracker(cfg, mode='eval')

    # load weights
    if FLAGS.model_type == 'deepsort':
        if cfg.det_weights != 'None':
            tracker.load_weights_deepsort(cfg.det_weights, cfg.reid_weights,
                                          'resume')
        else:
            tracker.load_weights_deepsort(None, cfg.reid_weights, 'resume')
    else:
        tracker.load_weights(cfg.weights, 'resume')

    # inference
    tracker.mot_evaluate(
        data_root=data_root,
        seqs=seqs,
        data_type=FLAGS.data_type,
        model_type=FLAGS.model_type,
        output_dir=FLAGS.output_dir,
        save_images=FLAGS.save_images,
        save_videos=FLAGS.save_videos,
        show_image=FLAGS.show_image,
        det_dir=FLAGS.det_dir)


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
