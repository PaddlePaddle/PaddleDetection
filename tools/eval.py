# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')

import paddle

from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_mlu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.engine import Trainer, init_parallel_env
from ppdet.metrics.coco_utils import json_eval_results
from ppdet.slim import build_slim_model

from ppdet.utils.logger import setup_logger
logger = setup_logger('eval')


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--output_eval",
        default=None,
        type=str,
        help="Evaluation directory, default is current directory.")

    parser.add_argument(
        '--json_eval',
        action='store_true',
        default=False,
        help='Whether to re eval with already exists bbox.json or mask.json')

    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")

    # TODO: bias should be unified
    parser.add_argument(
        "--bias",
        action="store_true",
        help="whether add bias or not while getting w and h")

    parser.add_argument(
        "--classwise",
        action="store_true",
        help="whether per-category AP and draw P-R Curve or not.")

    parser.add_argument(
        '--save_prediction_only',
        action='store_true',
        default=False,
        help='Whether to save the evaluation results only')

    parser.add_argument(
        "--amp",
        action='store_true',
        default=False,
        help="Enable auto mixed precision eval.")

    # for smalldet slice_infer
    parser.add_argument(
        "--slice_infer",
        action='store_true',
        help="Whether to slice the image and merge the inference results for small object detection."
    )
    parser.add_argument(
        '--slice_size',
        nargs='+',
        type=int,
        default=[640, 640],
        help="Height of the sliced image.")
    parser.add_argument(
        "--overlap_ratio",
        nargs='+',
        type=float,
        default=[0.25, 0.25],
        help="Overlap height ratio of the sliced image.")
    parser.add_argument(
        "--combine_method",
        type=str,
        default='nms',
        help="Combine method of the sliced images' detection results, choose in ['nms', 'nmm', 'concat']."
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.6,
        help="Combine method matching threshold.")
    parser.add_argument(
        "--match_metric",
        type=str,
        default='ios',
        help="Combine method matching metric, choose in ['iou', 'ios'].")
    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    if FLAGS.json_eval:
        logger.info(
            "In json_eval mode, PaddleDetection will evaluate json files in "
            "output_eval directly. And proposal.json, bbox.json and mask.json "
            "will be detected by default.")
        json_eval_results(
            cfg.metric,
            json_directory=FLAGS.output_eval,
            dataset=cfg['EvalDataset'])
        return

    # init parallel environment if nranks > 1
    init_parallel_env()

    # build trainer
    trainer = Trainer(cfg, mode='eval')

    # load weights
    trainer.load_weights(cfg.weights)

    # training
    if FLAGS.slice_infer:
        trainer.evaluate_slice(
            slice_size=FLAGS.slice_size,
            overlap_ratio=FLAGS.overlap_ratio,
            combine_method=FLAGS.combine_method,
            match_threshold=FLAGS.match_threshold,
            match_metric=FLAGS.match_metric)
    else:
        trainer.evaluate()


def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)

    # disable npu in config by default
    if 'use_npu' not in cfg:
        cfg.use_npu = False

    # disable xpu in config by default
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False

    if 'use_gpu' not in cfg:
        cfg.use_gpu = False

    # disable mlu in config by default
    if 'use_mlu' not in cfg:
        cfg.use_mlu = False

    if cfg.use_gpu:
        place = paddle.set_device('gpu')
    elif cfg.use_npu:
        place = paddle.set_device('npu')
    elif cfg.use_xpu:
        place = paddle.set_device('xpu')
    elif cfg.use_mlu:
        place = paddle.set_device('mlu')
    else:
        place = paddle.set_device('cpu')

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='eval')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_npu(cfg.use_npu)
    check_xpu(cfg.use_xpu)
    check_mlu(cfg.use_mlu)
    check_version()

    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
