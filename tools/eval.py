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

import os, sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import paddle

from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
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
    trainer.evaluate()


def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    # TODO: bias should be unified
    cfg['bias'] = 1 if FLAGS.bias else 0
    cfg['classwise'] = True if FLAGS.classwise else False
    cfg['output_eval'] = FLAGS.output_eval
    merge_config(FLAGS.opt)

    place = paddle.set_device('gpu' if cfg.use_gpu else 'cpu')

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='eval')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
