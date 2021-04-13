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

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np

import paddle
from paddle.distributed import ParallelEnv

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
from ppdet.engine import Trainer, init_parallel_env, set_random_seed

import ppdet.utils.cli as cli
import ppdet.utils.check as check
from ppdet.utils.logger import setup_logger
logger = setup_logger('train')


def parse_args():
    parser = cli.ArgsParser()
    parser.add_argument(
        "--weight_type",
        default='pretrain',
        type=str,
        help="Loading Checkpoints only support 'pretrain', 'finetune', 'resume'."
    )
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation in train")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    parser.add_argument(
        "--enable_ce",
        type=bool,
        default=False,
        help="If set True, enable continuous evaluation job."
        "This flag is only used for internal test.")
    parser.add_argument(
        '--save_prediction_only',
        action='store_true',
        default=False,
        help='Whether to save the evaluation results only')
    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    # init parallel environment if nranks > 1
    init_parallel_env()

    if FLAGS.enable_ce:
        set_random_seed(0)

    # build trainer
    trainer = Trainer(cfg, mode='train')

    # load weights
    if not FLAGS.slim_config and 'pretrain_weights' in cfg and cfg.pretrain_weights:
        trainer.load_weights(cfg.pretrain_weights, FLAGS.weight_type)

    # training
    trainer.train(FLAGS.eval)


def main():
    FLAGS = parse_args()

    cfg = load_config(FLAGS.config)
    cfg['save_prediction_only'] = FLAGS.save_prediction_only
    merge_config(FLAGS.opt)
    if FLAGS.slim_config:
        slim_cfg = load_config(FLAGS.slim_config)
        merge_config(slim_cfg)
        if 'weight_type' not in cfg:
            cfg.weight_type = FLAGS.weight_type
    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)
    check.check_version()

    place = 'gpu:{}'.format(ParallelEnv().dev_id) if cfg.use_gpu else 'cpu'
    place = paddle.set_device(place)

    run(FLAGS, cfg)


if __name__ == "__main__":
    main()
