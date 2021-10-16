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
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.engine import Trainer
from ppdet.slim import build_slim_model

from ppdet.utils.logger import setup_logger
logger = setup_logger('post_quant')


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_inference",
        help="Directory for storing the output model files.")
    parser.add_argument(
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    # build detector
    trainer = Trainer(cfg, mode='eval')

    # load weights
    if cfg.architecture in ['DeepSORT']:
        if cfg.det_weights != 'None':
            trainer.load_weights_sde(cfg.det_weights, cfg.reid_weights)
        else:
            trainer.load_weights_sde(None, cfg.reid_weights)
    else:
        trainer.load_weights(cfg.weights)

    # post quant model
    trainer.post_quant(FLAGS.output_dir)


def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    # TODO: to be refined in the future
    if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn':
        FLAGS.opt['norm_type'] = 'bn'
    merge_config(FLAGS.opt)

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    # FIXME: Temporarily solve the priority problem of FLAGS.opt
    merge_config(FLAGS.opt)
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
