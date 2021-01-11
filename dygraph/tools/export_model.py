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

# ignore numba warning
import warnings
warnings.filterwarnings('ignore')

import paddle

from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.engine import Detector

from ppdet.utils.logger import setup_logger
logger = setup_logger('export_model')


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_inference",
        help="Directory for storing the output model files.")
    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    # build detector
    detector = Detector(cfg, mode='test')

    # load weights
    detector.load_weights(cfg.weights, 'resume')

    # export model
    detector.export(FLAGS.output_dir)


def main():
    paddle.set_device("cpu")
    FLAGS = parse_args()

    cfg = load_config(FLAGS.config)
    # TODO: to be refined in the future
    if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn':
        FLAGS.opt['norm_type'] = 'bn'
    merge_config(FLAGS.opt)
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
