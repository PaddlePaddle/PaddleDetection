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
import glob
import numpy as np
from PIL import Image
import paddle
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.utils.checkpoint import load_weight
from ppdet.utils.export_utils import dump_infer_config
from paddle.jit import to_static
import paddle.nn as nn
from paddle.static import InputSpec
import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


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

    # Model
    main_arch = cfg.architecture
    model = create(cfg.architecture)
    inputs_def = cfg['TestReader']['inputs_def']
    assert 'image_shape' in inputs_def, 'image_shape must be specified.'
    image_shape = inputs_def.get('image_shape')

    assert not None in image_shape, 'image_shape should not contain None'
    cfg_name = os.path.basename(FLAGS.config).split('.')[0]
    save_dir = os.path.join(FLAGS.output_dir, cfg_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_shape = dump_infer_config(cfg,
                                    os.path.join(save_dir, 'infer_cfg.yml'),
                                    image_shape)

    class ExportModel(nn.Layer):
        def __init__(self, model):
            super(ExportModel, self).__init__()
            self.model = model

        @to_static(input_spec=[
            {
                'image': InputSpec(
                    shape=[None] + image_shape, name='image')
            },
            {
                'im_shape': InputSpec(
                    shape=[None, 2], dtype='int32', name='im_shape')
            },
            {
                'scale_factor': InputSpec(
                    shape=[None, 2], name='scale_factor')
            },
        ])
        def forward(self, image, im_shape, scale_factor):
            inputs = {}
            inputs_tensor = [image, im_shape, scale_factor]
            for t in inputs_tensor:
                inputs.update(t)
            outs = self.model.get_export_model(inputs)
            return outs

    export_model = ExportModel(model)
    # debug for dy2static, remove later
    #paddle.jit.set_code_level()

    # Init Model
    load_weight(export_model.model, cfg.weights)

    export_model.eval()

    # export config and model
    paddle.jit.save(export_model, os.path.join(save_dir, 'model'))
    logger.info('Export model to {}'.format(save_dir))


def main():
    paddle.set_device("cpu")
    FLAGS = parse_args()

    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
