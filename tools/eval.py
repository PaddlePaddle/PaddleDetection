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

import time
# ignore numba warning
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import paddle
from paddle.distributed import ParallelEnv
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.utils.eval_utils import get_infer_results, eval_results
from ppdet.utils.checkpoint import load_weight
import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--output_eval",
        default=None,
        type=str,
        help="Evaluation directory, default is current directory.")

    parser.add_argument(
        '--json_eval', action='store_true', default=False, help='')

    parser.add_argument(
        '--use_gpu', action='store_true', default=False, help='')

    args = parser.parse_args()
    return args


def run(FLAGS, cfg, place):

    # Model
    main_arch = cfg.architecture
    model = create(cfg.architecture)

    # Init Model
    load_weight(model, cfg.weights)

    # Data Reader
    dataset = cfg.EvalDataset
    eval_loader, _ = create('EvalReader')(dataset, cfg['worker_num'], place)

    # Run Eval
    outs_res = []
    start_time = time.time()
    sample_num = 0
    im_info = []
    for iter_id, data in enumerate(eval_loader):
        # forward
        fields = cfg['EvalReader']['inputs_def']['fields']
        model.eval()
        outs = model(data=data, input_def=fields, mode='infer')
        for key, value in outs.items():
            outs[key] = value.numpy()
        im_shape = data[fields.index('im_shape')].numpy()
        scale_factor = data[fields.index('scale_factor')].numpy()
        im_id = data[fields.index('im_id')].numpy()
        im_info.append([im_shape, scale_factor, im_id])

        if 'mask' in outs and 'bbox' in outs:
            mask_resolution = model.mask_post_process.mask_resolution
            from ppdet.py_op.post_process import mask_post_process
            outs['mask'] = mask_post_process(outs, im_shape, scale_factor,
                                             mask_resolution)

        outs_res.append(outs)
        # log
        sample_num += im_shape.shape[0]
        if iter_id % 100 == 0:
            logger.info("Eval iter: {}".format(iter_id))

    cost_time = time.time() - start_time
    logger.info('Total sample number: {}, averge FPS: {}'.format(
        sample_num, sample_num / cost_time))

    eval_type = []
    if 'bbox' in outs:
        eval_type.append('bbox')
    if 'mask' in outs:
        eval_type.append('mask')
    # Metric
    # TODO: support other metric
    from ppdet.utils.coco_eval import get_category_info
    anno_file = dataset.get_anno()
    with_background = cfg.with_background
    use_default_label = dataset.use_default_label
    clsid2catid, catid2name = get_category_info(anno_file, with_background,
                                                use_default_label)

    infer_res = get_infer_results(outs_res, eval_type, clsid2catid, im_info)
    eval_results(infer_res, cfg.metric, anno_file)


def main():
    FLAGS = parse_args()

    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    place = 'gpu:{}'.format(ParallelEnv().dev_id) if cfg.use_gpu else 'cpu'
    place = paddle.set_device(place)
    run(FLAGS, cfg, place)


if __name__ == '__main__':
    main()
