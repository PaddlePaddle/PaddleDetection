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
import paddle.fluid as fluid
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.utils.eval_utils import coco_eval_results
from ppdet.data.reader import create_reader
from ppdet.utils.checkpoint import load_dygraph_ckpt, save_dygraph_ckpt
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


def run(FLAGS, cfg):

    # Model
    main_arch = cfg.architecture
    model = create(cfg.architecture)

    # Init Model  
    model = load_dygraph_ckpt(model, ckpt=cfg.weights)

    # Data Reader 
    if FLAGS.use_gpu:
        devices_num = 1
    else:
        devices_num = int(os.environ.get('CPU_NUM', 1))
    eval_reader = create_reader(cfg.EvalReader, devices_num=devices_num)

    # Run Eval
    outs_res = []
    start_time = time.time()
    sample_num = 0
    for iter_id, data in enumerate(eval_reader()):
        # forward 
        model.eval()
        outs = model(data, cfg['EvalReader']['inputs_def']['fields'], 'infer')
        outs_res.append(outs)

        # log 
        sample_num += len(data)
        if iter_id % 100 == 0:
            logger.info("Eval iter: {}".format(iter_id))

    cost_time = time.time() - start_time
    logger.info('Total sample number: {}, averge FPS: {}'.format(
        sample_num, sample_num / cost_time))
    # Metric 
    coco_eval_results(
        outs_res,
        include_mask=True if getattr(cfg, 'MaskHead', None) else False,
        dataset=cfg['EvalReader']['dataset'])


def main():
    FLAGS = parse_args()

    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env()
                            .dev_id) if cfg.use_gpu else fluid.CPUPlace()

    with fluid.dygraph.guard(place):
        run(FLAGS, cfg)


if __name__ == '__main__':
    main()
