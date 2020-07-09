from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
# ignore numba warning
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import paddle.fluid as fluid
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.reader import create_reader
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.py_op.post_process import get_det_res, get_seg_res
from ppdet.utils.eval_utils import coco_eval_results


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
    model = create(cfg.architecture, mode='infer', open_debug=cfg.open_debug)

    # Init Model  
    if os.path.isfile(cfg.weights):
        param_state_dict, opti_state_dict = fluid.load_dygraph(cfg.weights)
        model.set_dict(param_state_dict)

    # Data Reader 
    if FLAGS.use_gpu:
        devices_num = 1
    else:
        devices_num = int(os.environ.get('CPU_NUM', 1))
    eval_reader = create_reader(cfg.EvalReader, devices_num=devices_num)

    # Run Eval
    outs_res = []
    for iter_id, data in enumerate(eval_reader()):
        start_time = time.time()

        # forward 
        model.eval()
        outs = model(data, cfg['EvalReader']['inputs_def']['fields'])
        outs_res.append(outs)

        # log 
        cost_time = time.time() - start_time
        print("Eval iter: {}, time: {}".format(iter_id, cost_time))

    # Metric 
    coco_eval_results(
        outs_res, include_mask=True if 'MaskHed' in cfg else False)


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
