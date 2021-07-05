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

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from paddle import fluid

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

try:
    from ppdet.core.workspace import load_config, merge_config, create
    from ppdet.utils.cli import ArgsParser
    import ppdet.utils.checkpoint as checkpoint
    from ppdet.utils.export_utils import save_infer_model, dump_infer_config
    from ppdet.utils.check import check_config, check_version, enable_static_mode
except ImportError as e:
    if sys.argv[0].find('static') >= 0:
        logger.error("Importing ppdet failed when running static model "
                     "with error: {}\n"
                     "please try:\n"
                     "\t1. run static model under PaddleDetection/static "
                     "directory\n"
                     "\t2. run 'pip uninstall ppdet' to uninstall ppdet "
                     "dynamic version firstly.".format(e))
        sys.exit(-1)
    else:
        raise e

from paddleslim.prune import Pruner
from paddleslim.analysis import flops


def main():
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    check_version()

    main_arch = cfg.architecture

    # Use CPU for exporting inference model instead of GPU
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    model = create(main_arch)

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs_def = cfg['TestReader']['inputs_def']
            inputs_def['use_dataloader'] = False
            feed_vars, _ = model.build_inputs(**inputs_def)
            test_fetches = model.test(feed_vars)
    infer_prog = infer_prog.clone(True)

    exe.run(startup_prog)
    checkpoint.load_checkpoint(exe, infer_prog, cfg.weights)

    pruned_params = FLAGS.pruned_params
    assert (
        FLAGS.pruned_params is not None
    ), "FLAGS.pruned_params is empty!!! Please set it by '--pruned_params' option."
    pruned_params = FLAGS.pruned_params.strip().split(",")
    logger.info("pruned params: {}".format(pruned_params))
    pruned_ratios = [float(n) for n in FLAGS.pruned_ratios.strip().split(",")]
    logger.info("pruned ratios: {}".format(pruned_ratios))
    assert (len(pruned_params) == len(pruned_ratios)
            ), "The length of pruned params and pruned ratios should be equal."
    assert (pruned_ratios > [0] * len(pruned_ratios) and
            pruned_ratios < [1] * len(pruned_ratios)
            ), "The elements of pruned ratios should be in range (0, 1)."

    base_flops = flops(infer_prog)
    pruner = Pruner()
    infer_prog, _, _ = pruner.prune(
        infer_prog,
        fluid.global_scope(),
        params=pruned_params,
        ratios=pruned_ratios,
        place=place,
        only_graph=True)
    pruned_flops = flops(infer_prog)
    logger.info("pruned FLOPS: {}".format(
        float(base_flops - pruned_flops) / base_flops))

    dump_infer_config(FLAGS, cfg)
    save_infer_model(FLAGS, exe, feed_vars, test_fetches, infer_prog)


if __name__ == '__main__':
    enable_static_mode()
    parser = ArgsParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output model files.")

    parser.add_argument(
        "-p",
        "--pruned_params",
        default=None,
        type=str,
        help="The parameters to be pruned when calculating sensitivities.")
    parser.add_argument(
        "--pruned_ratios",
        default=None,
        type=str,
        help="The ratios pruned iteratively for each parameter when calculating sensitivities."
    )

    FLAGS = parser.parse_args()
    main()
