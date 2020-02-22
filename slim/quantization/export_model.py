# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import fluid

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.cli import ArgsParser
import ppdet.utils.checkpoint as checkpoint
from tools.export_model import prune_feed_vars

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
from paddleslim.quant import quant_aware, convert


def save_infer_model(save_dir, exe, feed_vars, test_fetches, infer_prog):
    feed_var_names = [var.name for var in feed_vars.values()]
    target_vars = list(test_fetches.values())
    feed_var_names = prune_feed_vars(feed_var_names, target_vars, infer_prog)
    logger.info("Export inference model to {}, input: {}, output: "
                "{}...".format(save_dir, feed_var_names,
                               [str(var.name) for var in target_vars]))
    fluid.io.save_inference_model(
        save_dir,
        feeded_var_names=feed_var_names,
        target_vars=target_vars,
        executor=exe,
        main_program=infer_prog,
        params_filename="__params__")


def main():
    cfg = load_config(FLAGS.config)

    if 'architecture' in cfg:
        main_arch = cfg.architecture
    else:
        raise ValueError("'architecture' not specified in config file.")

    merge_config(FLAGS.opt)

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

    not_quant_pattern = []
    if FLAGS.not_quant_pattern:
        not_quant_pattern = FLAGS.not_quant_pattern
    config = {
        'weight_quantize_type': 'channel_wise_abs_max',
        'activation_quantize_type': 'moving_average_abs_max',
        'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d'],
        'not_quant_pattern': not_quant_pattern
    }

    infer_prog = quant_aware(infer_prog, place, config, for_test=True)

    exe.run(startup_prog)
    checkpoint.load_params(exe, infer_prog, cfg.weights)

    infer_prog, int8_program = convert(
        infer_prog, place, config, save_int8=True)

    save_infer_model(
        os.path.join(FLAGS.output_dir, 'float'), exe, feed_vars, test_fetches,
        infer_prog)

    save_infer_model(
        os.path.join(FLAGS.output_dir, 'int'), exe, feed_vars, test_fetches,
        int8_program)


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output model files.")
    parser.add_argument(
        "--not_quant_pattern",
        nargs='+',
        type=str,
        help="Layers which name_scope contains string in not_quant_pattern will not be quantized"
    )

    FLAGS = parser.parse_args()
    main()
