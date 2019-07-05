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
import multiprocessing

import paddle.fluid as fluid

from ppdet.utils.eval_utils import parse_fetches, eval_run, eval_results
import ppdet.utils.checkpoint as checkpoint
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu
from ppdet.modeling.model_input import create_feed
from ppdet.data.data_feed import create_reader
from ppdet.core.workspace import load_config, merge_config, create

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def main():
    """
    Main evaluate function
    """
    cfg = load_config(FLAGS.config)
    if 'architecture' in cfg:
        main_arch = cfg.architecture
    else:
        raise ValueError("'architecture' not specified in config file.")

    merge_config(FLAGS.opt)

    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)

    if cfg.use_gpu:
        devices_num = fluid.core.get_cuda_device_count()
    else:
        devices_num = int(
            os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    if 'eval_feed' not in cfg:
        eval_feed = create(main_arch + 'EvalFeed')
    else:
        eval_feed = create(cfg.eval_feed)

    # define executor
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # build program
    model = create(main_arch)
    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            pyreader, feed_vars = create_feed(eval_feed)
            fetches = model.eval(feed_vars)
    eval_prog = eval_prog.clone(True)

    reader = create_reader(eval_feed)
    pyreader.decorate_sample_list_generator(reader, place)

    # compile program for multi-devices
    if devices_num <= 1:
        compile_program = fluid.compiler.CompiledProgram(eval_prog)
    else:
        build_strategy = fluid.BuildStrategy()
        build_strategy.memory_optimize = False
        build_strategy.enable_inplace = False
        compile_program = fluid.compiler.CompiledProgram(
            eval_prog).with_data_parallel(build_strategy=build_strategy)

    # load model
    exe.run(startup_prog)
    if 'weights' in cfg:
        checkpoint.load_pretrain(exe, eval_prog, cfg.weights)

    extra_keys = []
    if 'metric' in cfg and cfg.metric == 'COCO':
        extra_keys = ['im_info', 'im_id', 'im_shape']

    keys, values, cls = parse_fetches(fetches, eval_prog, extra_keys)

    results = eval_run(exe, compile_program, pyreader, keys, values, cls)
    # evaluation
    resolution = None
    if 'mask' in results[0]:
        resolution = model.mask_head.resolution
    eval_results(results, eval_feed, cfg.metric, resolution, FLAGS.output_file)


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "-f",
        "--output_file",
        default=None,
        type=str,
        help="Evaluation file name, default to bbox.json and mask.json.")
    FLAGS = parser.parse_args()
    main()
