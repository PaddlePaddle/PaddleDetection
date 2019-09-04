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


def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be set before
# `import paddle`. Otherwise, it would not take any effect.
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

import paddle.fluid as fluid

from ppdet.utils.eval_utils import parse_fetches, eval_run, eval_results, json_eval_results
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

    reader = create_reader(eval_feed, args_path=FLAGS.dataset_dir)
    pyreader.decorate_sample_list_generator(reader, place)

    # eval already exists json file
    if FLAGS.json_eval:
        logger.info(
            "In json_eval mode, PaddleDetection will evaluate json files in "
            "output_eval directly. And proposal.json, bbox.json and mask.json "
            "will be detected by default.")
        json_eval_results(
            eval_feed, cfg.metric, json_directory=FLAGS.output_eval)
        return
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

    assert cfg.metric in ['COCO', 'VOC'], \
            "unknown metric type {}".format(cfg.metric)
    extra_keys = []
    if cfg.metric == 'COCO':
        extra_keys = ['im_info', 'im_id', 'im_shape']
    if cfg.metric == 'VOC':
        extra_keys = ['gt_box', 'gt_label', 'is_difficult']

    keys, values, cls = parse_fetches(fetches, eval_prog, extra_keys)

    # whether output bbox is normalized in model output layer
    is_bbox_normalized = False
    if hasattr(model, 'is_bbox_normalized') and \
            callable(model.is_bbox_normalized):
        is_bbox_normalized = model.is_bbox_normalized()

    results = eval_run(exe, compile_program, pyreader, keys, values, cls)
    # evaluation
    resolution = None
    if 'mask' in results[0]:
        resolution = model.mask_head.resolution
    eval_results(results, eval_feed, cfg.metric, cfg.num_classes, resolution,
                 is_bbox_normalized, FLAGS.output_eval, cfg.map_type)


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "--json_eval",
        action='store_true',
        default=False,
        help="Whether to re eval with already exists bbox.json or mask.json")
    parser.add_argument(
        "-d",
        "--dataset_dir",
        default=None,
        type=str,
        help="Dataset path, same as DataFeed.dataset.dataset_dir")
    parser.add_argument(
        "-f",
        "--output_eval",
        default=None,
        type=str,
        help="Evaluation file directory, default is current directory.")
    FLAGS = parser.parse_args()
    main()
