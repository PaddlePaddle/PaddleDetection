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
from ppdet.utils.check import check_gpu
from ppdet.modeling.model_input import create_feed
from ppdet.data.data_feed import create_reader
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.cli import print_total_cfg
from ppdet.utils.cli import ArgsParser

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
    print_total_cfg(cfg)

    if 'eval_feed' not in cfg:
        eval_feed = create(main_arch + 'EvalFeed')
    else:
        eval_feed = create(cfg.eval_feed)

    multi_scale_test = getattr(cfg, 'MultiScaleTEST', None)

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
            if multi_scale_test is None:
                fetches = model.eval(feed_vars)
            else:
                fetches = model.eval(feed_vars, multi_scale_test)
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

    compile_program = fluid.compiler.CompiledProgram(
        eval_prog).with_data_parallel()

    # load model
    exe.run(startup_prog)
    if 'weights' in cfg:
        checkpoint.load_params(exe, eval_prog, cfg.weights)

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

    sub_eval_prog = None
    sub_keys = None
    sub_values = None
    # build sub-program
    if 'Mask' in main_arch and multi_scale_test:
        sub_eval_prog = fluid.Program()
        with fluid.program_guard(sub_eval_prog, startup_prog):
            with fluid.unique_name.guard():
                _, feed_vars = create_feed(
                    eval_feed, use_pyreader=False, sub_prog_feed=True)
                sub_fetches = model.eval(
                    feed_vars, multi_scale_test, mask_branch=True)
                extra_keys = []
                if cfg.metric == 'COCO':
                    extra_keys = ['im_id', 'im_shape']
                if cfg.metric == 'VOC':
                    extra_keys = ['gt_box', 'gt_label', 'is_difficult']
        sub_keys, sub_values, _ = parse_fetches(sub_fetches, sub_eval_prog,
                                                extra_keys)
        sub_eval_prog = sub_eval_prog.clone(True)

        if 'weights' in cfg:
            checkpoint.load_params(exe, sub_eval_prog, cfg.weights)

    results = eval_run(exe, compile_program, pyreader, keys, values, cls, cfg,
                       sub_eval_prog, sub_keys, sub_values)

    # evaluation
    resolution = None
    if 'mask' in results[0]:
        resolution = model.mask_head.resolution
    # if map_type not set, use default 11point, only use in VOC eval
    map_type = cfg.map_type if 'map_type' in cfg else '11point'
    eval_results(results, eval_feed, cfg.metric, cfg.num_classes, resolution,
                 is_bbox_normalized, FLAGS.output_eval, map_type)


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
