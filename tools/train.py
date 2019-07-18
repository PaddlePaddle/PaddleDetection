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
import time
import multiprocessing
import numpy as np


def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect. 
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

from paddle import fluid

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.data_feed import create_reader

from ppdet.utils.eval_utils import parse_fetches, eval_run, eval_results
from ppdet.utils.stats import TrainingStats
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu
import ppdet.utils.checkpoint as checkpoint
from ppdet.modeling.model_input import create_feed

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def main():
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

    if 'train_feed' not in cfg:
        train_feed = create(main_arch + 'TrainFeed')
    else:
        train_feed = create(cfg.train_feed)

    if FLAGS.eval:
        if 'eval_feed' not in cfg:
            eval_feed = create(main_arch + 'EvalFeed')
        else:
            eval_feed = create(cfg.eval_feed)

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    lr_builder = create('LearningRate')
    optim_builder = create('OptimizerBuilder')

    # build program
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            model = create(main_arch)
            train_pyreader, feed_vars = create_feed(train_feed)
            train_fetches = model.train(feed_vars)
            loss = train_fetches['loss']
            lr = lr_builder()
            optimizer = optim_builder(lr)
            optimizer.minimize(loss)

    train_reader = create_reader(train_feed, cfg.max_iters * devices_num)
    train_pyreader.decorate_sample_list_generator(train_reader, place)

    # parse train fetches
    train_keys, train_values, _ = parse_fetches(train_fetches)
    train_values.append(lr)

    if FLAGS.eval:
        eval_prog = fluid.Program()
        with fluid.program_guard(eval_prog, startup_prog):
            with fluid.unique_name.guard():
                model = create(main_arch)
                eval_pyreader, feed_vars = create_feed(eval_feed)
                fetches = model.eval(feed_vars)
        eval_prog = eval_prog.clone(True)

        eval_reader = create_reader(eval_feed)
        eval_pyreader.decorate_sample_list_generator(eval_reader, place)

        # parse eval fetches
        extra_keys = ['im_info', 'im_id',
                      'im_shape'] if cfg.metric == 'COCO' else []
        eval_keys, eval_values, eval_cls = parse_fetches(fetches, eval_prog,
                                                         extra_keys)

    # compile program for multi-devices
    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = True
    sync_bn = getattr(model.backbone, 'norm_type', None) == 'sync_bn'
    # only enable sync_bn in multi GPU devices
    build_strategy.sync_batch_norm = sync_bn and devices_num > 1 \
         and cfg.use_gpu
    train_compile_program = fluid.compiler.CompiledProgram(
        train_prog).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy)
    if FLAGS.eval:
        eval_compile_program = fluid.compiler.CompiledProgram(eval_prog)

    exe.run(startup_prog)

    fuse_bn = getattr(model.backbone, 'norm_type', None) == 'affine_channel'
    start_iter = 0
    if FLAGS.resume_checkpoint:
        checkpoint.load_checkpoint(exe, train_prog, FLAGS.resume_checkpoint)
        start_iter = checkpoint.global_step()
    elif cfg.pretrain_weights and fuse_bn:
        checkpoint.load_and_fusebn(exe, train_prog, cfg.pretrain_weights)
    elif cfg.pretrain_weights:
        checkpoint.load_pretrain(exe, train_prog, cfg.pretrain_weights)

    train_stats = TrainingStats(cfg.log_smooth_window, train_keys)
    train_pyreader.start()
    start_time = time.time()
    end_time = time.time()

    cfg_name = os.path.basename(FLAGS.config).split('.')[0]
    save_dir = os.path.join(cfg.save_dir, cfg_name)
    for it in range(start_iter, cfg.max_iters):
        start_time = end_time
        end_time = time.time()
        outs = exe.run(train_compile_program, fetch_list=train_values)
        stats = {k: np.array(v).mean() for k, v in zip(train_keys, outs[:-1])}
        train_stats.update(stats)
        logs = train_stats.log()
        strs = 'iter: {}, lr: {:.6f}, {}, time: {:.3f}'.format(
            it, np.mean(outs[-1]), logs, end_time - start_time)
        logger.info(strs)

        if it > 0 and it % cfg.snapshot_iter == 0:
            checkpoint.save(exe, train_prog, os.path.join(save_dir, str(it)))

            if FLAGS.eval:
                # evaluation
                results = eval_run(exe, eval_compile_program, eval_pyreader,
                                   eval_keys, eval_values, eval_cls)
                resolution = None
                if 'mask' in results[0]:
                    resolution = model.mask_head.resolution
                eval_results(results, eval_feed, cfg.metric, resolution,
                             FLAGS.output_file)

    checkpoint.save(exe, train_prog, os.path.join(save_dir, "model_final"))
    train_pyreader.reset()


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        default=None,
        type=str,
        help="Checkpoint path for resuming training.")
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation in train")
    parser.add_argument(
        "-f",
        "--output_file",
        default=None,
        type=str,
        help="Evaluation file name, default to bbox.json and mask.json.")
    FLAGS = parser.parse_args()
    main()
