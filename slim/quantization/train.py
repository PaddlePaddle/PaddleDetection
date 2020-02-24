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
import time
import numpy as np
import datetime
from collections import deque
import shutil

from paddle import fluid

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.reader import create_reader

from ppdet.utils.cli import print_total_cfg
from ppdet.utils import dist_utils
from ppdet.utils.eval_utils import parse_fetches, eval_run, eval_results
from ppdet.utils.stats import TrainingStats
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu, check_version
import ppdet.utils.checkpoint as checkpoint
from paddleslim.quant import quant_aware, convert
import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def save_checkpoint(exe, prog, path, train_prog):
    if os.path.isdir(path):
        shutil.rmtree(path)
    logger.info('Save model to {}.'.format(path))
    fluid.io.save_persistables(exe, path, main_program=prog)

    v = train_prog.global_block().var('@LR_DECAY_COUNTER@')
    fluid.io.save_vars(exe, dirname=path, vars=[v])


def load_global_step(exe, prog, path):
    v = prog.global_block().var('@LR_DECAY_COUNTER@')
    fluid.io.load_vars(exe, path, prog, [v])


def main():
    env = os.environ
    FLAGS.dist = 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env
    if FLAGS.dist:
        trainer_id = int(env['PADDLE_TRAINER_ID'])
        import random
        local_seed = (99 + trainer_id)
        random.seed(local_seed)
        np.random.seed(local_seed)

    cfg = load_config(FLAGS.config)
    if 'architecture' in cfg:
        main_arch = cfg.architecture
    else:
        raise ValueError("'architecture' not specified in config file.")

    merge_config(FLAGS.opt)

    if 'log_iter' not in cfg:
        cfg.log_iter = 20

    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()
    if not FLAGS.dist or trainer_id == 0:
        print_total_cfg(cfg)

    if cfg.use_gpu:
        devices_num = fluid.core.get_cuda_device_count()
    else:
        devices_num = int(os.environ.get('CPU_NUM', 1))

    if 'FLAGS_selected_gpus' in env:
        device_id = int(env['FLAGS_selected_gpus'])
    else:
        device_id = 0
    place = fluid.CUDAPlace(device_id) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    lr_builder = create('LearningRate')
    optim_builder = create('OptimizerBuilder')

    # build program
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            model = create(main_arch)

            inputs_def = cfg['TrainReader']['inputs_def']
            feed_vars, train_loader = model.build_inputs(**inputs_def)
            train_fetches = model.train(feed_vars)
            loss = train_fetches['loss']
            lr = lr_builder()
            optimizer = optim_builder(lr)
            optimizer.minimize(loss)

    # parse train fetches
    train_keys, train_values, _ = parse_fetches(train_fetches)
    train_values.append(lr)

    if FLAGS.eval:
        eval_prog = fluid.Program()
        with fluid.program_guard(eval_prog, startup_prog):
            with fluid.unique_name.guard():
                model = create(main_arch)
                inputs_def = cfg['EvalReader']['inputs_def']
                feed_vars, eval_loader = model.build_inputs(**inputs_def)
                fetches = model.eval(feed_vars)
        eval_prog = eval_prog.clone(True)

        eval_reader = create_reader(cfg.EvalReader)
        eval_loader.set_sample_list_generator(eval_reader, place)

        # parse eval fetches
        extra_keys = []
        if cfg.metric == 'COCO':
            extra_keys = ['im_info', 'im_id', 'im_shape']
        if cfg.metric == 'VOC':
            extra_keys = ['gt_bbox', 'gt_class', 'is_difficult']
        if cfg.metric == 'WIDERFACE':
            extra_keys = ['im_id', 'im_shape', 'gt_bbox']
        eval_keys, eval_values, eval_cls = parse_fetches(fetches, eval_prog,
                                                         extra_keys)

    # compile program for multi-devices
    build_strategy = fluid.BuildStrategy()
    build_strategy.fuse_all_optimizer_ops = False
    build_strategy.fuse_elewise_add_act_ops = True
    build_strategy.fuse_all_reduce_ops = False

    # only enable sync_bn in multi GPU devices
    sync_bn = getattr(model.backbone, 'norm_type', None) == 'sync_bn'
    sync_bn = False
    build_strategy.sync_batch_norm = sync_bn and devices_num > 1 \
        and cfg.use_gpu

    exec_strategy = fluid.ExecutionStrategy()
    # iteration number when CompiledProgram tries to drop local execution scopes.
    # Set it to be 1 to save memory usages, so that unused variables in
    # local execution scopes can be deleted after each iteration.
    exec_strategy.num_iteration_per_drop_scope = 1
    if FLAGS.dist:
        dist_utils.prepare_for_multi_process(exe, build_strategy, startup_prog,
                                             train_prog)
        exec_strategy.num_threads = 1

    exe.run(startup_prog)
    not_quant_pattern = []
    if FLAGS.not_quant_pattern:
        not_quant_pattern = FLAGS.not_quant_pattern
    config = {
        'weight_quantize_type': 'channel_wise_abs_max',
        'activation_quantize_type': 'moving_average_abs_max',
        'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d'],
        'not_quant_pattern': not_quant_pattern
    }

    ignore_params = cfg.finetune_exclude_pretrained_params \
                 if 'finetune_exclude_pretrained_params' in cfg else []

    fuse_bn = getattr(model.backbone, 'norm_type', None) == 'affine_channel'

    if not FLAGS.resume_checkpoint:
        if cfg.pretrain_weights and fuse_bn and not ignore_params:
            checkpoint.load_and_fusebn(exe, train_prog, cfg.pretrain_weights)
        elif cfg.pretrain_weights:
            checkpoint.load_params(
                exe,
                train_prog,
                cfg.pretrain_weights,
                ignore_params=ignore_params)
    # insert quantize op in train_prog, return type is CompiledProgram
    train_prog_quant = quant_aware(train_prog, place, config, for_test=False)

    compiled_train_prog = train_prog_quant.with_data_parallel(
        loss_name=loss.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    if FLAGS.eval:
        # insert quantize op in eval_prog
        eval_prog = quant_aware(eval_prog, place, config, for_test=True)

        compiled_eval_prog = fluid.compiler.CompiledProgram(eval_prog)

    start_iter = 0
    if FLAGS.resume_checkpoint:
        checkpoint.load_checkpoint(exe, eval_prog, FLAGS.resume_checkpoint)
        load_global_step(exe, train_prog, FLAGS.resume_checkpoint)
        start_iter = checkpoint.global_step()

    train_reader = create_reader(cfg.TrainReader,
                                 (cfg.max_iters - start_iter) * devices_num)
    train_loader.set_sample_list_generator(train_reader, place)

    # whether output bbox is normalized in model output layer
    is_bbox_normalized = False
    if hasattr(model, 'is_bbox_normalized') and \
            callable(model.is_bbox_normalized):
        is_bbox_normalized = model.is_bbox_normalized()

    # if map_type not set, use default 11point, only use in VOC eval
    map_type = cfg.map_type if 'map_type' in cfg else '11point'

    train_stats = TrainingStats(cfg.log_smooth_window, train_keys)
    train_loader.start()
    start_time = time.time()
    end_time = time.time()

    cfg_name = os.path.basename(FLAGS.config).split('.')[0]
    save_dir = os.path.join(cfg.save_dir, cfg_name)
    time_stat = deque(maxlen=cfg.log_smooth_window)
    best_box_ap_list = [0.0, 0]  #[map, iter]

    for it in range(start_iter, cfg.max_iters):
        start_time = end_time
        end_time = time.time()
        time_stat.append(end_time - start_time)
        time_cost = np.mean(time_stat)
        eta_sec = (cfg.max_iters - it) * time_cost
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        outs = exe.run(compiled_train_prog, fetch_list=train_values)
        stats = {k: np.array(v).mean() for k, v in zip(train_keys, outs[:-1])}

        train_stats.update(stats)
        logs = train_stats.log()
        if it % cfg.log_iter == 0 and (not FLAGS.dist or trainer_id == 0):
            strs = 'iter: {}, lr: {:.6f}, {}, time: {:.3f}, eta: {}'.format(
                it, np.mean(outs[-1]), logs, time_cost, eta)
            logger.info(strs)

        if (it > 0 and it % cfg.snapshot_iter == 0 or it == cfg.max_iters - 1) \
           and (not FLAGS.dist or trainer_id == 0):
            save_name = str(it) if it != cfg.max_iters - 1 else "model_final"
            save_checkpoint(exe, eval_prog,
                            os.path.join(save_dir, save_name), train_prog)

            if FLAGS.eval:
                # evaluation
                results = eval_run(exe, compiled_eval_prog, eval_loader,
                                   eval_keys, eval_values, eval_cls)
                resolution = None
                if 'mask' in results[0]:
                    resolution = model.mask_head.resolution
                box_ap_stats = eval_results(
                    results, cfg.metric, cfg.num_classes, resolution,
                    is_bbox_normalized, FLAGS.output_eval, map_type,
                    cfg['EvalReader']['dataset'])

                if box_ap_stats[0] > best_box_ap_list[0]:
                    best_box_ap_list[0] = box_ap_stats[0]
                    best_box_ap_list[1] = it
                    save_checkpoint(exe, eval_prog,
                                    os.path.join(save_dir, "best_model"),
                                    train_prog)
                logger.info("Best test box ap: {}, in iter: {}".format(
                    best_box_ap_list[0], best_box_ap_list[1]))

    train_loader.reset()


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        default=None,
        type=str,
        help="Checkpoint path for resuming training.")
    parser.add_argument(
        "--loss_scale",
        default=8.,
        type=float,
        help="Mixed precision training loss scale.")
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation in train")
    parser.add_argument(
        "--output_eval",
        default=None,
        type=str,
        help="Evaluation directory, default is current directory.")
    parser.add_argument(
        "--not_quant_pattern",
        nargs='+',
        type=str,
        help="Layers which name_scope contains string in not_quant_pattern will not be quantized"
    )
    FLAGS = parser.parse_args()
    main()
