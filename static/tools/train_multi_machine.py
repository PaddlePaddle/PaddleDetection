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

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

import time
import numpy as np
import random
import datetime
import six
from collections import deque
from paddle.fluid import profiler

from paddle import fluid
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.optimizer import ExponentialMovingAverage

from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy  # new line 1
from paddle.fluid.incubate.fleet.base import role_maker  # new line 2

try:
    from ppdet.experimental import mixed_precision_context
    from ppdet.core.workspace import load_config, merge_config, create
    from ppdet.data.reader import create_reader

    from ppdet.utils import dist_utils
    from ppdet.utils.eval_utils import parse_fetches, eval_run, eval_results
    from ppdet.utils.stats import TrainingStats
    from ppdet.utils.cli import ArgsParser
    from ppdet.utils.check import check_gpu, check_version, check_config, enable_static_mode
    import ppdet.utils.checkpoint as checkpoint
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


def main():
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)  # new line 3
    fleet.init(role)  # new line 4
    env = os.environ

    num_trainers = int(env.get('PADDLE_TRAINERS_NUM', 0))
    assert num_trainers != 0, "multi-machine training process must be started using distributed.launch..."
    trainer_id = int(env.get("PADDLE_TRAINER_ID", 0))

    # set different seeds for different trainers
    random.seed(trainer_id)
    np.random.seed(trainer_id)

    if FLAGS.enable_ce:
        random.seed(0)
        np.random.seed(0)

    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()

    save_only = getattr(cfg, 'save_prediction_only', False)
    if save_only:
        raise NotImplementedError('The config file only support prediction,'
                                  ' training stage is not implemented now')
    main_arch = cfg.architecture

    assert cfg.use_gpu == True, "GPU must be supported for multi-machine training..."
    devices_num = fluid.core.get_cuda_device_count()

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
    if FLAGS.enable_ce:
        startup_prog.random_seed = 1000
        train_prog.random_seed = 1000
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            model = create(main_arch)
            if FLAGS.fp16:
                assert (getattr(model.backbone, 'norm_type', None)
                        != 'affine_channel'), \
                    '--fp16 currently does not support affine channel, ' \
                    ' please modify backbone settings to use batch norm'

            with mixed_precision_context(FLAGS.loss_scale, FLAGS.fp16) as ctx:
                inputs_def = cfg['TrainReader']['inputs_def']
                feed_vars, train_loader = model.build_inputs(**inputs_def)
                train_fetches = model.train(feed_vars)
                loss = train_fetches['loss']
                if FLAGS.fp16:
                    loss *= ctx.get_loss_scale_var()
                lr = lr_builder()
                optimizer = optim_builder(lr)

                dist_strategy = DistributedStrategy()
                sync_bn = getattr(model.backbone, 'norm_type',
                                  None) == 'sync_bn'
                dist_strategy.sync_batch_norm = sync_bn
                dist_strategy.nccl_comm_num = 1
                exec_strategy = fluid.ExecutionStrategy()
                exec_strategy.num_threads = 3
                exec_strategy.num_iteration_per_drop_scope = 30
                dist_strategy.exec_strategy = exec_strategy
                dist_strategy.fuse_all_reduce_ops = True
                optimizer = fleet.distributed_optimizer(
                    optimizer, strategy=dist_strategy)  # new line 5

                optimizer.minimize(loss)

                if FLAGS.fp16:
                    loss /= ctx.get_loss_scale_var()

            if 'use_ema' in cfg and cfg['use_ema']:
                global_steps = _decay_step_counter()
                ema = ExponentialMovingAverage(
                    cfg['ema_decay'], thres_steps=global_steps)
                ema.update()

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

        eval_reader = create_reader(cfg.EvalReader, devices_num=1)
        # When iterable mode, set set_sample_list_generator(eval_reader, place)
        eval_loader.set_sample_list_generator(eval_reader)

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

    exe.run(startup_prog)
    compiled_train_prog = fleet.main_program

    if FLAGS.eval:
        compiled_eval_prog = fluid.CompiledProgram(eval_prog)

    fuse_bn = getattr(model.backbone, 'norm_type', None) == 'affine_channel'

    ignore_params = cfg.finetune_exclude_pretrained_params \
                 if 'finetune_exclude_pretrained_params' in cfg else []

    start_iter = 0
    if FLAGS.resume_checkpoint:
        checkpoint.load_checkpoint(exe, train_prog, FLAGS.resume_checkpoint)
        start_iter = checkpoint.global_step()
    elif cfg.pretrain_weights and fuse_bn and not ignore_params:
        checkpoint.load_and_fusebn(exe, train_prog, cfg.pretrain_weights)
    elif cfg.pretrain_weights:
        checkpoint.load_params(
            exe, train_prog, cfg.pretrain_weights, ignore_params=ignore_params)

    train_reader = create_reader(
        cfg.TrainReader, (cfg.max_iters - start_iter) * devices_num,
        cfg,
        devices_num=devices_num)
    # When iterable mode, set set_sample_list_generator(train_reader, place)
    train_loader.set_sample_list_generator(train_reader)

    # whether output bbox is normalized in model output layer
    is_bbox_normalized = False
    if hasattr(model, 'is_bbox_normalized') and \
            callable(model.is_bbox_normalized):
        is_bbox_normalized = model.is_bbox_normalized()

    # if map_type not set, use default 11point, only use in VOC eval
    map_type = cfg.map_type if 'map_type' in cfg else '11point'

    train_stats = TrainingStats(cfg.log_iter, train_keys)
    train_loader.start()
    start_time = time.time()
    end_time = time.time()

    cfg_name = os.path.basename(FLAGS.config).split('.')[0]
    save_dir = os.path.join(cfg.save_dir, cfg_name)
    time_stat = deque(maxlen=cfg.log_iter)
    best_box_ap_list = [0.0, 0]  #[map, iter]

    # use VisualDL to log data
    if FLAGS.use_vdl:
        assert six.PY3, "VisualDL requires Python >= 3.5"
        from visualdl import LogWriter
        vdl_writer = LogWriter(FLAGS.vdl_log_dir)
        vdl_loss_step = 0
        vdl_mAP_step = 0

    for it in range(start_iter, cfg.max_iters):
        start_time = end_time
        end_time = time.time()
        time_stat.append(end_time - start_time)
        time_cost = np.mean(time_stat)
        eta_sec = (cfg.max_iters - it) * time_cost
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        outs = exe.run(compiled_train_prog, fetch_list=train_values)
        stats = {k: np.array(v).mean() for k, v in zip(train_keys, outs[:-1])}

        # use vdl-paddle to log loss
        if FLAGS.use_vdl:
            if it % cfg.log_iter == 0:
                for loss_name, loss_value in stats.items():
                    vdl_writer.add_scalar(loss_name, loss_value, vdl_loss_step)
                vdl_loss_step += 1

        train_stats.update(stats)
        logs = train_stats.log()
        if it % cfg.log_iter == 0 and trainer_id == 0:
            strs = 'iter: {}, lr: {:.6f}, {}, time: {:.3f}, eta: {}'.format(
                it, np.mean(outs[-1]), logs, time_cost, eta)
            logger.info(strs)

        # NOTE : profiler tools, used for benchmark
        if FLAGS.is_profiler and it == 5:
            profiler.start_profiler("All")
        elif FLAGS.is_profiler and it == 10:
            profiler.stop_profiler("total", FLAGS.profiler_path)
            return


        if (it > 0 and it % cfg.snapshot_iter == 0 or it == cfg.max_iters - 1) \
           and trainer_id == 0:
            save_name = str(it) if it != cfg.max_iters - 1 else "model_final"
            if 'use_ema' in cfg and cfg['use_ema']:
                exe.run(ema.apply_program)
            checkpoint.save(exe, train_prog, os.path.join(save_dir, save_name))

            if FLAGS.eval:
                # evaluation
                resolution = None
                if 'Mask' in cfg.architecture:
                    resolution = model.mask_head.resolution
                results = eval_run(
                    exe,
                    compiled_eval_prog,
                    eval_loader,
                    eval_keys,
                    eval_values,
                    eval_cls,
                    cfg,
                    resolution=resolution)
                box_ap_stats = eval_results(
                    results, cfg.metric, cfg.num_classes, resolution,
                    is_bbox_normalized, FLAGS.output_eval, map_type,
                    cfg['EvalReader']['dataset'])

                # use vdl_paddle to log mAP
                if FLAGS.use_vdl:
                    vdl_writer.add_scalar("mAP", box_ap_stats[0], vdl_mAP_step)
                    vdl_mAP_step += 1

                if box_ap_stats[0] > best_box_ap_list[0]:
                    best_box_ap_list[0] = box_ap_stats[0]
                    best_box_ap_list[1] = it
                    checkpoint.save(exe, train_prog,
                                    os.path.join(save_dir, "best_model"))
                logger.info("Best test box ap: {}, in iter: {}".format(
                    best_box_ap_list[0], best_box_ap_list[1]))

            if 'use_ema' in cfg and cfg['use_ema']:
                exe.run(ema.restore_program)

    train_loader.reset()


if __name__ == '__main__':
    enable_static_mode()
    parser = ArgsParser()
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        default=None,
        type=str,
        help="Checkpoint path for resuming training.")
    parser.add_argument(
        "--fp16",
        action='store_true',
        default=False,
        help="Enable mixed precision training.")
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
        "--use_vdl",
        type=bool,
        default=False,
        help="whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/scalar",
        help='VisualDL logging directory for scalar.')
    parser.add_argument(
        "--enable_ce",
        type=bool,
        default=False,
        help="If set True, enable continuous evaluation job."
        "This flag is only used for internal test.")

    #NOTE:args for profiler tools, used for benchmark
    parser.add_argument(
        '--is_profiler',
        type=int,
        default=0,
        help='The switch of profiler tools. (used for benchmark)')
    parser.add_argument(
        '--profiler_path',
        type=str,
        default="./detection.profiler",
        help='The profiler output file path. (used for benchmark)')
    FLAGS = parser.parse_args()
    main()
