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

import os, sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

# ignore numba warning
import warnings
warnings.filterwarnings('ignore')
import random
import datetime
import time
import numpy as np

import paddle
from paddle.distributed import ParallelEnv

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight, save_model

import ppdet.utils.cli as cli
import ppdet.utils.check as check
import ppdet.utils.stats as stats
from ppdet.utils.logger import setup_logger
logger = setup_logger('train')


def parse_args():
    parser = cli.ArgsParser()
    parser.add_argument(
        "--weight_type",
        default='pretrain',
        type=str,
        help="Loading Checkpoints only support 'pretrain', 'finetune', 'resume'."
    )
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
        "--enable_ce",
        type=bool,
        default=False,
        help="If set True, enable continuous evaluation job."
        "This flag is only used for internal test.")
    parser.add_argument(
        "--use_gpu", action='store_true', default=False, help="data parallel")
    args = parser.parse_args()
    return args


def run(FLAGS, cfg, place):
    env = os.environ
    FLAGS.dist = 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env
    if FLAGS.dist:
        trainer_id = int(env['PADDLE_TRAINER_ID'])
        local_seed = (99 + trainer_id)
        random.seed(local_seed)
        np.random.seed(local_seed)

    if FLAGS.enable_ce:
        random.seed(0)
        np.random.seed(0)

    if ParallelEnv().nranks > 1:
        paddle.distributed.init_parallel_env()

    # Data 
    datasets = cfg.TrainDataset
    train_loader = create('TrainReader')(datasets, cfg['worker_num'])
    steps = len(train_loader)

    # Model
    model = create(cfg.architecture)

    # Optimizer
    lr = create('LearningRate')(steps)
    optimizer = create('OptimizerBuilder')(lr, model.parameters())

    # Init Model & Optimzer   
    start_epoch = 0
    if FLAGS.weight_type == 'resume':
        start_epoch = load_weight(model, cfg.pretrain_weights, optimizer)
    else:
        load_pretrain_weight(model, cfg.pretrain_weights,
                             cfg.get('load_static_weights', False),
                             FLAGS.weight_type)

    if getattr(model.backbone, 'norm_type', None) == 'sync_bn':
        assert cfg.use_gpu and ParallelEnv(
        ).nranks > 1, 'you should use bn rather than sync_bn while using a single gpu'
    # sync_bn = (getattr(model.backbone, 'norm_type', None) == 'sync_bn' and
    #            cfg.use_gpu and ParallelEnv().nranks > 1)
    # if sync_bn:
    #     model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Parallel Model 
    if ParallelEnv().nranks > 1:
        model = paddle.DataParallel(model)

    cfg_name = os.path.basename(FLAGS.config).split('.')[0]
    save_dir = os.path.join(cfg.save_dir, cfg_name)

    # Run Train
    end_epoch = int(cfg.epoch)
    batch_size = int(cfg['TrainReader']['batch_size'])
    total_steps = (end_epoch - start_epoch) * steps
    step_id = 0

    train_stats = stats.TrainingStats(cfg.log_iter)
    batch_time = stats.SmoothedValue(fmt='{avg:.4f}')
    data_time = stats.SmoothedValue(fmt='{avg:.4f}')

    end_time = time.time()
    space_fmt = ':' + str(len(str(steps))) + 'd'
    # Run Train
    for cur_eid in range(start_epoch, end_epoch):
        datasets.set_epoch(cur_eid)
        for iter_id, data in enumerate(train_loader):
            data_time.update(time.time() - end_time)
            # Model Forward
            model.train()
            outputs = model(data, mode='train')
            loss = outputs['loss']
            # Model Backward
            loss.backward()
            optimizer.step()
            curr_lr = optimizer.get_lr()
            lr.step()
            optimizer.clear_grad()

            batch_time.update(time.time() - end_time)
            if ParallelEnv().nranks < 2 or ParallelEnv().local_rank == 0:
                train_stats.update(outputs)
                logs = train_stats.log()
                if iter_id % cfg.log_iter == 0:
                    eta_sec = (total_steps - step_id) * batch_time.global_avg
                    eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                    ips = float(batch_size) / batch_time.avg
                    fmt = ' '.join([
                        'Epoch: [{}]',
                        '[{' + space_fmt + '}/{}]',
                        '{meters}',
                        'eta: {eta}',
                        'batch_cost: {btime}',
                        'data_cost: {dtime}',
                        'ips: {ips:.4f} images/s',
                    ])
                    fmt = fmt.format(
                        cur_eid,
                        iter_id,
                        steps,
                        meters=logs,
                        eta=eta_str,
                        btime=str(batch_time),
                        dtime=str(data_time),
                        ips=ips)
                    logger.info(fmt)
            step_id += 1
            end_time = time.time()  # after copy outputs to CPU.
        # Save Stage 
        if (ParallelEnv().local_rank == 0 and \
            (cur_eid % cfg.snapshot_epoch) == 0) or (cur_eid + 1) == end_epoch:
            save_name = str(
                cur_eid) if cur_eid + 1 != end_epoch else "model_final"
            save_model(model, optimizer, save_dir, save_name, cur_eid + 1)


def main():
    FLAGS = parse_args()

    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)
    check.check_version()

    place = 'gpu:{}'.format(ParallelEnv().dev_id) if cfg.use_gpu else 'cpu'
    place = paddle.set_device(place)

    run(FLAGS, cfg, place)


if __name__ == "__main__":
    main()
