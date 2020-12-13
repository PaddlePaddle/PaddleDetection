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

import time
# ignore numba warning
import warnings
warnings.filterwarnings('ignore')
import random
import datetime
import numpy as np
from collections import deque
import paddle
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.stats import TrainingStats
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight, save_model
from export_model import dygraph_to_static
from paddle.distributed import ParallelEnv
import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgsParser()
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
        "--use_tb",
        type=bool,
        default=False,
        help="whether to record the data to Tensorboard.")
    parser.add_argument(
        '--tb_log_dir',
        type=str,
        default="tb_log_dir/scalar",
        help='Tensorboard logging directory for scalar.')
    parser.add_argument(
        "--enable_ce",
        type=bool,
        default=False,
        help="If set True, enable continuous evaluation job."
        "This flag is only used for internal test.")
    parser.add_argument(
        "--use_gpu", action='store_true', default=False, help="data parallel")

    parser.add_argument(
        '--is_profiler',
        type=int,
        default=0,
        help='The switch of profiler tools. (used for benchmark)')

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
    dataset = cfg.TrainDataset
    train_loader, step_per_epoch = create('TrainReader')(
        dataset, cfg['worker_num'], place)

    # Model
    model = create(cfg.architecture)

    # Optimizer
    lr = create('LearningRate')(step_per_epoch)
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

    fields = train_loader.collate_fn.output_fields
    cfg_name = os.path.basename(FLAGS.config).split('.')[0]
    save_dir = os.path.join(cfg.save_dir, cfg_name)
    # Run Train
    time_stat = deque(maxlen=cfg.log_iter)
    start_time = time.time()
    end_time = time.time()
    # Run Train
    for cur_eid in range(start_epoch, int(cfg.epoch)):
        train_loader.dataset.epoch = cur_eid
        for iter_id, data in enumerate(train_loader):
            start_time = end_time
            end_time = time.time()
            time_stat.append(end_time - start_time)
            time_cost = np.mean(time_stat)
            eta_sec = (
                (cfg.epoch - cur_eid) * step_per_epoch - iter_id) * time_cost
            eta = str(datetime.timedelta(seconds=int(eta_sec)))

            # Model Forward
            model.train()
            outputs = model(data=data, input_def=fields, mode='train')

            # Model Backward
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            curr_lr = optimizer.get_lr()
            lr.step()
            optimizer.clear_grad()

            if ParallelEnv().nranks < 2 or ParallelEnv().local_rank == 0:
                # Log state 
                if cur_eid == start_epoch and iter_id == 0:
                    train_stats = TrainingStats(cfg.log_iter, outputs.keys())
                train_stats.update(outputs)
                logs = train_stats.log()
                if iter_id % cfg.log_iter == 0:
                    ips = float(cfg['TrainReader']['batch_size']) / time_cost
                    strs = 'Epoch:{}: iter: {}, lr: {:.6f}, {}, eta: {}, batch_cost: {:.5f} sec, ips: {:.5f} images/sec'.format(
                        cur_eid, iter_id, curr_lr, logs, eta, time_cost, ips)
                    logger.info(strs)

        # Save Stage 
        if ParallelEnv().local_rank == 0 and (
                cur_eid % cfg.snapshot_epoch == 0 or
            (cur_eid + 1) == int(cfg.epoch)):
            save_name = str(cur_eid) if cur_eid + 1 != int(
                cfg.epoch) else "model_final"
            save_model(model, optimizer, save_dir, save_name, cur_eid + 1)
        # TODO(guanghua): dygraph model to static model
        # if ParallelEnv().local_rank == 0 and (cur_eid + 1) == int(cfg.epoch)):
        #     dygraph_to_static(model, os.path.join(save_dir, 'static_model_final'), cfg)


def main():
    FLAGS = parse_args()

    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    place = 'gpu:{}'.format(ParallelEnv().dev_id) if cfg.use_gpu else 'cpu'
    place = paddle.set_device(place)

    run(FLAGS, cfg, place)


if __name__ == "__main__":
    main()
