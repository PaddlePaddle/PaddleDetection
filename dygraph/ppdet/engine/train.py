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
import time
import random
import datetime

import paddle
from paddle.distributed import ParallelEnv

from ppdet.utils.checkpoint import save_model
from ppdet.core.workspace import create
import ppdet.utils.stats as stats

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['init_parallel_env', 'set_random_seed', 'train_detector']


def init_parallel_env():
    env = os.environ
    dist = 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env
    if dist:
        trainer_id = int(env['PADDLE_TRAINER_ID'])
        local_seed = (99 + trainer_id)
        random.seed(local_seed)
        np.random.seed(local_seed)

    if ParallelEnv().nranks > 1:
        paddle.distributed.init_parallel_env()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def train_detector(model, loader, cfg, start_epoch=0, validate=False):
    # FIXME: add validation in training
    assert not validate, "validation in training not supported currently"

    # build optimizer
    steps = len(loader)
    lr = create('LearningRate')(steps)
    optimizer = create('OptimizerBuilder')(lr, model.parameters())

    # sync_bn can only be set nranks > 1
    if getattr(model.backbone, 'norm_type', None) == 'sync_bn':
        assert cfg.use_gpu and ParallelEnv(
        ).nranks > 1, 'you should use bn rather than sync_bn while using a single gpu'

    # Parallel Model 
    if ParallelEnv().nranks > 1:
        model = paddle.DataParallel(model)

    save_dir = os.path.join(cfg.save_dir, cfg.filename)

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
        loader.dataset.set_epoch(cur_eid)
        for iter_id, data in enumerate(loader):
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
