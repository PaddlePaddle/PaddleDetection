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
from paddle import fluid
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.reader import create_reader
from ppdet.utils.stats import TrainingStats
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight, save_model
import paddle.distributed as dist
import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "-weight_type",
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


def run(FLAGS, cfg):
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

    if dist.ParallelEnv().nranks > 1:
        paddle.distributed.init_parallel_env()

    # Model
    main_arch = cfg.architecture
    model = create(cfg.architecture)

    # Optimizer
    lr = create('LearningRate')()
    optimizer = create('OptimizerBuilder')(lr, model.parameters())

    # Init Model & Optimzer   
    if FLAGS.weight_type == 'resume':
        load_weight(model, cfg.pretrain_weights, optimizer)
    else:
        load_pretrain_weight(model, cfg.pretrain_weights,
                             cfg.get('load_static_weights', False),
                             FLAGS.weight_type)

    # Parallel Model 
    if dist.ParallelEnv().nranks > 1:
        model = paddle.DataParallel(model)

    # Data Reader 
    start_iter = 0
    if cfg.use_gpu:
        devices_num = fluid.core.get_cuda_device_count()
    else:
        devices_num = int(os.environ.get('CPU_NUM', 1))

    train_reader = create_reader(
        cfg.TrainDataset,
        cfg.TrainReader, (cfg.max_iters - start_iter),
        cfg,
        devices_num=devices_num)

    time_stat = deque(maxlen=cfg.log_iter)
    start_time = time.time()
    end_time = time.time()
    # Run Train 
    start_iter = optimizer.state_dict()['LR_Scheduler']['last_epoch']
    for iter_id, data in enumerate(train_reader()):
        idx = iter_id + start_iter
        start_time = end_time
        end_time = time.time()
        time_stat.append(end_time - start_time)
        time_cost = np.mean(time_stat)
        eta_sec = (cfg.max_iters - idx) * time_cost
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        # Model Forward
        model.train()
        outputs = model(data, cfg['TrainReader']['inputs_def']['fields'],
                        'train')

        # Model Backward
        loss = outputs['loss']
        if dist.ParallelEnv().nranks > 1:
            loss = model.scale_loss(loss)
            loss.backward()
            model.apply_collective_grads()
        else:
            loss.backward()
        optimizer.minimize(loss)
        optimizer.step()
        curr_lr = optimizer.get_lr()
        lr.step()
        optimizer.clear_grad()

        if dist.ParallelEnv().nranks < 2 or dist.ParallelEnv().local_rank == 0:
            # Log state 
            if idx == start_iter:
                train_stats = TrainingStats(cfg.log_iter, outputs.keys())
            train_stats.update(outputs)
            logs = train_stats.log()
            if idx % cfg.log_iter == 0:
                ips = float(cfg['TrainReader']['batch_size']) / time_cost
                strs = 'iter: {}, lr: {:.6f}, {}, eta: {}, batch_cost: {:.5f} sec, ips: {:.5f} images/sec'.format(
                    idx, curr_lr, logs, eta, time_cost, ips)
                logger.info(strs)
            # Save Stage 
            if idx > 0 and idx % int(
                    cfg.snapshot_iter) == 0 or idx == cfg.max_iters - 1:
                cfg_name = os.path.basename(FLAGS.config).split('.')[0]
                save_name = str(
                    idx) if idx != cfg.max_iters - 1 else "model_final"
                save_dir = os.path.join(cfg.save_dir, cfg_name)
                save_model(model, optimizer, save_dir, save_name)


def main():
    FLAGS = parse_args()

    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    run(FLAGS, cfg)


if __name__ == "__main__":
    main()
