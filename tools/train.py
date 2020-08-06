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
import paddle.fluid as fluid
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.reader import create_reader
from ppdet.utils.stats import TrainingStats
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.utils.checkpoint import load_dygraph_ckpt, save_dygraph_ckpt
import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "-ckpt_type",
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
        "--use_parallel",
        action='store_true',
        default=False,
        help="data parallel")

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

    # Model
    main_arch = cfg.architecture
    model = create(cfg.architecture)

    # Optimizer
    lr = create('LearningRate')()
    optimizer = create('OptimizerBuilder')(lr, model.parameters())

    # Init Model & Optimzer   
    model = load_dygraph_ckpt(
        model,
        optimizer,
        cfg.pretrain_weights,
        ckpt_type=FLAGS.ckpt_type,
        load_static_weights=cfg.load_static_weights)

    # Parallel Model 
    if FLAGS.use_parallel:
        strategy = fluid.dygraph.parallel.prepare_context()
        model = fluid.dygraph.parallel.DataParallel(model, strategy)

    # Data Reader 
    start_iter = 0
    if cfg.use_gpu:
        devices_num = fluid.core.get_cuda_device_count()
    else:
        devices_num = int(os.environ.get('CPU_NUM', 1))

    train_reader = create_reader(
        cfg.TrainReader, (cfg.max_iters - start_iter) * devices_num,
        cfg,
        devices_num=devices_num)

    time_stat = deque(maxlen=cfg.log_smooth_window)
    start_time = time.time()
    end_time = time.time()
    # Run Train 
    for iter_id, data in enumerate(train_reader()):

        start_time = end_time
        end_time = time.time()
        time_stat.append(end_time - start_time)
        time_cost = np.mean(time_stat)
        eta_sec = (cfg.max_iters - iter_id) * time_cost
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        # Model Forward
        model.train()
        outputs = model(data, cfg['TrainReader']['inputs_def']['fields'],
                        'train')

        # Model Backward
        loss = outputs['loss']
        if FLAGS.use_parallel:
            loss = model.scale_loss(loss)
            loss.backward()
            model.apply_collective_grads()
        else:
            loss.backward()
        optimizer.minimize(loss)
        model.clear_gradients()
        curr_lr = optimizer.current_step_lr()

        # Log state 
        if iter_id == 0:
            train_stats = TrainingStats(cfg.log_smooth_window, outputs.keys())
        train_stats.update(outputs)
        logs = train_stats.log()
        if iter_id % cfg.log_iter == 0:
            strs = 'iter: {}, lr: {:.6f}, {}, time: {:.3f}, eta: {}'.format(
                iter_id, curr_lr, logs, time_cost, eta)
            logger.info(strs)
        # Save Stage 
        if iter_id > 0 and iter_id % int(
                cfg.snapshot_iter) == 0 and fluid.dygraph.parallel.Env(
                ).local_rank == 0:
            cfg_name = os.path.basename(FLAGS.config).split('.')[0]
            save_name = str(
                iter_id) if iter_id != cfg.max_iters - 1 else "model_final"
            save_dir = os.path.join(cfg.save_dir, cfg_name, save_name)
            save_dygraph_ckpt(model, optimizer, save_dir)


def main():
    FLAGS = parse_args()

    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
                    if cfg.use_gpu else fluid.CPUPlace()

    with fluid.dygraph.guard(place):
        run(FLAGS, cfg)


if __name__ == "__main__":
    main()
