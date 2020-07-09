from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
# ignore numba warning
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import paddle.fluid as fluid
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.reader import create_reader
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.utils.checkpoint import load_dygraph_ckpt, save_dygraph_ckpt


def parse_args():
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

    if FLAGS.enable_ce or cfg.open_debug:
        fluid.default_startup_program().random_seed = 1000
        fluid.default_main_program().random_seed = 1000
        random.seed(0)
        np.random.seed(0)

    if FLAGS.use_parallel:
        strategy = fluid.dygraph.parallel.prepare_context()
        parallel_log = "Note: use parallel "

    # Model
    main_arch = cfg.architecture
    model = create(
        cfg.architecture,
        mode='train',
        open_debug=cfg.open_debug,
        debug_names=[
            'im_id', 'image', 'im_info', 'gt_bbox', 'res2', 'res3', 'res4',
            'rpn_feat', 'rpn_rois_score', 'rpn_rois_delta', 'rpn_rois', {
                'proposal_0': 'rois'
            }, {
                'bbox_head_0':
                ['rois_feat', 'bbox_feat', 'bbox_score', 'bbox_delta']
            }
        ])

    # Init Model  
    if os.path.isfile(cfg.pretrain_weights):
        model = load_dygraph_ckpt(
            model,
            pretrain_ckpt=cfg.pretrain_weights,
            open_debug=cfg.open_debug)

    # Parallel Model 
    if FLAGS.use_parallel:
        #strategy = fluid.dygraph.parallel.prepare_context()
        model = fluid.dygraph.parallel.DataParallel(model, strategy)
        parallel_log += "with data parallel!"
        print(parallel_log)

    # Optimizer
    lr = create('LearningRate')()
    optimizer = create('OptimizerBuilder')(lr, model.parameters())

    # Data Reader 
    start_iter = 0
    if cfg.use_gpu:
        devices_num = fluid.core.get_cuda_device_count(
        ) if FLAGS.use_parallel else 1
    else:
        devices_num = int(os.environ.get('CPU_NUM', 1))

    train_reader = create_reader(
        cfg.TrainReader, (cfg.max_iters - start_iter) * devices_num,
        cfg,
        devices_num=devices_num)

    # Run Train 
    for iter_id, data in enumerate(train_reader()):
        start_time = time.time()

        # Model Forward
        model.train()
        outputs = model(data, cfg['TrainReader']['inputs_def']['fields'])

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

        # Log state 
        cost_time = time.time() - start_time
        # TODO: check this method   
        curr_lr = optimizer.current_step_lr()
        log_info = "iter: {}, time: {:.4f}, lr: {:.6f}".format(
            iter_id, cost_time, curr_lr)
        for k, v in outputs.items():
            log_info += ", {}: {:.6f}".format(k, v.numpy()[0])
        print(log_info)

        # Debug 
        if cfg.open_debug and iter_id > 10:
            break

        # Save Stage 
        if iter_id > 0 and iter_id % int(cfg.snapshot_iter) == 0:
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
                    if FLAGS.use_parallel else fluid.CUDAPlace(0) \
                    if cfg.use_gpu else fluid.CPUPlace()

    with fluid.dygraph.guard(place):
        run(FLAGS, cfg)


if __name__ == "__main__":
    main()
