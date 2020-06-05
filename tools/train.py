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


def main(FLAGS):
    env = os.environ
    FLAGS.dist = 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env
    if FLAGS.dist:
        trainer_id = int(env['PADDLE_TRAINER_ID'])
        local_seed = (99 + trainer_id)
        random.seed(local_seed)
        np.random.seed(local_seed)

    if FLAGS.enable_ce:
        random.seed(0)

    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()

    if FLAGS.use_gpu:
        devices_num = fluid.core.get_cuda_device_count(
        ) if FLAGS.use_parallel else 1
    else:
        devices_num = int(os.environ.get('CPU_NUM', 1))

    # Model
    main_arch = cfg.architecture
    model = create(cfg.architecture)

    # Optimizer
    lr = create('LearningRate')()
    optimizer = create('OptimizerBuilder')(lr, model.parameters())

    if FLAGS.use_parallel:
        strategy = fluid.dygraph.parallel.prepare_context()
        model = fluid.dygraph.parallel.DataParallel(model, strategy)

    if False:
        model_state = model.state_dict()
        w_dict = np.load(cfg.pretrained_model)
        for k, v in w_dict.items():
            for wk in model_state.keys():
                res = re.search(k, wk)
                if res is not None:
                    print("load: ", k, v.shape, np.mean(np.abs(v)), " --> ", wk,
                          model_state[wk].shape)
                    model_state[wk] = v
                    break
        model.set_dict(model_state)

    elif False:
        para_state_dict, _ = fluid.load_dygraph(cfg.resume_model)
        model.set_dict(para_state_dict)
        new_dict = {}
        for k, v in para_state_dict.items():
            if "conv2d" in k:
                new_k = k.split('.')[1]
            elif 'linear' in k:
                new_k = k.split('.')[1]
            elif 'conv2dtranspose' in k:
                new_k = k.split('.')[1]
            else:
                new_k = k
            new_dict[new_k] = v.numpy()
        np.savez("rcnn_dyg.npz", **new_dict)

    start_iter = 0
    train_reader = create_reader(
        cfg.TrainReader, (cfg.max_iters - start_iter) * devices_num,
        cfg,
        devices_num=devices_num)

    for iter_id, data in enumerate(train_reader()):
        start_time = time.time()
        #for batch in data:
        #    print(data)
        # forward
        outputs = model(data)

        # backward
        loss = outputs['loss']
        if FLAGS.use_parallel:
            loss = model.scale_loss(loss)
            loss.backward()
            model.apply_collective_grads()
        else:
            loss.backward()
        optimizer.minimize(loss)
        model.clear_gradients()

        cost_time = time.time() - start_time

        print("iter: {}, time: {}, loss: {}".format(iter_id, cost_time,
                                                    loss.numpy()[0]))


if __name__ == '__main__':
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

    #NOTE:args for profiler tools, used for benchmark
    parser.add_argument(
        '--is_profiler',
        type=int,
        default=0,
        help='The switch of profiler tools. (used for benchmark)')
    FLAGS = parser.parse_args()
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
                    if FLAGS.use_parallel else fluid.CUDAPlace(0) \
                    if FLAGS.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        main(FLAGS)
