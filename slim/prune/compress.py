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
import sys
sys.path.append("../../")
from paddle.fluid.contrib.slim import Compressor


def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be set before
# `import paddle`. Otherwise, it would not take any effect.
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

from paddle import fluid
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.data_feed import create_reader
from ppdet.utils.eval_utils import parse_fetches, eval_results
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu
import ppdet.utils.checkpoint as checkpoint
from ppdet.modeling.model_input import create_feed

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def eval_run(exe, compile_program, reader, keys, values, cls, test_feed, cfg):
    """
    Run evaluation program, return program outputs.
    """
    iter_id = 0
    results = []
    if len(cls) != 0:
        values = []
        for i in range(len(cls)):
            _, accum_map = cls[i].get_map_var()
            cls[i].reset(exe)
            values.append(accum_map)

    images_num = 0
    start_time = time.time()
    has_bbox = 'bbox' in keys
    for data in reader():
        data = test_feed.feed(data)
        feed_data = {'image': data['image'], 'im_size': data['im_size']}
        outs = exe.run(compile_program,
                       feed=feed_data,
                       fetch_list=[values[0]],
                       return_numpy=False)

        if cfg.metric == 'VOC':
            outs.append(data['gt_box'])
            outs.append(data['gt_label'])
            outs.append(data['is_difficult'])
        elif cfg.metric == 'COCO':
            outs.append(data['im_info'])
            outs.append(data['im_id'])
            outs.append(data['im_shape'])

        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(keys, outs)
        }
        results.append(res)
        if iter_id % 100 == 0:
            logger.info('Test iter {}'.format(iter_id))
        iter_id += 1
        images_num += len(res['bbox'][1][0]) if has_bbox else 1
    logger.info('Test finish iter {}'.format(iter_id))

    end_time = time.time()
    fps = images_num / (end_time - start_time)
    if has_bbox:
        logger.info('Total number of images: {}, inference time: {} fps.'.
                    format(images_num, fps))
    else:
        logger.info('Total iteration: {}, inference time: {} batch/s.'.format(
            images_num, fps))

    return results


def main():
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

    if cfg.use_gpu:
        devices_num = fluid.core.get_cuda_device_count()
    else:
        devices_num = int(
            os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    if 'train_feed' not in cfg:
        train_feed = create(main_arch + 'TrainFeed')
    else:
        train_feed = create(cfg.train_feed)

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
            _, feed_vars = create_feed(train_feed, True)
            train_fetches = model.train(feed_vars)
            loss = train_fetches['loss']
            lr = lr_builder()
            optimizer = optim_builder(lr)
            optimizer.minimize(loss)

    train_reader = create_reader(train_feed, cfg.max_iters, FLAGS.dataset_dir)

    # parse train fetches
    train_keys, train_values, _ = parse_fetches(train_fetches)
    train_keys.append("lr")
    train_values.append(lr.name)

    train_fetch_list = []
    for k, v in zip(train_keys, train_values):
        train_fetch_list.append((k, v))

    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            model = create(main_arch)
            _, test_feed_vars = create_feed(eval_feed, True)
            fetches = model.eval(test_feed_vars)

    eval_prog = eval_prog.clone(True)

    eval_reader = create_reader(eval_feed, args_path=FLAGS.dataset_dir)
    test_data_feed = fluid.DataFeeder(test_feed_vars.values(), place)

    # parse eval fetches
    extra_keys = []
    if cfg.metric == 'COCO':
        extra_keys = ['im_info', 'im_id', 'im_shape']
    if cfg.metric == 'VOC':
        extra_keys = ['gt_box', 'gt_label', 'is_difficult']
    eval_keys, eval_values, eval_cls = parse_fetches(fetches, eval_prog,
                                                     extra_keys)
    eval_fetch_list = []
    for k, v in zip(eval_keys, eval_values):
        eval_fetch_list.append((k, v))

    exe.run(startup_prog)
    checkpoint.load_params(exe, train_prog, cfg.pretrain_weights)

    best_box_ap_list = []

    def eval_func(program, scope):

        #place = fluid.CPUPlace()
        #exe = fluid.Executor(place)
        results = eval_run(exe, program, eval_reader, eval_keys, eval_values,
                           eval_cls, test_data_feed, cfg)

        resolution = None
        if 'mask' in results[0]:
            resolution = model.mask_head.resolution
        box_ap_stats = eval_results(results, eval_feed, cfg.metric,
                                    cfg.num_classes, resolution, False,
                                    FLAGS.output_eval)
        if len(best_box_ap_list) == 0:
            best_box_ap_list.append(box_ap_stats[0])
        elif box_ap_stats[0] > best_box_ap_list[0]:
            best_box_ap_list[0] = box_ap_stats[0]
        logger.info("Best test box ap: {}".format(best_box_ap_list[0]))
        return best_box_ap_list[0]

    test_feed = [('image', test_feed_vars['image'].name),
                 ('im_size', test_feed_vars['im_size'].name)]

    com = Compressor(
        place,
        fluid.global_scope(),
        train_prog,
        train_reader=train_reader,
        train_feed_list=[(key, value.name) for key, value in feed_vars.items()],
        train_fetch_list=train_fetch_list,
        eval_program=eval_prog,
        eval_reader=eval_reader,
        eval_feed_list=test_feed,
        eval_func={'map': eval_func},
        eval_fetch_list=[eval_fetch_list[0]],
        save_eval_model=True,
        prune_infer_model=[["image", "im_size"], ["multiclass_nms_0.tmp_0"]],
        train_optimizer=None)
    com.config(FLAGS.slim_file)
    com.run()


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "-s",
        "--slim_file",
        default=None,
        type=str,
        help="Config file of PaddleSlim.")
    parser.add_argument(
        "--output_eval",
        default=None,
        type=str,
        help="Evaluation directory, default is current directory.")
    parser.add_argument(
        "-d",
        "--dataset_dir",
        default=None,
        type=str,
        help="Dataset path, same as DataFeed.dataset.dataset_dir")
    FLAGS = parser.parse_args()
    main()
