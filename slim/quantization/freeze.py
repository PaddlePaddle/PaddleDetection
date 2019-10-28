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
import datetime
from collections import deque
import sys
sys.path.append("../../")
from paddle.fluid.contrib.slim import Compressor
from paddle.fluid.framework import IrGraph
from paddle.fluid import core
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass
from paddle.fluid.contrib.slim.quantization import ConvertToInt8Pass
from paddle.fluid.contrib.slim.quantization import TransformForMobilePass


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
from ppdet.utils.stats import TrainingStats
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu
import ppdet.utils.checkpoint as checkpoint
from ppdet.modeling.model_input import create_feed

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def eval_run(exe, compile_program, reader, keys, values, cls, test_feed):
    """
    Run evaluation program, return program outputs.
    """
    iter_id = 0
    results = []

    images_num = 0
    start_time = time.time()
    has_bbox = 'bbox' in keys
    for data in reader():
        data = test_feed.feed(data)
        feed_data = {'image': data['image'], 'im_size': data['im_size']}
        outs = exe.run(compile_program,
                       feed=feed_data,
                       fetch_list=values[0],
                       return_numpy=False)
        outs.append(data['gt_box'])
        outs.append(data['gt_label'])
        outs.append(data['is_difficult'])
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

    if 'eval_feed' not in cfg:
        eval_feed = create(main_arch + 'EvalFeed')
    else:
        eval_feed = create(cfg.eval_feed)

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    _, test_feed_vars = create_feed(eval_feed, False)

    eval_reader = create_reader(eval_feed, args_path=FLAGS.dataset_dir)
    #eval_pyreader.decorate_sample_list_generator(eval_reader, place)
    test_data_feed = fluid.DataFeeder(test_feed_vars.values(), place)

    assert os.path.exists(FLAGS.model_path)
    infer_prog, feed_names, fetch_targets = fluid.io.load_inference_model(
        dirname=FLAGS.model_path,
        executor=exe,
        model_filename='__model__.infer',
        params_filename='__params__')

    eval_keys = ['bbox', 'gt_box', 'gt_label', 'is_difficult']
    eval_values = [
        'multiclass_nms_0.tmp_0', 'gt_box', 'gt_label', 'is_difficult'
    ]
    eval_cls = []
    eval_values[0] = fetch_targets[0]

    results = eval_run(exe, infer_prog, eval_reader, eval_keys, eval_values,
                       eval_cls, test_data_feed)

    resolution = None
    if 'mask' in results[0]:
        resolution = model.mask_head.resolution
    box_ap_stats = eval_results(results, eval_feed, cfg.metric, cfg.num_classes,
                                resolution, False, FLAGS.output_eval)

    logger.info("freeze the graph for inference")
    test_graph = IrGraph(core.Graph(infer_prog.desc), for_test=True)

    freeze_pass = QuantizationFreezePass(
        scope=fluid.global_scope(),
        place=place,
        weight_quantize_type=FLAGS.weight_quant_type)
    freeze_pass.apply(test_graph)
    server_program = test_graph.to_program()
    fluid.io.save_inference_model(
        dirname=os.path.join(FLAGS.save_path, 'float'),
        feeded_var_names=feed_names,
        target_vars=fetch_targets,
        executor=exe,
        main_program=server_program,
        model_filename='model',
        params_filename='weights')

    logger.info("convert the weights into int8 type")
    convert_int8_pass = ConvertToInt8Pass(
        scope=fluid.global_scope(), place=place)
    convert_int8_pass.apply(test_graph)
    server_int8_program = test_graph.to_program()
    fluid.io.save_inference_model(
        dirname=os.path.join(FLAGS.save_path, 'int8'),
        feeded_var_names=feed_names,
        target_vars=fetch_targets,
        executor=exe,
        main_program=server_int8_program,
        model_filename='model',
        params_filename='weights')

    logger.info("convert the freezed pass to paddle-lite execution")
    mobile_pass = TransformForMobilePass()
    mobile_pass.apply(test_graph)
    mobile_program = test_graph.to_program()
    fluid.io.save_inference_model(
        dirname=os.path.join(FLAGS.save_path, 'mobile'),
        feeded_var_names=feed_names,
        target_vars=fetch_targets,
        executor=exe,
        main_program=mobile_program,
        model_filename='model',
        params_filename='weights')


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "-m", "--model_path", default=None, type=str, help="path of checkpoint")
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
    parser.add_argument(
        "--weight_quant_type",
        default='abs_max',
        type=str,
        help="quantization type for weight")
    parser.add_argument(
        "--save_path",
        default='./output',
        type=str,
        help="path to save quantization inference model")

    FLAGS = parser.parse_args()
    main()
