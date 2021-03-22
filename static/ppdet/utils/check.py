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

import sys

import paddle
import paddle.fluid as fluid

import logging
import six
import paddle.version as fluid_version
logger = logging.getLogger(__name__)

__all__ = [
    'check_gpu',
    'check_xpu',
    'check_version',
    'check_config',
    'check_py_func',
]


def check_xpu(use_xpu):
    """
    Log error and exit when set use_xpu=true in paddlepaddle
    cpu/gpu version.
    """
    err = "Config use_xpu cannot be set as true while you are " \
          "using paddlepaddle cpu/gpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-xpu to run model on XPU \n" \
          "\t2. Set use_xpu as false in config file to run " \
          "model on CPU/GPU"

    try:
        if use_xpu and not fluid.is_compiled_with_xpu():
            logger.error(err)
            sys.exit(1)
    except Exception as e:
        pass


def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = "Config use_gpu cannot be set as true while you are " \
          "using paddlepaddle cpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
          "\t2. Set use_gpu as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and not fluid.is_compiled_with_cuda():
            logger.error(err)
            sys.exit(1)
    except Exception as e:
        pass


def check_version(version='1.7.0'):
    """
    Log error and exit when the installed version of paddlepaddle is
    not satisfied.
    """
    err = "PaddlePaddle version {} or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code.".format(version)

    version_installed = [
        fluid_version.major, fluid_version.minor, fluid_version.patch,
        fluid_version.rc
    ]
    if version_installed == ['0', '0', '0', '0']:
        return
    version_split = version.split('.')

    length = min(len(version_installed), len(version_split))
    for i in six.moves.range(length):
        if version_installed[i] > version_split[i]:
            return
        if len(version_installed[i]) == 1 and len(version_split[i]) > 1:
            return
        if version_installed[i] < version_split[i]:
            raise Exception(err)


def check_config(cfg):
    """
    Check the correctness of the configuration file. Log error and exit
    when Config is not compliant.
    """
    err = "'{}' not specified in config file. Please set it in config file."
    check_list = ['architecture', 'num_classes']
    try:
        for var in check_list:
            if not var in cfg:
                logger.error(err.format(var))
                sys.exit(1)
    except Exception as e:
        pass

    if 'log_iter' not in cfg:
        cfg.log_iter = 20

    train_dataset = cfg['TrainReader']['dataset']
    eval_dataset = cfg['EvalReader']['dataset']
    test_dataset = cfg['TestReader']['dataset']
    assert train_dataset.with_background == eval_dataset.with_background, \
        "'with_background' of TrainReader is not equal to EvalReader."
    assert train_dataset.with_background == test_dataset.with_background, \
        "'with_background' of TrainReader is not equal to TestReader."

    actual_num_classes = int(cfg.num_classes) - int(
        train_dataset.with_background)
    logger.debug("The 'num_classes'(number of classes) you set is {}, " \
                "and 'with_background' in 'dataset' sets {}.\n" \
                "So please note the actual number of categories is {}."
                .format(cfg.num_classes, train_dataset.with_background,
                    actual_num_classes))

    return cfg


def check_py_func(program):
    for block in program.blocks:
        for op in block.ops:
            if op.type == 'py_func':
                input_arg = op.input_arg_names
                output_arg = op.output_arg_names
                err = "The program contains py_func with input: {}, "\
                      "output: {}. It is not supported in Paddle inference "\
                      "engine. please replace it by paddle ops. For example, "\
                      "if you use MultiClassSoftNMS, better to replace it by "\
                      "MultiClassNMS.".format(input_arg, output_arg)
                raise Exception(err)


def enable_static_mode():
    try:
        paddle.enable_static()
    except:
        pass
