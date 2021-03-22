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

import os

import paddle.fluid as fluid


def nccl2_prepare(trainer_id, startup_prog, main_prog):
    config = fluid.DistributeTranspilerConfig()
    config.mode = "nccl2"
    t = fluid.DistributeTranspiler(config=config)
    t.transpile(
        trainer_id,
        trainers=os.environ.get('PADDLE_TRAINER_ENDPOINTS'),
        current_endpoint=os.environ.get('PADDLE_CURRENT_ENDPOINT'),
        startup_program=startup_prog,
        program=main_prog)


def prepare_for_multi_process(exe, build_strategy, startup_prog, main_prog):
    trainer_id = int(os.environ.get('PADDLE_TRAINER_ID', 0))
    num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    if num_trainers < 2:
        return
    build_strategy.num_trainers = num_trainers
    build_strategy.trainer_id = trainer_id
    nccl2_prepare(trainer_id, startup_prog, main_prog)
