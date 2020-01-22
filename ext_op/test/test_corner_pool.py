#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid as fluid
import cornerpool_lib


def bottom_pool_np(x):
    height = x.shape[2]
    output = x.copy()
    for ind in range(height):
        cur = output[:, :, ind:height, :]
        next = output[:, :, :height - ind, :]
        output[:, :, ind:height, :] = np.maximum(cur, next)
    return output


def top_pool_np(x):
    height = x.shape[2]
    output = x.copy()
    for ind in range(height):
        cur = output[:, :, :height - ind, :]
        next = output[:, :, ind:height, :]
        output[:, :, :height - ind, :] = np.maximum(cur, next)
    return output


def right_pool_np(x):
    width = x.shape[3]
    output = x.copy()
    for ind in range(width):
        cur = output[:, :, :, ind:width]
        next = output[:, :, :, :width - ind]
        output[:, :, :, ind:width] = np.maximum(cur, next)
    return output


def left_pool_np(x):
    width = x.shape[3]
    output = x.copy()
    for ind in range(width):
        cur = output[:, :, :, :width - ind]
        next = output[:, :, :, ind:width]
        output[:, :, :, :width - ind] = np.maximum(cur, next)
    return output


class TestRightPoolOp(unittest.TestCase):
    def funcmap(self):
        self.func_map = {
            'bottom_x': [cornerpool_lib.bottom_pool, bottom_pool_np],
            'top_x': [cornerpool_lib.top_pool, top_pool_np],
            'right_x': [cornerpool_lib.right_pool, right_pool_np],
            'left_x': [cornerpool_lib.left_pool, left_pool_np]
        }

    def setup(self):
        self.name = 'right_x'

    def test_check_output(self):
        self.funcmap()
        self.setup()
        x_shape = (2, 3, 6, 6)
        x_type = "float64"

        sp = fluid.Program()
        tp = fluid.Program()

        with fluid.program_guard(tp, sp):
            x = fluid.layers.data(
                name=self.name,
                shape=x_shape,
                dtype=x_type,
                append_batch_size=False)
            y = self.func_map[self.name][0](x)

        np.random.seed(0)
        x_np = np.random.uniform(-10, 10, x_shape).astype(x_type)
        out_np = self.func_map[self.name][1](x_np)

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        outs, = exe.run(tp, feed={self.name: x_np}, fetch_list=[y])

        self.assertTrue(np.allclose(outs, out_np))


class TestTopPoolOp(TestRightPoolOp):
    def setup(self):
        self.name = 'top_x'


class TestBottomPoolOp(TestRightPoolOp):
    def setup(self):
        self.name = 'bottom_x'


class TestLeftPoolOp(TestRightPoolOp):
    def setup(self):
        self.name = 'left_x'


if __name__ == "__main__":
    unittest.main()
