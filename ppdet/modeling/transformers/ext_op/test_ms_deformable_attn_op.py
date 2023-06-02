# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function
from __future__ import division

import os
import sys
import random
import numpy as np
import paddle
# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 5)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from ppdet.modeling.transformers.utils import deformable_attention_core_func
ms_deform_attn_core_paddle = deformable_attention_core_func

try:
    gpu_index = int(sys.argv[1])
except:
    gpu_index = 0
print(f'Use gpu {gpu_index} to test...')
paddle.set_device(f'gpu:{gpu_index}')

try:
    from deformable_detr_ops import ms_deformable_attn
except Exception as e:
    print('import deformable_detr_ops error', e)
    sys.exit(-1)

paddle.seed(1)
random.seed(1)
np.random.seed(1)

bs, n_heads, c = 2, 8, 8
query_length, n_levels, n_points = 2, 2, 2
spatial_shapes = paddle.to_tensor([(6, 4), (3, 2)], dtype=paddle.int64)
level_start_index = paddle.concat((paddle.to_tensor(
    [0], dtype=paddle.int64), spatial_shapes.prod(1).cumsum(0)[:-1]))
value_length = sum([(H * W).item() for H, W in spatial_shapes])


def get_test_tensors(channels):
    value = paddle.rand(
        [bs, value_length, n_heads, channels], dtype=paddle.float32) * 0.01
    sampling_locations = paddle.rand(
        [bs, query_length, n_heads, n_levels, n_points, 2],
        dtype=paddle.float32)
    attention_weights = paddle.rand(
        [bs, query_length, n_heads, n_levels, n_points],
        dtype=paddle.float32) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(
        -2, keepdim=True)

    return [value, sampling_locations, attention_weights]


@paddle.no_grad()
def check_forward_equal_with_paddle_float():
    value, sampling_locations, attention_weights = get_test_tensors(c)

    output_paddle = ms_deform_attn_core_paddle(
        value, spatial_shapes, level_start_index, sampling_locations,
        attention_weights).detach().cpu()
    output_cuda = ms_deformable_attn(value, spatial_shapes, level_start_index,
                                     sampling_locations,
                                     attention_weights).detach().cpu()
    fwdok = paddle.allclose(
        output_cuda, output_paddle, rtol=1e-2, atol=1e-3).item()
    max_abs_err = (output_cuda - output_paddle).abs().max().item()
    max_rel_err = (
        (output_cuda - output_paddle).abs() / output_paddle.abs()).max().item()

    print(
        f'*{fwdok} check_forward_equal_with_paddle_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}'
    )


def check_gradient_numerical(channels=4):
    value_paddle, sampling_locations_paddle, attention_weights_paddle = get_test_tensors(
        channels)
    value_paddle.stop_gradient = False
    sampling_locations_paddle.stop_gradient = False
    attention_weights_paddle.stop_gradient = False

    value_cuda = value_paddle.detach().clone()
    sampling_locations_cuda = sampling_locations_paddle.detach().clone()
    attention_weights_cuda = attention_weights_paddle.detach().clone()
    value_cuda.stop_gradient = False
    sampling_locations_cuda.stop_gradient = False
    attention_weights_cuda.stop_gradient = False

    output_paddle = ms_deform_attn_core_paddle(
        value_paddle, spatial_shapes, level_start_index,
        sampling_locations_paddle, attention_weights_paddle)
    output_paddle.sum().backward()

    output_cuda = ms_deformable_attn(value_cuda, spatial_shapes,
                                     level_start_index, sampling_locations_cuda,
                                     attention_weights_cuda)
    output_cuda.sum().backward()

    res = paddle.allclose(
        value_paddle.grad, value_cuda.grad, rtol=1e-2, atol=1e-3).item()
    print(f'*tensor1 {res} check_gradient_numerical(D={channels})')

    res = paddle.allclose(
        sampling_locations_paddle.grad,
        sampling_locations_cuda.grad,
        rtol=1e-2,
        atol=1e-3).item()
    print(f'*tensor2 {res} check_gradient_numerical(D={channels})')

    res = paddle.allclose(
        attention_weights_paddle.grad,
        attention_weights_cuda.grad,
        rtol=1e-2,
        atol=1e-3).item()
    print(f'*tensor3 {res} check_gradient_numerical(D={channels})')


if __name__ == '__main__':
    check_forward_equal_with_paddle_float()

    for channels in [30, 32, 64, 71, 128, 1024, 1025, 2048, 3096]:
        check_gradient_numerical(channels)
