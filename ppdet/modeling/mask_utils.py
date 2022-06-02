#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn.functional as F


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list):
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = paddle.zeros(batch_shape, dtype=dtype)
        mask = paddle.ones((b, h, w), dtype=paddle.bool)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def nested_masks_from_list(tensor_list, input_shape=None):
    if tensor_list[0].ndim == 3:
        dim_size = sum([img.shape[0] for img in tensor_list])
        if input_shape is None:
            max_size = _max_by_axis(
                [list(img.shape[-2:]) for img in tensor_list])
        else:
            max_size = [input_shape[0], input_shape[1]]
        batch_shape = [dim_size] + max_size
        # b, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = paddle.zeros(batch_shape, dtype=dtype)
        mask = paddle.zeros(batch_shape, dtype=paddle.bool)
        idx = 0
        for img in tensor_list:
            c = img.shape[0]
            c_ = idx + c
            tensor[idx:c_, :img.shape[1], :img.shape[2]].copy_(img)
            mask[idx:c_, :img.shape[1], :img.shape[2]] = True
            idx = c_
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def aligned_bilinear(tensor, factor):
    # borrowed from Adelaidet: https://github1s.com/aim-uofa/AdelaiDet/blob/HEAD/adet/utils/comm.py
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.shape[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow), mode='bilinear', align_corners=True)
    tensor = F.pad(tensor,
                   pad=(factor // 2, 0, factor // 2, 0),
                   mode="replicate")

    return tensor[:, :, :oh - 1, :ow - 1]
