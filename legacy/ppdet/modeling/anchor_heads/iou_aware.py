# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import fluid


def _split_ioup(output, an_num, num_classes):
    """
    Split new output feature map to output, predicted iou
    along channel dimension
    """
    ioup = fluid.layers.slice(output, axes=[1], starts=[0], ends=[an_num])
    ioup = fluid.layers.sigmoid(ioup)

    oriout = fluid.layers.slice(
        output, axes=[1], starts=[an_num], ends=[an_num * (num_classes + 6)])

    return (ioup, oriout)


def _de_sigmoid(x, eps=1e-7):
    x = fluid.layers.clip(x, eps, 1 / eps)
    one = fluid.layers.fill_constant(
        shape=[1, 1, 1, 1], dtype=x.dtype, value=1.)
    x = fluid.layers.clip((one / x - 1.0), eps, 1 / eps)
    x = -fluid.layers.log(x)
    return x


def _postprocess_output(ioup, output, an_num, num_classes, iou_aware_factor):
    """
    post process output objectness score
    """
    tensors = []
    stride = output.shape[1] // an_num
    for m in range(an_num):
        tensors.append(
            fluid.layers.slice(
                output,
                axes=[1],
                starts=[stride * m + 0],
                ends=[stride * m + 4]))
        obj = fluid.layers.slice(
            output, axes=[1], starts=[stride * m + 4], ends=[stride * m + 5])
        obj = fluid.layers.sigmoid(obj)
        ip = fluid.layers.slice(ioup, axes=[1], starts=[m], ends=[m + 1])

        new_obj = fluid.layers.pow(obj, (
            1 - iou_aware_factor)) * fluid.layers.pow(ip, iou_aware_factor)
        new_obj = _de_sigmoid(new_obj)

        tensors.append(new_obj)

        tensors.append(
            fluid.layers.slice(
                output,
                axes=[1],
                starts=[stride * m + 5],
                ends=[stride * m + 5 + num_classes]))

    output = fluid.layers.concat(tensors, axis=1)

    return output


def get_iou_aware_score(output, an_num, num_classes, iou_aware_factor):
    ioup, output = _split_ioup(output, an_num, num_classes)
    output = _postprocess_output(ioup, output, an_num, num_classes,
                                 iou_aware_factor)
    return output
