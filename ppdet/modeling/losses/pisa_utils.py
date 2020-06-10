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
import numpy as np

__all__ = ['get_isr_p_func']


def get_isr_p_func(pos_iou_thresh=0.25, bias=0, k=2):
    def irs_p(x):
        np.save("data", x)
        x = np.array(x)
        max_ious = x[:, :, 0]
        gt_inds = x[:, :, 1].astype('int32')
        cls = x[:, :, 2].astype('int32')

        # # n_{max}: max gt box num in each class
        # valid_gt = gt_box[:, :, 2] > 0. 
        # valid_gt_label = gt_label[valid_gt]
        # max_l_num = np.bincount(valid_gt_label).max()

        # divide gt index in each sample
        gt_inds = gt_inds + np.arange(gt_inds.shape[
            0])[:, np.newaxis] * gt_inds.shape[1]

        all_pos_weights = np.ones_like(max_ious)
        pos_mask = max_ious > pos_iou_thresh
        cls = np.reshape(cls, list(max_ious.shape) + [-1])
        max_ious = max_ious[pos_mask]
        pos_weights = all_pos_weights[pos_mask]
        gt_inds = gt_inds[pos_mask]
        cls = cls[pos_mask]
        max_l_num = np.bincount(cls.reshape(-1)).max()
        for l in np.unique(cls):
            l_inds = np.nonzero(cls == l)[0]
            l_gt_inds = gt_inds[l_inds]
            for t in np.unique(l_gt_inds):
                t_inds = np.array(l_inds)[l_gt_inds == t]
                t_max_ious = max_ious[t_inds]
                t_max_iou_rank = np.argsort(-t_max_ious).argsort().astype(
                    'float32')
                max_ious[t_inds] += np.clip(t_max_iou_rank, 0., None)
            l_max_ious = max_ious[l_inds]
            l_max_iou_rank = np.argsort(-l_max_ious).argsort().astype('float32')
            weight_factor = np.clip(max_l_num - l_max_iou_rank, 0.,
                                    None) / max_l_num
            weight_factor = np.power(bias + (1 - bias) * weight_factor, k)
            pos_weights[l_inds] *= weight_factor
        pos_weights = pos_weights / np.mean(pos_weights)
        all_pos_weights[pos_mask] = pos_weights

        return all_pos_weights

    return irs_p


if __name__ == "__main__":
    import numpy as np
    import paddle.fluid as fluid
    x_np = np.load('./data.npy')

    x = fluid.data(name='x', shape=[8, 15552, 3], dtype='float32')
    pos_weights = fluid.default_main_program().current_block().create_var(
        name="pos_weights", dtype='float32', shape=[8, 15552])
    isr_p = get_isr_p_func()
    fluid.layers.py_func(isr_p, x, pos_weights)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    ret = exe.run(fetch_list=[pos_weights.name], feed={'x': x_np})
    print(ret)
    np.save("ret", ret[0])
