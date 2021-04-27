# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
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

import paddle
import numpy as np
import math
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ..keypoint_utils import transform_preds

__all__ = ['TopDownHrnet']


@register
class TopDownHrnet(BaseArch):
    __category__ = 'architecture'
    __shared__ = ['data_format']

    def __init__(self,
                 backbone='Hrnet',
                 hrnet_head='HrnetHead',
                 post_process='HrnetPostProcess',
                 flip_perm=None,
                 data_format='NCHW'):
        """
        HRNnet network, see https://arxiv.org/abs/1902.09212

        Args:
            backbone (nn.Layer): backbone instance
            hrnet_head (nn.Layer): keypoint_head instance
            post_process (object): `HrnetPostProcess` instance
            flip_perm (list): The left-right joints exchange order list
            data_format (str): data format, NCHW or NHWC
        """
        super(TopDownHrnet, self).__init__(data_format=data_format)
        self.backbone = backbone
        self.hrnet_head = hrnet_head
        self.post_process = HrnetPostProcess()
        self.flip_perm = flip_perm

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # head
        kwargs = {'input_shape': backbone.out_shape}
        hrnet_head = create(cfg['hrnet_head'], **kwargs)

        return {
            'backbone': backbone,
            "hrnet_head": hrnet_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)

        if self.training:
            return self.hrnet_head(body_feats, self.inputs)
        else:
            hrnet_head_outs = self.hrnet_head(body_feats)
            preds, maxvals = self.post_process(hrnet_head_outs, self.inputs,
                                               self.backbone, self.hrnet_head,
                                               self.flip_perm)
            return preds, maxvals

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        preds, maxvals = self._forward()
        output = {'kpt_coord': preds, 'kpt_score': maxvals}
        return output


class HrnetPostProcess(object):
    def __init__(self, flip=True, shift_heatmap=True):
        self.flip = flip
        self.shift_heatmap = shift_heatmap

    def flip_back(self, output_flipped, matched_parts):
        assert output_flipped.ndim == 4,\
                'output_flipped should be [batch_size, num_joints, height, width]'

        output_flipped = output_flipped[:, :, :, ::-1]

        for pair in matched_parts:
            tmp = output_flipped[:, pair[0], :, :].copy()
            output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
            output_flipped[:, pair[1], :, :] = tmp

        return output_flipped

    def get_max_preds(self, heatmaps):
        '''get predictions from score maps

        Args:
            heatmaps: numpy.ndarray([batch_size, num_joints, height, width])

        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 2]), the maximum confidence of the keypoints
        '''
        assert isinstance(heatmaps,
                          np.ndarray), 'heatmaps should be numpy.ndarray'
        assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        width = heatmaps.shape[3]
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask

        return preds, maxvals

    def get_final_preds(self, heatmaps, center, scale):
        """the highest heatvalue location with a quarter offset in the
        direction from the highest response to the second highest response.

        Args:
            heatmaps (numpy.ndarray): The predicted heatmaps
            center (numpy.ndarray): The boxes center
            scale (numpy.ndarray): The scale factor

        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 2]), the maximum confidence of the keypoints
        """

        coords, maxvals = self.get_max_preds(heatmaps)

        heatmap_height = heatmaps.shape[2]
        heatmap_width = heatmaps.shape[3]

        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array([
                        hm[py][px + 1] - hm[py][px - 1],
                        hm[py + 1][px] - hm[py - 1][px]
                    ])
                    coords[n][p] += np.sign(diff) * .25
        preds = coords.copy()

        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(coords[i], center[i], scale[i],
                                       [heatmap_width, heatmap_height])

        return preds, maxvals

    def __call__(self, output, inputs, backbone, head, flip_perm):
        if self.flip:
            inputs['image'] = inputs['image'].flip([3])
            feats = backbone(inputs)
            output_flipped = head(feats)
            output_flipped = self.flip_back(output_flipped.numpy(), flip_perm)
            output_flipped = paddle.to_tensor(output_flipped.copy())
            if self.shift_heatmap:
                output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:
                                                                     -1]
            output = (output + output_flipped) * 0.5
        preds, maxvals = self.get_final_preds(
            output.numpy(), inputs['center'].numpy(), inputs['scale'].numpy())

        return preds, maxvals
