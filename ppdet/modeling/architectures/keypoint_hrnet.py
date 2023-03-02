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
import cv2
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ..keypoint_utils import transform_preds
from .. import layers as L
from paddle.nn import functional as F

__all__ = ['TopDownHRNet', 'TinyPose3DHRNet', 'TinyPose3DHRHeatmapNet']


@register
class TopDownHRNet(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['loss']

    def __init__(self,
                 width,
                 num_joints,
                 backbone='HRNet',
                 loss='KeyPointMSELoss',
                 post_process='HRNetPostProcess',
                 flip_perm=None,
                 flip=True,
                 shift_heatmap=True,
                 use_dark=True):
        """
        HRNet network, see https://arxiv.org/abs/1902.09212
 
        Args:
            backbone (nn.Layer): backbone instance
            post_process (object): `HRNetPostProcess` instance
            flip_perm (list): The left-right joints exchange order list
            use_dark(bool): Whether to use DARK in post processing
        """
        super(TopDownHRNet, self).__init__()
        self.backbone = backbone
        self.post_process = HRNetPostProcess(use_dark)
        self.loss = loss
        self.flip_perm = flip_perm
        self.flip = flip
        self.final_conv = L.Conv2d(width, num_joints, 1, 1, 0, bias=True)
        self.shift_heatmap = shift_heatmap
        self.deploy = False

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        return {'backbone': backbone, }

    def _forward(self):
        feats = self.backbone(self.inputs)
        hrnet_outputs = self.final_conv(feats[0])

        if self.training:
            return self.loss(hrnet_outputs, self.inputs)
        elif self.deploy:
            outshape = hrnet_outputs.shape
            max_idx = paddle.argmax(
                hrnet_outputs.reshape(
                    (outshape[0], outshape[1], outshape[2] * outshape[3])),
                axis=-1)
            return hrnet_outputs, max_idx
        else:
            if self.flip:
                self.inputs['image'] = self.inputs['image'].flip([3])
                feats = self.backbone(self.inputs)
                output_flipped = self.final_conv(feats[0])
                output_flipped = self.flip_back(output_flipped.numpy(),
                                                self.flip_perm)
                output_flipped = paddle.to_tensor(output_flipped.copy())
                if self.shift_heatmap:
                    output_flipped[:, :, :, 1:] = output_flipped.clone(
                    )[:, :, :, 0:-1]
                hrnet_outputs = (hrnet_outputs + output_flipped) * 0.5
            imshape = (self.inputs['im_shape'].numpy()
                       )[:, ::-1] if 'im_shape' in self.inputs else None
            center = self.inputs['center'].numpy(
            ) if 'center' in self.inputs else np.round(imshape / 2.)
            scale = self.inputs['scale'].numpy(
            ) if 'scale' in self.inputs else imshape / 200.
            outputs = self.post_process(hrnet_outputs, center, scale)
            return outputs

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        res_lst = self._forward()
        outputs = {'keypoint': res_lst}
        return outputs

    def flip_back(self, output_flipped, matched_parts):
        assert output_flipped.ndim == 4,\
                'output_flipped should be [batch_size, num_joints, height, width]'

        output_flipped = output_flipped[:, :, :, ::-1]

        for pair in matched_parts:
            tmp = output_flipped[:, pair[0], :, :].copy()
            output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
            output_flipped[:, pair[1], :, :] = tmp

        return output_flipped


class HRNetPostProcess(object):
    def __init__(self, use_dark=True):
        self.use_dark = use_dark

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

    def gaussian_blur(self, heatmap, kernel):
        border = (kernel - 1) // 2
        batch_size = heatmap.shape[0]
        num_joints = heatmap.shape[1]
        height = heatmap.shape[2]
        width = heatmap.shape[3]
        for i in range(batch_size):
            for j in range(num_joints):
                origin_max = np.max(heatmap[i, j])
                dr = np.zeros((height + 2 * border, width + 2 * border))
                dr[border:-border, border:-border] = heatmap[i, j].copy()
                dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
                heatmap[i, j] = dr[border:-border, border:-border].copy()
                heatmap[i, j] *= origin_max / np.max(heatmap[i, j])
        return heatmap

    def dark_parse(self, hm, coord):
        heatmap_height = hm.shape[0]
        heatmap_width = hm.shape[1]
        px = int(coord[0])
        py = int(coord[1])
        if 1 < px < heatmap_width - 2 and 1 < py < heatmap_height - 2:
            dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
            dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
            dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
            dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
                + hm[py-1][px-1])
            dyy = 0.25 * (
                hm[py + 2 * 1][px] - 2 * hm[py][px] + hm[py - 2 * 1][px])
            derivative = np.matrix([[dx], [dy]])
            hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy**2 != 0:
                hessianinv = hessian.I
                offset = -hessianinv * derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                coord += offset
        return coord

    def dark_postprocess(self, hm, coords, kernelsize):
        '''DARK postpocessing, Zhang et al. Distribution-Aware Coordinate
        Representation for Human Pose Estimation (CVPR 2020).
        '''

        hm = self.gaussian_blur(hm, kernelsize)
        hm = np.maximum(hm, 1e-10)
        hm = np.log(hm)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n, p] = self.dark_parse(hm[n][p], coords[n][p])
        return coords

    def get_final_preds(self, heatmaps, center, scale, kernelsize=3):
        """the highest heatvalue location with a quarter offset in the
        direction from the highest response to the second highest response.
 
        Args:
            heatmaps (numpy.ndarray): The predicted heatmaps
            center (numpy.ndarray): The boxes center
            scale (numpy.ndarray): The scale factor
 
        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 1]), the maximum confidence of the keypoints
        """
        coords, maxvals = self.get_max_preds(heatmaps)

        heatmap_height = heatmaps.shape[2]
        heatmap_width = heatmaps.shape[3]

        if self.use_dark:
            coords = self.dark_postprocess(heatmaps, coords, kernelsize)
        else:
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

    def __call__(self, output, center, scale):
        preds, maxvals = self.get_final_preds(output.numpy(), center, scale)
        outputs = [[
            np.concatenate(
                (preds, maxvals), axis=-1), np.mean(
                    maxvals, axis=1)
        ]]
        return outputs


class TinyPose3DPostProcess(object):
    def __init__(self):
        pass

    def __call__(self, output, center, scale):
        """
        Args:
            output (numpy.ndarray): numpy.ndarray([batch_size, num_joints, 3]), keypoints coords
            scale (numpy.ndarray): The scale factor
        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 3]), keypoints coords
        """

        preds = output.numpy().copy()

        # Transform back
        for i in range(output.shape[0]):  # batch_size
            preds[i][:, 0] = preds[i][:, 0] * scale[i][0]
            preds[i][:, 1] = preds[i][:, 1] * scale[i][1]

        return preds


def soft_argmax(heatmaps, joint_num):
    dims = heatmaps.shape
    depth_dim = (int)(dims[1] / joint_num)
    heatmaps = heatmaps.reshape((-1, joint_num, depth_dim * dims[2] * dims[3]))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, depth_dim, dims[2], dims[3]))

    accu_x = heatmaps.sum(axis=(2, 3))
    accu_y = heatmaps.sum(axis=(2, 4))
    accu_z = heatmaps.sum(axis=(3, 4))

    accu_x = accu_x * paddle.arange(1, 33)
    accu_y = accu_y * paddle.arange(1, 33)
    accu_z = accu_z * paddle.arange(1, 33)

    accu_x = accu_x.sum(axis=2, keepdim=True) - 1
    accu_y = accu_y.sum(axis=2, keepdim=True) - 1
    accu_z = accu_z.sum(axis=2, keepdim=True) - 1

    coord_out = paddle.concat(
        (accu_x, accu_y, accu_z), axis=2)  # [batch_size, joint_num, 3]

    return coord_out


@register
class TinyPose3DHRHeatmapNet(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['loss']

    def __init__(
            self,
            width,  # 40, backbone输出的channel数目
            num_joints,
            backbone='HRNet',
            loss='KeyPointRegressionMSELoss',
            post_process=TinyPose3DPostProcess):
        """
        Args:
            backbone (nn.Layer): backbone instance
            post_process (object): post process instance
        """
        super(TinyPose3DHRHeatmapNet, self).__init__()

        self.backbone = backbone
        self.post_process = TinyPose3DPostProcess()
        self.loss = loss
        self.deploy = False
        self.num_joints = num_joints

        self.final_conv = L.Conv2d(width, num_joints * 32, 1, 1, 0, bias=True)

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        return {'backbone': backbone, }

    def _forward(self):
        feats = self.backbone(self.inputs)  # feats:[[batch_size, 40, 32, 24]]

        hrnet_outputs = self.final_conv(feats[0])
        res = soft_argmax(hrnet_outputs, self.num_joints)
        return res

    def get_loss(self):
        pose3d = self._forward()
        loss = self.loss(pose3d, None, self.inputs)
        outputs = {'loss': loss}
        return outputs

    def get_pred(self):
        res_lst = self._forward()
        outputs = {'pose3d': res_lst}
        return outputs

    def flip_back(self, output_flipped, matched_parts):
        assert output_flipped.ndim == 4,\
                'output_flipped should be [batch_size, num_joints, height, width]'

        output_flipped = output_flipped[:, :, :, ::-1]

        for pair in matched_parts:
            tmp = output_flipped[:, pair[0], :, :].copy()
            output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
            output_flipped[:, pair[1], :, :] = tmp

        return output_flipped


@register
class TinyPose3DHRNet(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['loss']

    def __init__(self,
                 width,
                 num_joints,
                 fc_channel=768,
                 backbone='HRNet',
                 loss='KeyPointRegressionMSELoss',
                 post_process=TinyPose3DPostProcess):
        """
        Args:
            backbone (nn.Layer): backbone instance
            post_process (object): post process instance
        """
        super(TinyPose3DHRNet, self).__init__()
        self.backbone = backbone
        self.post_process = TinyPose3DPostProcess()
        self.loss = loss
        self.deploy = False
        self.num_joints = num_joints

        self.final_conv = L.Conv2d(width, num_joints, 1, 1, 0, bias=True)

        self.flatten = paddle.nn.Flatten(start_axis=2, stop_axis=3)
        self.fc1 = paddle.nn.Linear(fc_channel, 256)
        self.act1 = paddle.nn.ReLU()
        self.fc2 = paddle.nn.Linear(256, 64)
        self.act2 = paddle.nn.ReLU()
        self.fc3 = paddle.nn.Linear(64, 3)

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        return {'backbone': backbone, }

    def _forward(self):
        '''
        self.inputs is a dict
        '''
        feats = self.backbone(
            self.inputs)  # feats:[[batch_size, 40, width/4, height/4]]

        hrnet_outputs = self.final_conv(
            feats[0])  # hrnet_outputs: [batch_size, num_joints*32,32,32]

        flatten_res = self.flatten(
            hrnet_outputs)  # [batch_size,num_joints*32,32*32]

        res = self.fc1(flatten_res)
        res = self.act1(res)
        res = self.fc2(res)
        res = self.act2(res)
        res = self.fc3(res)

        if self.training:
            return self.loss(res, self.inputs)
        else:  # export model need
            return res

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        res_lst = self._forward()
        outputs = {'pose3d': res_lst}
        return outputs

    def flip_back(self, output_flipped, matched_parts):
        assert output_flipped.ndim == 4,\
                'output_flipped should be [batch_size, num_joints, height, width]'

        output_flipped = output_flipped[:, :, :, ::-1]

        for pair in matched_parts:
            tmp = output_flipped[:, pair[0], :, :].copy()
            output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
            output_flipped[:, pair[1], :, :] = tmp

        return output_flipped
