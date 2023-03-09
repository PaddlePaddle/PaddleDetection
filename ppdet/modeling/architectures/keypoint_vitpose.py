# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
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
from ppdet.core.workspace import register, create, serializable
from .meta_arch import BaseArch
from ..keypoint_utils import transform_preds
from .. import layers as L

__all__ = ['VitPose_TopDown', 'VitPosePostProcess']


@register
class VitPose_TopDown(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['loss']

    def __init__(self, backbone, head, loss, post_process, flip_test):
        """
        VitPose network, see https://arxiv.org/pdf/2204.12484v2.pdf

        Args:
            backbone (nn.Layer): backbone instance
            post_process (object): `HRNetPostProcess` instance
            
        """
        super(VitPose_TopDown, self).__init__()
        self.backbone = backbone
        self.head = head
        self.loss = loss
        self.post_process = post_process
        self.flip_test = flip_test

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        #head
        head = create(cfg['head'])
        #post_process
        post_process = create(cfg['post_process'])

        return {
            'backbone': backbone,
            'head': head,
            'post_process': post_process
        }

    def _forward_train(self):

        feats = self.backbone.forward_features(self.inputs['image'])
        vitpost_output = self.head(feats)
        return self.loss(vitpost_output, self.inputs)

    def _forward_test(self):

        feats = self.backbone.forward_features(self.inputs['image'])
        output_heatmap = self.head(feats)

        if self.flip_test:
            img_flipped = self.inputs['image'].flip(3)
            features_flipped = self.backbone.forward_features(img_flipped)
            output_flipped_heatmap = self.head.inference_model(features_flipped,
                                                               self.flip_test)

            output_heatmap = (output_heatmap + output_flipped_heatmap) * 0.5

        imshape = (self.inputs['im_shape'].numpy()
                   )[:, ::-1] if 'im_shape' in self.inputs else None
        center = self.inputs['center'].numpy(
        ) if 'center' in self.inputs else np.round(imshape / 2.)
        scale = self.inputs['scale'].numpy(
        ) if 'scale' in self.inputs else imshape / 200.

        result = self.post_process(output_heatmap.cpu().numpy(), center, scale)

        return result

    def get_loss(self):
        return self._forward_train()

    def get_pred(self):
        res_lst = self._forward_test()
        outputs = {'keypoint': res_lst}
        return outputs


@register
@serializable
class VitPosePostProcess(object):
    def __init__(self, use_dark=False):
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
        preds[:, :, 1] = np.floor((preds[:, :, 1]) // width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask

        return preds, maxvals

    def post_datk_udp(self, coords, batch_heatmaps, kernel=3):
        """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
        Devil is in the Details: Delving into Unbiased Data Processing for Human
        Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
        Representation for Human Pose Estimation (CVPR 2020).

        Note:
            - batch size: B
            - num keypoints: K
            - num persons: N
            - height of heatmaps: H
            - width of heatmaps: W

            B=1 for bottom_up paradigm where all persons share the same heatmap.
            B=N for top_down paradigm where each person has its own heatmaps.

        Args:
            coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
            batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
            kernel (int): Gaussian kernel size (K) for modulation.

        Returns:
            np.ndarray([N, K, 2]): Refined coordinates.
        """
        if not isinstance(batch_heatmaps, np.ndarray):
            batch_heatmaps = batch_heatmaps.cpu().numpy()
        B, K, H, W = batch_heatmaps.shape
        N = coords.shape[0]
        assert (B == 1 or B == N)
        for heatmaps in batch_heatmaps:
            for heatmap in heatmaps:
                cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
        np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
        np.log(batch_heatmaps, batch_heatmaps)

        batch_heatmaps_pad = np.pad(batch_heatmaps, ((0, 0), (0, 0), (1, 1),
                                                     (1, 1)),
                                    mode='edge').flatten()

        index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = batch_heatmaps_pad[index]
        ix1 = batch_heatmaps_pad[index + 1]
        iy1 = batch_heatmaps_pad[index + W + 2]
        ix1y1 = batch_heatmaps_pad[index + W + 3]
        ix1_y1_ = batch_heatmaps_pad[index - W - 3]
        ix1_ = batch_heatmaps_pad[index - 1]
        iy1_ = batch_heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(N, K, 2, 1)
        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(N, K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
        return coords

    def transform_preds_udp(self,
                            coords,
                            center,
                            scale,
                            output_size,
                            use_udp=True):
        """Get final keypoint predictions from heatmaps and apply scaling and
        translation to map them back to the image.

        Note:
            num_keypoints: K

        Args:
            coords (np.ndarray[K, ndims]):

                * If ndims=2, corrds are predicted keypoint location.
                * If ndims=4, corrds are composed of (x, y, scores, tags)
                * If ndims=5, corrds are composed of (x, y, scores, tags,
                flipped_tags)

            center (np.ndarray[2, ]): Center of the bounding box (x, y).
            scale (np.ndarray[2, ]): Scale of the bounding box
                wrt [width, height].
            output_size (np.ndarray[2, ] | list(2,)): Size of the
                destination heatmaps.
            use_udp (bool): Use unbiased data processing

        Returns:
            np.ndarray: Predicted coordinates in the images.
        """

        assert coords.shape[1] in (2, 4, 5)
        assert len(center) == 2
        assert len(scale) == 2
        assert len(output_size) == 2

        # Recover the scale which is normalized by a factor of 200.
        scale = scale * 200.0

        if use_udp:
            scale_x = scale[0] / (output_size[0] - 1.0)
            scale_y = scale[1] / (output_size[1] - 1.0)
        else:
            scale_x = scale[0] / output_size[0]
            scale_y = scale[1] / output_size[1]

        target_coords = np.ones_like(coords)
        target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[
            0] * 0.5
        target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[
            1] * 0.5

        return target_coords

    def get_final_preds(self, heatmaps, center, scale, kernelsize=11):
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

        N, K, H, W = heatmaps.shape

        if self.use_dark:
            coords = self.post_datk_udp(coords, heatmaps, kernelsize)
            preds = coords.copy()
            # Transform back to the image
            for i in range(N):
                preds[i] = self.transform_preds_udp(preds[i], center[i],
                                                    scale[i], [W, H])
        else:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ])
                        coords[n][p] += np.sign(diff) * .25
            preds = coords.copy()

            # Transform back
            for i in range(coords.shape[0]):
                preds[i] = transform_preds(coords[i], center[i], scale[i],
                                           [W, H])

        return preds, maxvals

    def __call__(self, output, center, scale):
        preds, maxvals = self.get_final_preds(output, center, scale)
        outputs = [[
            np.concatenate(
                (preds, maxvals), axis=-1), np.mean(
                    maxvals, axis=1)
        ]]
        return outputs