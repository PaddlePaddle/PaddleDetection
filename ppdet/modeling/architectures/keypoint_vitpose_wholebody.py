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
from typing import Tuple
import paddle
import numpy as np
import math
import cv2
from ppdet.core.workspace import register, create, serializable
from .meta_arch import BaseArch
from ..keypoint_utils import transform_preds
from .. import layers as L

__all__ = ['VitPose_TopDown_WholeBody', 'VitPoseWholeBodyPostProcess']


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals

def _taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (
            heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] -
            heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (
            heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] +
            heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord

def _box2cs(image_size, box):
        """This encodes bbox(x,y,w,h) into (center, scale)

        Args:
            x, y, w, h

        Returns:
            tuple: A tuple containing center and scale.

            - np.ndarray[float32](2,): Center of the bbox (x, y).
            - np.ndarray[float32](2,): Scale of the bbox w & h.
        """

        x, y, w, h = box[:4]
        aspect_ratio = image_size[0] / image_size[1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        scale = scale * 1.25

        return center, scale

def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([N, K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border),
                          dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps

def keypoints_from_heatmaps(heatmaps,
                                center,
                                scale,
                                unbiased=False,
                                post_process='default',
                                kernel=11,
                                valid_radius_factor=0.0546875,
                                use_udp=False,
                                target_type='GaussianHeatmap'):
        """Get final keypoint predictions from heatmaps and transform them back to
        the image.

        Note:
            - batch size: N
            - num keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
            center (np.ndarray[N, 2]): Center of the bounding box (x, y).
            scale (np.ndarray[N, 2]): Scale of the bounding box
                wrt height/width.
            post_process (str/None): Choice of methods to post-process
                heatmaps. Currently supported: None, 'default', 'unbiased',
                'megvii'.
            unbiased (bool): Option to use unbiased decoding. Mutually
                exclusive with megvii.
                Note: this arg is deprecated and unbiased=True can be replaced
                by post_process='unbiased'
                Paper ref: Zhang et al. Distribution-Aware Coordinate
                Representation for Human Pose Estimation (CVPR 2020).
            kernel (int): Gaussian kernel size (K) for modulation, which should
                match the heatmap gaussian sigma when training.
                K=17 for sigma=3 and k=11 for sigma=2.
            valid_radius_factor (float): The radius factor of the positive area
                in classification heatmap for UDP.
            use_udp (bool): Use unbiased data processing.
            target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
                GaussianHeatmap: Classification target with gaussian distribution.
                CombinedTarget: The combination of classification target
                (response map) and regression target (offset map).
                Paper ref: Huang et al. The Devil is in the Details: Delving into
                Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

        Returns:
            tuple: A tuple containing keypoint predictions and scores.

            - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
            - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
        """
        # Avoid being affected
        # heatmaps = heatmaps.copy()

        # detect conflicts
        # if unbiased:
        #     assert post_process not in [False, None, 'megvii']
        if post_process in ['megvii', 'unbiased']:
            assert kernel > 0
        if use_udp:
            assert not post_process == 'megvii'

        # normalize configs
        if post_process is False:
            warnings.warn(
                'post_process=False is deprecated, '
                'please use post_process=None instead', DeprecationWarning)
            post_process = None
        elif post_process is True:
            if unbiased is True:
                warnings.warn(
                    'post_process=True, unbiased=True is deprecated,'
                    " please use post_process='unbiased' instead",
                    DeprecationWarning)
                post_process = 'unbiased'
            else:
                warnings.warn(
                    'post_process=True, unbiased=False is deprecated, '
                    "please use post_process='default' instead",
                    DeprecationWarning)
                post_process = 'default'
        elif post_process == 'default':
            if unbiased is True:
                warnings.warn(
                    'unbiased=True is deprecated, please use '
                    "post_process='unbiased' instead", DeprecationWarning)
                post_process = 'unbiased'

        # start processing
        if post_process == 'megvii':
            heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

        N, K, H, W = heatmaps.shape
        if use_udp:
            if target_type.lower() == 'GaussianHeatMap'.lower():
                preds, maxvals = _get_max_preds(heatmaps)
                preds = post_dark_udp(preds, heatmaps, kernel=kernel)
            elif target_type.lower() == 'CombinedTarget'.lower():
                for person_heatmaps in heatmaps:
                    for i, heatmap in enumerate(person_heatmaps):
                        kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                        cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
                # valid radius is in direct proportion to the height of heatmap.
                valid_radius = valid_radius_factor * H
                offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
                offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
                heatmaps = heatmaps[:, ::3, :]
                preds, maxvals = _get_max_preds(heatmaps)
                index = preds[..., 0] + preds[..., 1] * W
                index += W * H * np.arange(0, N * K / 3)
                index = index.astype(int).reshape(N, K // 3, 1)
                preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
            else:
                raise ValueError('target_type should be either '
                                "'GaussianHeatmap' or 'CombinedTarget'")
        else:
            preds, maxvals = _get_max_preds(heatmaps)
            if post_process == 'unbiased':  # alleviate biased coordinate
                # apply Gaussian distribution modulation.
                heatmaps = np.log(
                    np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
                for n in range(N):
                    for k in range(K):
                        preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
            elif post_process is not None:
                # add +/-0.25 shift to the predicted locations for higher acc.
                for n in range(N):
                    for k in range(K):
                        heatmap = heatmaps[n][k]
                        px = int(preds[n][k][0])
                        py = int(preds[n][k][1])
                        if 1 < px < W - 1 and 1 < py < H - 1:
                            diff = np.array([
                                heatmap[py][px + 1] - heatmap[py][px - 1],
                                heatmap[py + 1][px] - heatmap[py - 1][px]
                            ])
                            preds[n][k] += np.sign(diff) * .25
                            if post_process == 'megvii':
                                preds[n][k] += 0.5

        # Transform back to the image
        for i in range(N):
            preds[i] = transform_preds(
                preds[i], center[i], scale[i], [W, H])

        if post_process == 'megvii':
            maxvals = maxvals / 255.0 + 0.5

        return preds, maxvals


# def post_process_vitpose

@register
@serializable
class VitPoseWholeBodyPostProcess(object):
    def __call__(self, img_metas, output, **kwargs):
        """Decode keypoints from heatmaps.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        img_metas = [{'center': img_metas['center'], 'scale': img_metas['scale'], 'rotation': 0, 'bbox_score': 1, 'flip_pairs': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 20], [18, 21], [19, 22], [23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34], [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], [43, 46], [44, 45], [54, 58], [55, 57], [59, 68], [60, 67], [61, 66], [62, 65], [63, 70], [64, 69], [71, 77], [72, 76], [73, 75], [78, 82], [79, 81], [83, 87], [84, 86], [88, 90], [91, 112], [92, 113], [93, 114], [94, 115], [95, 116], [96, 117], [97, 118], [98, 119], [99, 120], [100, 121], [101, 122], [102, 123], [103, 124], [104, 125], [105, 126], [106, 127], [107, 128], [108, 129], [109, 130], [110, 131], [111, 132]], 'bbox_id': 0}]


        batch_size = len(img_metas)

        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = keypoints_from_heatmaps(
            output,
            c,
            s,
            unbiased=False,
            post_process='unbiased',
            kernel=17,
            valid_radius_factor=0.0546875,
            use_udp=False,
            target_type='GaussianHeatmap'
            )

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['bbox_ids'] = bbox_ids

        return result

@register
class VitPose_TopDown_WholeBody(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['loss']

    def __init__(self, backbone, head, loss,  flip_test):
        """
        VitPose network, see https://arxiv.org/pdf/2204.12484v2.pdf

        Args:
            backbone (nn.Layer): backbone instance
            post_process (object): `HRNetPostProcess` instance
            
        """
        super(VitPose_TopDown_WholeBody, self).__init__()
        self.backbone = backbone
        self.head = head
        self.loss = loss
        self.flip_test = flip_test


    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        head = create(cfg['head'])

        return {
            'backbone': backbone,
            'head': head,
        }
    
    


    def _forward_train(self):

        feats = self.backbone.forward_features(self.inputs['image'])
        vitpost_output = self.head(feats)
        return self.loss(vitpost_output, self.inputs)

    def _forward_test(self,bbox=None):
        feats = self.backbone.forward_features(self.inputs['image'])
        print("feats")
        print(feats)
        output_heatmap = self.head(feats)

        if self.flip_test:
            img_flipped = self.inputs['image'].flip(3)
            features_flipped = self.backbone.forward_features(img_flipped)
            output_flipped_heatmap = self.head.inference_model(features_flipped,
                                                               self.flip_test)

            output_heatmap = (output_heatmap + output_flipped_heatmap) * 0.5

        imshape = (self.inputs['im_shape'].numpy()
                   )[:, ::-1] if 'im_shape' in self.inputs else none
        return output_heatmap

    def get_loss(self):
        return self._forward_train()

    def get_pred(self):
        res_lst = self._forward_test()
        outputs = {'keypoint': res_lst}
        return outputs

def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt

def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    # pixel_std is 200.
    scale_tmp = scale * 200.0

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans




@register
@serializable
class VitPosePreProcess(object):
    def __init__(self, use_dark=False):
        self.use_dark = use_dark

    def __call__(self, input=None):
        trans = get_affine_transform(np.array([124. , 180.5]), np.array([1.6921875, 2.25625]), 0, np.array([288, 384]))
        return input
