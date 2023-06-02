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
"""
this code is base on https://github.com/hikvision-research/opera/blob/main/opera/models/detectors/petr.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from ppdet.core.workspace import register
from .meta_arch import BaseArch
from .. import layers as L

__all__ = ['PETR']


@register
class PETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['backbone', 'neck', 'bbox_head']

    def __init__(self,
                 backbone='ResNet',
                 neck='ChannelMapper',
                 bbox_head='PETRHead'):
        """
        PETR, see https://openaccess.thecvf.com/content/CVPR2022/papers/Shi_End-to-End_Multi-Person_Pose_Estimation_With_Transformers_CVPR_2022_paper.pdf

        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck between backbone and head
            bbox_head (nn.Layer): model output and loss
        """
        super(PETR, self).__init__()
        self.backbone = backbone
        if neck is not None:
            self.with_neck = True
        self.neck = neck
        self.bbox_head = bbox_head
        self.deploy = False

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def get_inputs(self):
        img_metas = []
        gt_bboxes = []
        gt_labels = []
        gt_keypoints = []
        gt_areas = []
        pad_gt_mask = self.inputs['pad_gt_mask'].astype("bool").squeeze(-1)
        for idx, im_shape in enumerate(self.inputs['im_shape']):
            img_meta = {
                'img_shape': im_shape.astype("int32").tolist() + [1, ],
                'batch_input_shape': self.inputs['image'].shape[-2:],
                'image_name': self.inputs['image_file'][idx]
            }
            img_metas.append(img_meta)
            if (not pad_gt_mask[idx].any()):
                gt_keypoints.append(self.inputs['gt_joints'][idx][:1])
                gt_labels.append(self.inputs['gt_class'][idx][:1])
                gt_bboxes.append(self.inputs['gt_bbox'][idx][:1])
                gt_areas.append(self.inputs['gt_areas'][idx][:1])
                continue

            gt_keypoints.append(self.inputs['gt_joints'][idx][pad_gt_mask[idx]])
            gt_labels.append(self.inputs['gt_class'][idx][pad_gt_mask[idx]])
            gt_bboxes.append(self.inputs['gt_bbox'][idx][pad_gt_mask[idx]])
            gt_areas.append(self.inputs['gt_areas'][idx][pad_gt_mask[idx]])

        return img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_areas

    def get_loss(self):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            gt_keypoints (list[Tensor]): Each item are the truth keypoints for
                each image in [p^{1}_x, p^{1}_y, p^{1}_v, ..., p^{K}_x,
                p^{K}_y, p^{K}_v] format.
            gt_areas (list[Tensor]): mask areas corresponding to each box.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        img_metas, gt_bboxes, gt_labels, gt_keypoints, gt_areas = self.get_inputs(
        )
        gt_bboxes_ignore = getattr(self.inputs, 'gt_bboxes_ignore', None)

        x = self.extract_feat(self.inputs)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_keypoints, gt_areas,
                                              gt_bboxes_ignore)
        loss = 0
        for k, v in losses.items():
            loss += v
        losses['loss'] = loss

        return losses

    def get_pred_numpy(self):
        """Used for computing network flops.
        """

        img = self.inputs['image']
        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3),
                scale_factor=(1., 1., 1., 1.)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, img_metas=dummy_img_metas)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, dummy_img_metas, rescale=True)
        return bbox_list

    def get_pred(self):
        """
        """
        img = self.inputs['image']
        batch_size, _, height, width = img.shape
        img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3),
                scale_factor=self.inputs['scale_factor'][i])
            for i in range(batch_size)
        ]
        kptpred = self.simple_test(
            self.inputs, img_metas=img_metas, rescale=True)
        keypoints = kptpred[0][1][0]
        bboxs = kptpred[0][0][0]
        keypoints[..., 2] = bboxs[:, None, 4]
        res_lst = [[keypoints, bboxs[:, 4]]]
        outputs = {'keypoint': res_lst}
        return outputs

    def simple_test(self, inputs, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            inputs (list[paddle.Tensor]): List of multiple images.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox and keypoint results of each image
                and classes. The outer list corresponds to each image.
                The inner list corresponds to each class.
        """
        batch_size = len(img_metas)
        assert batch_size == 1, 'Currently only batch_size 1 for inference ' \
            f'mode is supported. Found batch_size {batch_size}.'
        feat = self.extract_feat(inputs)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)

        bbox_kpt_results = [
            self.bbox_kpt2result(det_bboxes, det_labels, det_kpts,
                                 self.bbox_head.num_classes)
            for det_bboxes, det_labels, det_kpts in results_list
        ]
        return bbox_kpt_results

    def bbox_kpt2result(self, bboxes, labels, kpts, num_classes):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (paddle.Tensor | np.ndarray): shape (n, 5).
            labels (paddle.Tensor | np.ndarray): shape (n, ).
            kpts (paddle.Tensor | np.ndarray): shape (n, K, 3).
            num_classes (int): class number, including background class.

        Returns:
            list(ndarray): bbox and keypoint results of each class.
        """
        if bboxes.shape[0] == 0:
            return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)], \
                [np.zeros((0, kpts.size(1), 3), dtype=np.float32)
                    for i in range(num_classes)]
        else:
            if isinstance(bboxes, paddle.Tensor):
                bboxes = bboxes.numpy()
                labels = labels.numpy()
                kpts = kpts.numpy()
            return [bboxes[labels == i, :] for i in range(num_classes)], \
                [kpts[labels == i, :, :] for i in range(num_classes)]
