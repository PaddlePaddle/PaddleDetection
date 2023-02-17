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
this code is base on https://github.com/hikvision-research/opera/blob/main/opera/models/dense_heads/petr_head.py
"""
import copy
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
import paddle.distributed as dist

from ..transformers.petr_transformer import inverse_sigmoid, masked_fill
from ..initializer import constant_, normal_

__all__ = ["PETRHead"]

from functools import partial


def bias_init_with_prob(prior_prob: float) -> float:
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    res = tuple(map(list, zip(*map_results)))
    return res


def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.get_world_size() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(
        tensor.divide(
            paddle.to_tensor(
                dist.get_world_size(), dtype='float32')),
        op=dist.ReduceOp.SUM)
    return tensor


def gaussian_radius(det_size, min_overlap=0.7):
    """calculate gaussian radius according to object size.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = paddle.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = paddle.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = paddle.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y = paddle.arange(-m, m + 1, dtype="float32")[:, None]
    x = paddle.arange(-n, n + 1, dtype="float32")[None, :]
    # y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = paddle.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(np.float32).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    gaussian = paddle.to_tensor(gaussian, dtype=heatmap.dtype)

    x, y = int(center[0]), int(center[1])
    radius = int(radius)

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:
                               radius + right]
    # assert masked_gaussian.equal(1).float().sum() == 1
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        heatmap[y - top:y + bottom, x - left:x + right] = paddle.maximum(
            masked_heatmap, masked_gaussian * k)
    return heatmap


@register
class PETRHead(nn.Layer):
    """Head of `End-to-End Multi-Person Pose Estimation with Transformers`.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_kpt_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the keypoint regression head.
            Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): ConfigDict is used for
            building the Encoder and Decoder. Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_kpt (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_oks (obj:`mmcv.ConfigDict`|dict): Config of the
            regression oks loss. Default `OKSLoss`.
        loss_hm (obj:`mmcv.ConfigDict`|dict): Config of the
            regression heatmap loss. Default `NegLoss`.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        with_kpt_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to True.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """
    __inject__ = [
        "transformer", "positional_encoding", "assigner", "sampler", "loss_cls",
        "loss_kpt", "loss_oks", "loss_hm", "loss_kpt_rpn", "loss_kpt_refine",
        "loss_oks_refine"
    ]

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_kpt_fcs=2,
                 num_keypoints=17,
                 transformer=None,
                 sync_cls_avg_factor=True,
                 positional_encoding='SinePositionalEncoding',
                 loss_cls='FocalLoss',
                 loss_kpt='L1Loss',
                 loss_oks='OKSLoss',
                 loss_hm='CenterFocalLoss',
                 with_kpt_refine=True,
                 assigner='PoseHungarianAssigner',
                 sampler='PseudoSampler',
                 loss_kpt_rpn='L1Loss',
                 loss_kpt_refine='L1Loss',
                 loss_oks_refine='opera.OKSLoss',
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super().__init__()
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.assigner = assigner
        self.sampler = sampler
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_kpt_fcs = num_kpt_fcs
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.as_two_stage = transformer.as_two_stage
        self.with_kpt_refine = with_kpt_refine
        self.num_keypoints = num_keypoints
        self.loss_cls = loss_cls
        self.loss_kpt = loss_kpt
        self.loss_kpt_rpn = loss_kpt_rpn
        self.loss_kpt_refine = loss_kpt_refine
        self.loss_oks = loss_oks
        self.loss_oks_refine = loss_oks_refine
        self.loss_hm = loss_hm
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.positional_encoding = positional_encoding
        self.transformer = transformer
        self.embed_dims = self.transformer.embed_dims
        # assert 'num_feats' in positional_encoding
        num_feats = positional_encoding.num_pos_feats
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        """Initialize classification branch and keypoint branch of head."""

        fc_cls = nn.Linear(self.embed_dims, self.cls_out_channels)

        kpt_branch = []
        kpt_branch.append(nn.Linear(self.embed_dims, 512))
        kpt_branch.append(nn.ReLU())
        for _ in range(self.num_kpt_fcs):
            kpt_branch.append(nn.Linear(512, 512))
            kpt_branch.append(nn.ReLU())
        kpt_branch.append(nn.Linear(512, 2 * self.num_keypoints))
        kpt_branch = nn.Sequential(*kpt_branch)

        def _get_clones(module, N):
            return nn.LayerList([copy.deepcopy(module) for i in range(N)])

        # last kpt_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_kpt_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.kpt_branches = _get_clones(kpt_branch, num_pred)
        else:
            self.cls_branches = nn.LayerList([fc_cls for _ in range(num_pred)])
            self.kpt_branches = nn.LayerList(
                [kpt_branch for _ in range(num_pred)])

        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

        refine_kpt_branch = []
        for _ in range(self.num_kpt_fcs):
            refine_kpt_branch.append(
                nn.Linear(self.embed_dims, self.embed_dims))
            refine_kpt_branch.append(nn.ReLU())
        refine_kpt_branch.append(nn.Linear(self.embed_dims, 2))
        refine_kpt_branch = nn.Sequential(*refine_kpt_branch)
        if self.with_kpt_refine:
            num_pred = self.transformer.refine_decoder.num_layers
            self.refine_kpt_branches = _get_clones(refine_kpt_branch, num_pred)
        self.fc_hm = nn.Linear(self.embed_dims, self.num_keypoints)

    def init_weights(self):
        """Initialize weights of the PETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                constant_(m.bias, bias_init)
        for m in self.kpt_branches:
            constant_(m[-1].bias, 0)
        # initialization of keypoint refinement branch
        if self.with_kpt_refine:
            for m in self.refine_kpt_branches:
                constant_(m[-1].bias, 0)
        # initialize bias for heatmap prediction
        bias_init = bias_init_with_prob(0.1)
        normal_(self.fc_hm.weight, std=0.01)
        constant_(self.fc_hm.bias, bias_init)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            outputs_classes (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should include background.
            outputs_kpts (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, K*2].
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (N, h*w, num_class). Only when
                as_two_stage is Ture it would be returned, otherwise
                `None` would be returned.
            enc_outputs_kpt (Tensor): The proposal generate from the
                encode feature map, has shape (N, h*w, K*2). Only when
                as_two_stage is Ture it would be returned, otherwise
                `None` would be returned.
        """

        batch_size = mlvl_feats[0].shape[0]
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = paddle.zeros(
            (batch_size, input_img_h, input_img_w), dtype=mlvl_feats[0].dtype)
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 1

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(
                    img_masks[None], size=feat.shape[-2:]).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]).transpose(
                    [0, 3, 1, 2]))

        query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_kpt, hm_proto, memory = \
                self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    kpt_branches=self.kpt_branches \
                        if self.with_kpt_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches \
                        if self.as_two_stage else None  # noqa:E501
            )

        outputs_classes = []
        outputs_kpts = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp_kpt = self.kpt_branches[lvl](hs[lvl])
            assert reference.shape[-1] == self.num_keypoints * 2
            tmp_kpt += reference
            outputs_kpt = F.sigmoid(tmp_kpt)
            outputs_classes.append(outputs_class)
            outputs_kpts.append(outputs_kpt)

        outputs_classes = paddle.stack(outputs_classes)
        outputs_kpts = paddle.stack(outputs_kpts)

        if hm_proto is not None:
            # get heatmap prediction (training phase)
            hm_memory, hm_mask = hm_proto
            hm_pred = self.fc_hm(hm_memory)
            hm_proto = (hm_pred.transpose((0, 3, 1, 2)), hm_mask)

        if self.as_two_stage:
            return outputs_classes, outputs_kpts, \
                enc_outputs_class, F.sigmoid(enc_outputs_kpt), \
                hm_proto, memory, mlvl_masks
        else:
            raise RuntimeError('only "as_two_stage=True" is supported.')

    def forward_refine(self, memory, mlvl_masks, refine_targets, losses,
                       img_metas):
        """Forward function.

        Args:
            mlvl_masks (tuple[Tensor]): The key_padding_mask from
                different level used for encoder and decoder,
                each is a 3D-tensor with shape (bs, H, W).
            losses (dict[str, Tensor]): A dictionary of loss components.
            img_metas (list[dict]): List of image information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        kpt_preds, kpt_targets, area_targets, kpt_weights = refine_targets
        pos_inds = kpt_weights.sum(-1) > 0
        if not pos_inds.any():
            pos_kpt_preds = paddle.zeros_like(kpt_preds[:1])
            pos_img_inds = paddle.zeros([1], dtype="int64")
        else:
            pos_kpt_preds = kpt_preds[pos_inds]
            pos_img_inds = (pos_inds.nonzero() /
                            self.num_query).squeeze(1).astype("int64")
        hs, init_reference, inter_references = self.transformer.forward_refine(
            mlvl_masks,
            memory,
            pos_kpt_preds.detach(),
            pos_img_inds,
            kpt_branches=self.refine_kpt_branches
            if self.with_kpt_refine else None,  # noqa:E501
        )

        outputs_kpts = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            tmp_kpt = self.refine_kpt_branches[lvl](hs[lvl])
            assert reference.shape[-1] == 2
            tmp_kpt += reference
            outputs_kpt = F.sigmoid(tmp_kpt)
            outputs_kpts.append(outputs_kpt)
        outputs_kpts = paddle.stack(outputs_kpts)

        if not self.training:
            return outputs_kpts

        num_valid_kpt = paddle.clip(
            reduce_mean(kpt_weights.sum()), min=1).item()
        num_total_pos = paddle.to_tensor(
            [outputs_kpts.shape[1]], dtype=kpt_weights.dtype)
        num_total_pos = paddle.clip(reduce_mean(num_total_pos), min=1).item()

        if not pos_inds.any():
            for i, kpt_refine_preds in enumerate(outputs_kpts):
                loss_kpt = loss_oks = kpt_refine_preds.sum() * 0
                losses[f'd{i}.loss_kpt_refine'] = loss_kpt
                losses[f'd{i}.loss_oks_refine'] = loss_oks
                continue
            return losses

        batch_size = mlvl_masks[0].shape[0]
        factors = []
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            factor = paddle.to_tensor(
                [img_w, img_h, img_w, img_h],
                dtype="float32").squeeze(-1).unsqueeze(0).tile(
                    (self.num_query, 1))
            factors.append(factor)
        factors = paddle.concat(factors, 0)
        factors = factors[pos_inds][:, :2].tile((1, kpt_preds.shape[-1] // 2))

        pos_kpt_weights = kpt_weights[pos_inds]
        pos_kpt_targets = kpt_targets[pos_inds]
        pos_kpt_targets_scaled = pos_kpt_targets * factors
        pos_areas = area_targets[pos_inds]
        pos_valid = kpt_weights[pos_inds][:, 0::2]
        for i, kpt_refine_preds in enumerate(outputs_kpts):
            if not pos_inds.any():
                print("refine kpt and oks skip")
                loss_kpt = loss_oks = kpt_refine_preds.sum() * 0
                losses[f'd{i}.loss_kpt_refine'] = loss_kpt
                losses[f'd{i}.loss_oks_refine'] = loss_oks
                continue

            # kpt L1 Loss
            pos_refine_preds = kpt_refine_preds.reshape(
                (kpt_refine_preds.shape[0], -1))
            loss_kpt = self.loss_kpt_refine(
                pos_refine_preds,
                pos_kpt_targets,
                pos_kpt_weights,
                avg_factor=num_valid_kpt)
            losses[f'd{i}.loss_kpt_refine'] = loss_kpt
            # kpt oks loss
            pos_refine_preds_scaled = pos_refine_preds * factors
            assert (pos_areas > 0).all()
            loss_oks = self.loss_oks_refine(
                pos_refine_preds_scaled,
                pos_kpt_targets_scaled,
                pos_valid,
                pos_areas,
                avg_factor=num_total_pos)
            losses[f'd{i}.loss_oks_refine'] = loss_oks
        return losses

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_keypoints=None,
                      gt_areas=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gts,).
            gt_keypoints (list[Tensor]): Ground truth keypoints of the image,
                shape (num_gts, K*3).
            gt_areas (list[Tensor]): Ground truth mask areas of each box,
                shape (num_gts,).
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        memory, mlvl_masks = outs[-2:]
        outs = outs[:-2]
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, gt_keypoints, gt_areas, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_keypoints, gt_areas,
                                  img_metas)
        losses_and_targets = self.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # losses = losses_and_targets
        losses, refine_targets = losses_and_targets
        # get pose refinement loss
        losses = self.forward_refine(memory, mlvl_masks, refine_targets, losses,
                                     img_metas)
        return losses

    def loss(self,
             all_cls_scores,
             all_kpt_preds,
             enc_cls_scores,
             enc_kpt_preds,
             enc_hm_proto,
             gt_bboxes_list,
             gt_labels_list,
             gt_keypoints_list,
             gt_areas_list,
             img_metas,
             gt_bboxes_ignore=None):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_kpt_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (x_{i}, y_{i}) and shape
                [nb_dec, bs, num_query, K*2].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map, has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_kpt_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, K*2). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v,
                    ..., p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_keypoints_list = [
            gt_keypoints_list for _ in range(num_dec_layers)
        ]
        all_gt_areas_list = [gt_areas_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_kpt, losses_oks, kpt_preds_list, kpt_targets_list, \
            area_targets_list, kpt_weights_list = multi_apply(
                self.loss_single, all_cls_scores, all_kpt_preds,
                all_gt_labels_list, all_gt_keypoints_list,
                all_gt_areas_list, img_metas_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                paddle.zeros_like(gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_kpt = \
                self.loss_single_rpn(
                    enc_cls_scores, enc_kpt_preds, binary_labels_list,
                    gt_keypoints_list, gt_areas_list, img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_kpt'] = enc_losses_kpt

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_kpt'] = losses_kpt[-1]
        loss_dict['loss_oks'] = losses_oks[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_kpt_i, loss_oks_i in zip(
                losses_cls[:-1], losses_kpt[:-1], losses_oks[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_kpt'] = loss_kpt_i
            loss_dict[f'd{num_dec_layer}.loss_oks'] = loss_oks_i
            num_dec_layer += 1

        # losses of heatmap generated from P3 feature map
        hm_pred, hm_mask = enc_hm_proto
        loss_hm = self.loss_heatmap(hm_pred, hm_mask, gt_keypoints_list,
                                    gt_labels_list, gt_bboxes_list)
        loss_dict['loss_hm'] = loss_hm

        return loss_dict, (kpt_preds_list[-1], kpt_targets_list[-1],
                           area_targets_list[-1], kpt_weights_list[-1])

    def loss_heatmap(self, hm_pred, hm_mask, gt_keypoints, gt_labels,
                     gt_bboxes):
        assert hm_pred.shape[-2:] == hm_mask.shape[-2:]
        num_img, _, h, w = hm_pred.shape
        # placeholder of heatmap target (Gaussian distribution)
        hm_target = paddle.zeros(hm_pred.shape, hm_pred.dtype)
        for i, (gt_label, gt_bbox, gt_keypoint
                ) in enumerate(zip(gt_labels, gt_bboxes, gt_keypoints)):
            if gt_label.shape[0] == 0:
                continue
            gt_keypoint = gt_keypoint.reshape((gt_keypoint.shape[0], -1,
                                               3)).clone()
            gt_keypoint[..., :2] /= 8

            assert gt_keypoint[..., 0].max() <= w + 0.5  # new coordinate system
            assert gt_keypoint[..., 1].max() <= h + 0.5  # new coordinate system
            gt_bbox /= 8
            gt_w = gt_bbox[:, 2] - gt_bbox[:, 0]
            gt_h = gt_bbox[:, 3] - gt_bbox[:, 1]
            for j in range(gt_label.shape[0]):
                # get heatmap radius
                kp_radius = paddle.clip(
                    paddle.floor(
                        gaussian_radius(
                            (gt_h[j], gt_w[j]), min_overlap=0.9)),
                    min=0,
                    max=3)
                for k in range(self.num_keypoints):
                    if gt_keypoint[j, k, 2] > 0:
                        gt_kp = gt_keypoint[j, k, :2]
                        gt_kp_int = paddle.floor(gt_kp)
                        hm_target[i, k] = draw_umich_gaussian(
                            hm_target[i, k], gt_kp_int, kp_radius)
        # compute heatmap loss
        hm_pred = paddle.clip(
            F.sigmoid(hm_pred), min=1e-4, max=1 - 1e-4)  # refer to CenterNet
        loss_hm = self.loss_hm(
            hm_pred,
            hm_target.detach(),
            mask=~hm_mask.astype("bool").unsqueeze(1))
        return loss_hm

    def loss_single(self, cls_scores, kpt_preds, gt_labels_list,
                    gt_keypoints_list, gt_areas_list, img_metas):
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            kpt_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (x_{i}, y_{i}) and
                shape [bs, num_query, K*2].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v,
                ..., p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.shape[0]
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        kpt_preds_list = [kpt_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, kpt_preds_list,
                                           gt_labels_list, gt_keypoints_list,
                                           gt_areas_list, img_metas)
        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list,
         area_targets_list, num_total_pos, num_total_neg) = cls_reg_targets
        labels = paddle.concat(labels_list, 0)
        label_weights = paddle.concat(label_weights_list, 0)
        kpt_targets = paddle.concat(kpt_targets_list, 0)
        kpt_weights = paddle.concat(kpt_weights_list, 0)
        area_targets = paddle.concat(area_targets_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape((-1, self.cls_out_channels))
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                paddle.to_tensor(
                    [cls_avg_factor], dtype=cls_scores.dtype))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt keypoints accross all gpus, for
        # normalization purposes
        num_total_pos = paddle.to_tensor([num_total_pos], dtype=loss_cls.dtype)
        num_total_pos = paddle.clip(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale keypoints
        factors = []
        for img_meta, kpt_pred in zip(img_metas, kpt_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = paddle.to_tensor(
                [img_w, img_h, img_w, img_h],
                dtype=kpt_pred.dtype).squeeze().unsqueeze(0).tile(
                    (kpt_pred.shape[0], 1))
            factors.append(factor)
        factors = paddle.concat(factors, 0)

        # keypoint regression loss
        kpt_preds = kpt_preds.reshape((-1, kpt_preds.shape[-1]))
        num_valid_kpt = paddle.clip(
            reduce_mean(kpt_weights.sum()), min=1).item()
        # assert num_valid_kpt == (kpt_targets>0).sum().item()
        loss_kpt = self.loss_kpt(
            kpt_preds,
            kpt_targets.detach(),
            kpt_weights.detach(),
            avg_factor=num_valid_kpt)

        # keypoint oks loss
        pos_inds = kpt_weights.sum(-1) > 0
        if not pos_inds.any():
            loss_oks = kpt_preds.sum() * 0
        else:
            factors = factors[pos_inds][:, :2].tile((
                (1, kpt_preds.shape[-1] // 2)))
            pos_kpt_preds = kpt_preds[pos_inds] * factors
            pos_kpt_targets = kpt_targets[pos_inds] * factors
            pos_areas = area_targets[pos_inds]
            pos_valid = kpt_weights[pos_inds][..., 0::2]
            assert (pos_areas > 0).all()
            loss_oks = self.loss_oks(
                pos_kpt_preds,
                pos_kpt_targets,
                pos_valid,
                pos_areas,
                avg_factor=num_total_pos)
        return loss_cls, loss_kpt, loss_oks, kpt_preds, kpt_targets, \
            area_targets, kpt_weights

    def get_targets(self, cls_scores_list, kpt_preds_list, gt_labels_list,
                    gt_keypoints_list, gt_areas_list, img_metas):
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            kpt_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (x_{i}, y_{i}) and shape [num_query, K*2].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3).
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all
                    images.
                - kpt_targets_list (list[Tensor]): Keypoint targets for all
                    images.
                - kpt_weights_list (list[Tensor]): Keypoint weights for all
                    images.
                - area_targets_list (list[Tensor]): area targets for all
                    images.
                - num_total_pos (int): Number of positive samples in all
                    images.
                - num_total_neg (int): Number of negative samples in all
                    images.
        """
        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list,
         area_targets_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, kpt_preds_list,
             gt_labels_list, gt_keypoints_list, gt_areas_list, img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, kpt_targets_list,
                kpt_weights_list, area_targets_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self, cls_score, kpt_pred, gt_labels, gt_keypoints,
                           gt_areas, img_meta):
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            kpt_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (x_{i}, y_{i}) and
                shape [num_query, K*2].
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_keypoints (Tensor): Ground truth keypoints for one image with
                shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v, ..., \
                    p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas (Tensor): Ground truth mask areas for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor): Label weights of each image.
                - kpt_targets (Tensor): Keypoint targets of each image.
                - kpt_weights (Tensor): Keypoint weights of each image.
                - area_targets (Tensor): Area targets of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = kpt_pred.shape[0]
        # assigner and sampler
        assign_result = self.assigner.assign(cls_score, kpt_pred, gt_labels,
                                             gt_keypoints, gt_areas, img_meta)
        sampling_result = self.sampler.sample(assign_result, kpt_pred,
                                              gt_keypoints)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = paddle.full((num_bboxes, ), self.num_classes, dtype="int64")
        label_weights = paddle.ones((num_bboxes, ), dtype=gt_labels.dtype)
        kpt_targets = paddle.zeros_like(kpt_pred)
        kpt_weights = paddle.zeros_like(kpt_pred)
        area_targets = paddle.zeros((kpt_pred.shape[0], ), dtype=kpt_pred.dtype)

        if pos_inds.size == 0:
            return (labels, label_weights, kpt_targets, kpt_weights,
                    area_targets, pos_inds, neg_inds)

        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds][
            ..., 0].astype("int64")

        img_h, img_w, _ = img_meta['img_shape']
        # keypoint targets
        pos_gt_kpts = gt_keypoints[sampling_result.pos_assigned_gt_inds]
        pos_gt_kpts = pos_gt_kpts.reshape(
            (len(sampling_result.pos_assigned_gt_inds), -1, 3))
        valid_idx = pos_gt_kpts[:, :, 2] > 0
        pos_kpt_weights = kpt_weights[pos_inds].reshape(
            (pos_gt_kpts.shape[0], kpt_weights.shape[-1] // 2, 2))
        # pos_kpt_weights[valid_idx][...] = 1.0
        pos_kpt_weights = masked_fill(pos_kpt_weights,
                                      valid_idx.unsqueeze(-1), 1.0)
        kpt_weights[pos_inds] = pos_kpt_weights.reshape(
            (pos_kpt_weights.shape[0], kpt_pred.shape[-1]))

        factor = paddle.to_tensor(
            [img_w, img_h], dtype=kpt_pred.dtype).squeeze().unsqueeze(0)
        pos_gt_kpts_normalized = pos_gt_kpts[..., :2]
        pos_gt_kpts_normalized[..., 0] = pos_gt_kpts_normalized[..., 0] / \
            factor[:, 0:1]
        pos_gt_kpts_normalized[..., 1] = pos_gt_kpts_normalized[..., 1] / \
            factor[:, 1:2]
        kpt_targets[pos_inds] = pos_gt_kpts_normalized.reshape(
            (pos_gt_kpts.shape[0], kpt_pred.shape[-1]))

        pos_gt_areas = gt_areas[sampling_result.pos_assigned_gt_inds][..., 0]
        area_targets[pos_inds] = pos_gt_areas

        return (labels, label_weights, kpt_targets, kpt_weights, area_targets,
                pos_inds, neg_inds)

    def loss_single_rpn(self, cls_scores, kpt_preds, gt_labels_list,
                        gt_keypoints_list, gt_areas_list, img_metas):
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            kpt_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (x_{i}, y_{i}) and
                shape [bs, num_query, K*2].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_keypoints_list (list[Tensor]): Ground truth keypoints for each
                image with shape (num_gts, K*3) in [p^{1}_x, p^{1}_y, p^{1}_v,
                ..., p^{K}_x, p^{K}_y, p^{K}_v] format.
            gt_areas_list (list[Tensor]): Ground truth mask areas for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.shape[0]
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        kpt_preds_list = [kpt_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, kpt_preds_list,
                                           gt_labels_list, gt_keypoints_list,
                                           gt_areas_list, img_metas)
        (labels_list, label_weights_list, kpt_targets_list, kpt_weights_list,
         area_targets_list, num_total_pos, num_total_neg) = cls_reg_targets
        labels = paddle.concat(labels_list, 0)
        label_weights = paddle.concat(label_weights_list, 0)
        kpt_targets = paddle.concat(kpt_targets_list, 0)
        kpt_weights = paddle.concat(kpt_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape((-1, self.cls_out_channels))
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                paddle.to_tensor(
                    [cls_avg_factor], dtype=cls_scores.dtype))
        cls_avg_factor = max(cls_avg_factor, 1)

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt keypoints accross all gpus, for
        # normalization purposes
        # num_total_pos = loss_cls.to_tensor([num_total_pos])
        # num_total_pos = paddle.clip(reduce_mean(num_total_pos), min=1).item()

        # keypoint regression loss
        kpt_preds = kpt_preds.reshape((-1, kpt_preds.shape[-1]))
        num_valid_kpt = paddle.clip(
            reduce_mean(kpt_weights.sum()), min=1).item()
        # assert num_valid_kpt == (kpt_targets>0).sum().item()
        loss_kpt = self.loss_kpt_rpn(
            kpt_preds, kpt_targets, kpt_weights, avg_factor=num_valid_kpt)

        return loss_cls, loss_kpt

    def get_bboxes(self,
                   all_cls_scores,
                   all_kpt_preds,
                   enc_cls_scores,
                   enc_kpt_preds,
                   hm_proto,
                   memory,
                   mlvl_masks,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_kpt_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (x_{i}, y_{i}) and shape
                [nb_dec, bs, num_query, K*2].
            enc_cls_scores (Tensor): Classification scores of points on
                encode feature map, has shape (N, h*w, num_classes).
                Only be passed when as_two_stage is True, otherwise is None.
            enc_kpt_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, K*2). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Defalut False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 3-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box. The third item is an (n, K, 3) tensor
                with [p^{1}_x, p^{1}_y, p^{1}_v, ..., p^{K}_x, p^{K}_y,
                p^{K}_v] format.
        """
        cls_scores = all_cls_scores[-1]
        kpt_preds = all_kpt_preds[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            kpt_pred = kpt_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            # TODO: only support single image test
            # memory_i = memory[:, img_id, :]
            # mlvl_mask = mlvl_masks[img_id]
            proposals = self._get_bboxes_single(cls_score, kpt_pred, img_shape,
                                                scale_factor, memory,
                                                mlvl_masks, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           kpt_pred,
                           img_shape,
                           scale_factor,
                           memory,
                           mlvl_masks,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            kpt_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (x_{i}, y_{i}) and
                shape [num_query, K*2].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5],
                    where the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with
                    shape [num_query].
                - det_kpts: Predicted keypoints with shape [num_query, K, 3].
        """
        assert len(cls_score) == len(kpt_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = F.sigmoid(cls_score)
            scores, indexs = cls_score.reshape([-1]).topk(max_per_img)
            det_labels = indexs % self.num_classes
            bbox_index = indexs // self.num_classes
            kpt_pred = kpt_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, axis=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            kpt_pred = kpt_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        # ----- results after pose decoder -----
        # det_kpts = kpt_pred.reshape((kpt_pred.shape[0], -1, 2))

        # ----- results after joint decoder (default) -----
        # import time
        # start = time.time()
        refine_targets = (kpt_pred, None, None, paddle.ones_like(kpt_pred))
        refine_outputs = self.forward_refine(memory, mlvl_masks, refine_targets,
                                             None, None)
        # end = time.time()
        # print(f'refine time: {end - start:.6f}')
        det_kpts = refine_outputs[-1]

        det_kpts[..., 0] = det_kpts[..., 0] * img_shape[1]
        det_kpts[..., 1] = det_kpts[..., 1] * img_shape[0]
        det_kpts[..., 0].clip_(min=0, max=img_shape[1])
        det_kpts[..., 1].clip_(min=0, max=img_shape[0])
        if rescale:
            det_kpts /= paddle.to_tensor(
                scale_factor[:2],
                dtype=det_kpts.dtype).unsqueeze(0).unsqueeze(0)

        # use circumscribed rectangle box of keypoints as det bboxes
        x1 = det_kpts[..., 0].min(axis=1, keepdim=True)
        y1 = det_kpts[..., 1].min(axis=1, keepdim=True)
        x2 = det_kpts[..., 0].max(axis=1, keepdim=True)
        y2 = det_kpts[..., 1].max(axis=1, keepdim=True)
        det_bboxes = paddle.concat([x1, y1, x2, y2], axis=1)
        det_bboxes = paddle.concat((det_bboxes, scores.unsqueeze(1)), -1)

        det_kpts = paddle.concat(
            (det_kpts, paddle.ones(
                det_kpts[..., :1].shape, dtype=det_kpts.dtype)),
            axis=2)

        return det_bboxes, det_labels, det_kpts

    def simple_test(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[paddle.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
                3-tuple. The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,). The third item is ``kpts`` with shape
                (n, K, 3), in [p^{1}_x, p^{1}_y, p^{1}_v, p^{K}_x, p^{K}_y,
                p^{K}_v] format.
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    def get_loss(self, boxes, scores, gt_bbox, gt_class, prior_boxes):
        return self.loss(boxes, scores, gt_bbox, gt_class, prior_boxes)
