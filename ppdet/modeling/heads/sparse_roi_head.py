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

# This code is referenced from: https://github.com/open-mmlab/mmdetection

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import paddle
from paddle import nn

from ppdet.core.workspace import register
from ppdet.modeling import initializer as init
from .roi_extractor import RoIAlign
from ..bbox_utils import delta2bbox_v2
from ..cls_utils import _get_class_default_kwargs
from ..layers import MultiHeadAttention

__all__ = ['SparseRoIHead', 'DIIHead', 'DynamicMaskHead']


class DynamicConv(nn.Layer):
    def __init__(self,
                 in_channels=256,
                 feature_channels=64,
                 out_channels=None,
                 roi_resolution=7,
                 with_proj=True):
        super(DynamicConv, self).__init__()

        self.in_channels = in_channels
        self.feature_channels = feature_channels
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.in_channels * self.feature_channels
        self.num_params_out = self.out_channels * self.feature_channels
        self.dynamic_layer = nn.Linear(self.in_channels,
                                       self.num_params_in + self.num_params_out)

        self.norm_in = nn.LayerNorm(self.feature_channels)
        self.norm_out = nn.LayerNorm(self.out_channels)

        self.activation = nn.ReLU()

        self.with_proj = with_proj
        if self.with_proj:
            num_output = self.out_channels * roi_resolution**2
            self.fc_layer = nn.Linear(num_output, self.out_channels)
            self.fc_norm = nn.LayerNorm(self.out_channels)

    def forward(self, param_feature, input_feature):
        input_feature = input_feature.flatten(2).transpose([2, 0, 1])
        input_feature = input_feature.transpose([1, 0, 2])

        parameters = self.dynamic_layer(param_feature)

        param_in = parameters[:, :self.num_params_in].reshape(
            [-1, self.in_channels, self.feature_channels])
        param_out = parameters[:, -self.num_params_out:].reshape(
            [-1, self.feature_channels, self.out_channels])

        features = paddle.bmm(input_feature, param_in)
        features = self.norm_in(features)
        features = self.activation(features)

        features = paddle.bmm(features, param_out)
        features = self.norm_out(features)
        features = self.activation(features)

        if self.with_proj:
            features = features.flatten(1)
            features = self.fc_layer(features)
            features = self.fc_norm(features)
            features = self.activation(features)

        return features


class FFN(nn.Layer):
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=2048,
                 num_fcs=2,
                 ffn_drop=0.0,
                 add_identity=True):
        super(FFN, self).__init__()

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    nn.ReLU(), nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

        self.add_identity = add_identity

    def forward(self, x):
        identity = x
        out = self.layers(x)
        if not self.add_identity:
            return out
        else:
            return out + identity


@register
class DynamicMaskHead(nn.Layer):
    __shared__ = ['num_classes', 'proposal_embedding_dim', 'norm_type']

    def __init__(self,
                 num_classes=80,
                 proposal_embedding_dim=256,
                 dynamic_feature_channels=64,
                 roi_resolution=14,
                 num_convs=4,
                 conv_kernel_size=3,
                 conv_channels=256,
                 upsample_method='deconv',
                 upsample_scale_factor=2,
                 norm_type='bn'):
        super(DynamicMaskHead, self).__init__()

        self.d_model = proposal_embedding_dim

        self.instance_interactive_conv = DynamicConv(
            self.d_model,
            dynamic_feature_channels,
            roi_resolution=roi_resolution,
            with_proj=False)

        self.convs = nn.LayerList()
        for i in range(num_convs):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2D(
                        self.d_model if i == 0 else conv_channels,
                        conv_channels,
                        conv_kernel_size,
                        padding='same',
                        bias_attr=False),
                    nn.BatchNorm2D(conv_channels),
                    nn.ReLU()))
        if norm_type == 'sync_bn':
            self.convs = nn.SyncBatchNorm.convert_sync_batchnorm(self.convs)

        self.upsample_method = upsample_method
        if upsample_method is None:
            self.upsample = None
        elif upsample_method == 'deconv':
            self.upsample = nn.Conv2DTranspose(
                conv_channels if num_convs > 0 else self.d_model,
                conv_channels,
                upsample_scale_factor,
                stride=upsample_scale_factor)
            self.relu = nn.ReLU()
        else:
            self.upsample = nn.Upsample(None, upsample_scale_factor)

        cls_in_channels = conv_channels if num_convs > 0 else self.d_model
        cls_in_channels = conv_channels if upsample_method == 'deconv' else cls_in_channels
        self.conv_cls = nn.Conv2D(cls_in_channels, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

        init.constant_(self.conv_cls.bias, 0.)

    def forward(self, roi_features, attn_features):
        attn_features = attn_features.reshape([-1, self.d_model])
        attn_features_iic = self.instance_interactive_conv(attn_features,
                                                           roi_features)

        x = attn_features_iic.transpose([0, 2, 1]).reshape(roi_features.shape)

        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_cls(x)
        return mask_pred


@register
class DIIHead(nn.Layer):
    __shared__ = ['num_classes', 'proposal_embedding_dim']

    def __init__(self,
                 num_classes=80,
                 proposal_embedding_dim=256,
                 feedforward_channels=2048,
                 dynamic_feature_channels=64,
                 roi_resolution=7,
                 num_attn_heads=8,
                 dropout=0.0,
                 num_ffn_fcs=2,
                 num_cls_fcs=1,
                 num_reg_fcs=3):
        super(DIIHead, self).__init__()

        self.num_classes = num_classes
        self.d_model = proposal_embedding_dim

        self.attention = MultiHeadAttention(self.d_model, num_attn_heads,
                                            dropout)
        self.attention_norm = nn.LayerNorm(self.d_model)

        self.instance_interactive_conv = DynamicConv(
            self.d_model,
            dynamic_feature_channels,
            roi_resolution=roi_resolution,
            with_proj=True)
        self.instance_interactive_conv_dropout = nn.Dropout(dropout)
        self.instance_interactive_conv_norm = nn.LayerNorm(self.d_model)

        self.ffn = FFN(self.d_model, feedforward_channels, num_ffn_fcs, dropout)
        self.ffn_norm = nn.LayerNorm(self.d_model)

        self.cls_fcs = nn.LayerList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(
                    self.d_model, self.d_model, bias_attr=False))
            self.cls_fcs.append(nn.LayerNorm(self.d_model))
            self.cls_fcs.append(nn.ReLU())
        self.fc_cls = nn.Linear(self.d_model, self.num_classes)

        self.reg_fcs = nn.LayerList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(
                    self.d_model, self.d_model, bias_attr=False))
            self.reg_fcs.append(nn.LayerNorm(self.d_model))
            self.reg_fcs.append(nn.ReLU())
        self.fc_reg = nn.Linear(self.d_model, 4)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

        bias_init = init.bias_init_with_prob(0.01)
        init.constant_(self.fc_cls.bias, bias_init)

    def forward(self, roi_features, proposal_features):
        N, num_proposals = proposal_features.shape[:2]

        proposal_features = proposal_features + self.attention(
            proposal_features)
        attn_features = self.attention_norm(proposal_features)

        proposal_features = attn_features.reshape([-1, self.d_model])
        proposal_features_iic = self.instance_interactive_conv(
            proposal_features, roi_features)
        proposal_features = proposal_features + self.instance_interactive_conv_dropout(
            proposal_features_iic)
        obj_features = self.instance_interactive_conv_norm(proposal_features)

        obj_features = self.ffn(obj_features)
        obj_features = self.ffn_norm(obj_features)

        cls_feature = obj_features.clone()
        reg_feature = obj_features.clone()

        for cls_layer in self.cls_fcs:
            cls_feature = cls_layer(cls_feature)
        class_logits = self.fc_cls(cls_feature)
        for reg_layer in self.reg_fcs:
            reg_feature = reg_layer(reg_feature)
        bbox_deltas = self.fc_reg(reg_feature)

        class_logits = class_logits.reshape(
            [N, num_proposals, self.num_classes])
        bbox_deltas = bbox_deltas.reshape([N, num_proposals, 4])
        obj_features = obj_features.reshape([N, num_proposals, self.d_model])

        return class_logits, bbox_deltas, obj_features, attn_features

    @staticmethod
    def refine_bboxes(proposal_bboxes, bbox_deltas):
        pred_bboxes = delta2bbox_v2(
            bbox_deltas.reshape([-1, 4]),
            proposal_bboxes.reshape([-1, 4]),
            delta_mean=[0.0, 0.0, 0.0, 0.0],
            delta_std=[0.5, 0.5, 1.0, 1.0],
            ctr_clip=None)
        return pred_bboxes.reshape(proposal_bboxes.shape)


@register
class SparseRoIHead(nn.Layer):
    __inject__ = ['bbox_head', 'mask_head', 'loss_func']

    def __init__(self,
                 num_stages=6,
                 bbox_roi_extractor=_get_class_default_kwargs(RoIAlign),
                 mask_roi_extractor=_get_class_default_kwargs(RoIAlign),
                 bbox_head='DIIHead',
                 mask_head='DynamicMaskHead',
                 loss_func='QueryInstLoss'):
        super(SparseRoIHead, self).__init__()

        self.num_stages = num_stages

        self.bbox_roi_extractor = bbox_roi_extractor
        self.mask_roi_extractor = mask_roi_extractor
        if isinstance(bbox_roi_extractor, dict):
            self.bbox_roi_extractor = RoIAlign(**bbox_roi_extractor)
        if isinstance(mask_roi_extractor, dict):
            self.mask_roi_extractor = RoIAlign(**mask_roi_extractor)

        self.bbox_heads = nn.LayerList(
            [copy.deepcopy(bbox_head) for _ in range(num_stages)])
        self.mask_heads = nn.LayerList(
            [copy.deepcopy(mask_head) for _ in range(num_stages)])

        self.loss_helper = loss_func

    @classmethod
    def from_config(cls, cfg, input_shape):
        bbox_roi_extractor = cfg['bbox_roi_extractor']
        mask_roi_extractor = cfg['mask_roi_extractor']
        assert isinstance(bbox_roi_extractor, dict)
        assert isinstance(mask_roi_extractor, dict)

        kwargs = RoIAlign.from_config(cfg, input_shape)
        bbox_roi_extractor.update(kwargs)
        mask_roi_extractor.update(kwargs)

        return {
            'bbox_roi_extractor': bbox_roi_extractor,
            'mask_roi_extractor': mask_roi_extractor
        }

    @staticmethod
    def get_roi_features(features, bboxes, roi_extractor):
        rois_list = [
            bboxes[i] for i in range(len(bboxes)) if len(bboxes[i]) > 0
        ]
        rois_num = paddle.to_tensor(
            [len(bboxes[i]) for i in range(len(bboxes))], dtype='int32')

        pos_ids = paddle.cast(rois_num, dtype='bool')
        if pos_ids.sum() != len(rois_num):
            rois_num = rois_num[pos_ids]
            features = [features[i][pos_ids] for i in range(len(features))]

        return roi_extractor(features, rois_list, rois_num)

    def _forward_train(self, body_feats, pro_bboxes, pro_feats, targets):
        all_stage_losses = {}
        for stage in range(self.num_stages):
            bbox_head = self.bbox_heads[stage]
            mask_head = self.mask_heads[stage]

            roi_feats = self.get_roi_features(body_feats, pro_bboxes,
                                              self.bbox_roi_extractor)
            class_logits, bbox_deltas, pro_feats, attn_feats = bbox_head(
                roi_feats, pro_feats)
            bbox_pred = self.bbox_heads[stage].refine_bboxes(pro_bboxes,
                                                             bbox_deltas)

            indices = self.loss_helper.matcher({
                'pred_logits': class_logits.detach(),
                'pred_boxes': bbox_pred.detach()
            }, targets)
            avg_factor = paddle.to_tensor(
                [sum(len(tgt['labels']) for tgt in targets)], dtype='float32')
            if paddle.distributed.get_world_size() > 1:
                paddle.distributed.all_reduce(avg_factor)
                avg_factor /= paddle.distributed.get_world_size()
            avg_factor = paddle.clip(avg_factor, min=1.)

            loss_classes = self.loss_helper.loss_classes(class_logits, targets,
                                                         indices, avg_factor)
            if sum(len(v['labels']) for v in targets) == 0:
                loss_bboxes = {
                    'loss_bbox': paddle.to_tensor([0.]),
                    'loss_giou': paddle.to_tensor([0.])
                }
                loss_masks = {'loss_mask': paddle.to_tensor([0.])}
            else:
                loss_bboxes = self.loss_helper.loss_bboxes(bbox_pred, targets,
                                                           indices, avg_factor)

                pos_attn_feats = paddle.concat([
                    paddle.gather(
                        src, src_idx, axis=0)
                    for src, (src_idx, _) in zip(attn_feats, indices)
                ])
                pos_bbox_pred = [
                    paddle.gather(
                        src, src_idx, axis=0)
                    for src, (src_idx, _) in zip(bbox_pred.detach(), indices)
                ]
                pos_roi_feats = self.get_roi_features(body_feats, pos_bbox_pred,
                                                      self.mask_roi_extractor)
                mask_logits = mask_head(pos_roi_feats, pos_attn_feats)
                loss_masks = self.loss_helper.loss_masks(
                    pos_bbox_pred, mask_logits, targets, indices, avg_factor)

            for loss in [loss_classes, loss_bboxes, loss_masks]:
                for key in loss.keys():
                    all_stage_losses[f'stage{stage}_{key}'] = loss[key]

            pro_bboxes = bbox_pred.detach()

        return all_stage_losses

    def _forward_test(self, body_feats, pro_bboxes, pro_feats):
        for stage in range(self.num_stages):
            roi_feats = self.get_roi_features(body_feats, pro_bboxes,
                                              self.bbox_roi_extractor)
            class_logits, bbox_deltas, pro_feats, attn_feats = self.bbox_heads[
                stage](roi_feats, pro_feats)
            bbox_pred = self.bbox_heads[stage].refine_bboxes(pro_bboxes,
                                                             bbox_deltas)

            pro_bboxes = bbox_pred.detach()

        roi_feats = self.get_roi_features(body_feats, bbox_pred,
                                          self.mask_roi_extractor)
        mask_logits = self.mask_heads[stage](roi_feats, attn_feats)

        return {
            'class_logits': class_logits,
            'bbox_pred': bbox_pred,
            'mask_logits': mask_logits
        }

    def forward(self,
                body_features,
                proposal_bboxes,
                proposal_features,
                targets=None):
        if self.training:
            return self._forward_train(body_features, proposal_bboxes,
                                       proposal_features, targets)
        else:
            return self._forward_test(body_features, proposal_bboxes,
                                      proposal_features)
