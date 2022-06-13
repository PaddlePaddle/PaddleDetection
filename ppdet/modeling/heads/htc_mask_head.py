# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
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
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingUniform, KaimingNormal
from paddle import ParamAttr

from ppdet.core.workspace import register, create
from ppdet.modeling.layers import ConvNormLayer
from ..shape_spec import ShapeSpec

from .roi_extractor import RoIAlign

__all__ = [
    'HybridTaskMaskFeatSub', 'HybridTaskMaskFeat', 'HybridTaskMaskHead',
    'FusedSemanticHead'
]


@register
class HybridTaskMaskFeatSub(nn.Layer):
    """
    Feature extraction in Mask head
    Args:
        in_channel (int): Input channels
        out_channel (int): Output channels
        num_convs (int): The number of conv layers, default 4
        norm_type (string | None): Norm type, bn, gn, sync_bn are available,
            default None
    """

    def __init__(self,
                 in_channel=256,
                 out_channel=256,
                 num_convs=4,
                 norm_type=None,
                 num_classes=80,
                 with_conv_res=True):
        super(HybridTaskMaskFeatSub, self).__init__()
        self.num_convs = num_convs
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.norm_type = norm_type
        self.num_classes = num_classes
        self.with_conv_res = with_conv_res
        fan_conv = out_channel * 3 * 3
        fan_deconv = out_channel * 2 * 2

        mask_conv = nn.Sequential()
        if norm_type == 'gn':
            for i in range(self.num_convs):
                conv_name = '{}.conv'.format(i)
                mask_conv.add_sublayer(
                    conv_name,
                    ConvNormLayer(
                        ch_in=in_channel if i == 0 else out_channel,
                        ch_out=out_channel,
                        filter_size=3,
                        stride=1,
                        norm_type=self.norm_type,
                        initializer=KaimingNormal(fan_in=fan_conv),
                        skip_quant=True))
                mask_conv.add_sublayer(conv_name + 'act', nn.ReLU())
        else:
            for i in range(self.num_convs):
                conv_name = '{}.conv'.format(i)
                conv = nn.Conv2D(
                    in_channels=in_channel if i == 0 else out_channel,
                    out_channels=out_channel,
                    kernel_size=3,
                    padding=1,
                    weight_attr=paddle.ParamAttr(
                        initializer=KaimingNormal(fan_in=fan_conv)))
                conv.skip_quant = True
                mask_conv.add_sublayer(conv_name, conv)
                mask_conv.add_sublayer(conv_name + 'act', nn.ReLU())
        self.convs = mask_conv

        self.upsample = nn.Conv2DTranspose(
            in_channels=self.in_channel,
            out_channels=self.out_channel,
            kernel_size=2,
            stride=2,
            weight_attr=paddle.ParamAttr(
                initializer=KaimingNormal(fan_in=fan_deconv)))
        self.upsampleact = nn.ReLU()

        self.conv_logits = nn.Conv2D(
            in_channels=self.out_channel,
            out_channels=self.num_classes,
            kernel_size=1,
            weight_attr=paddle.ParamAttr(initializer=KaimingNormal(
                fan_in=self.num_classes)))
        self.conv_logits.skip_quant = True

        if self.with_conv_res:
            conv = nn.Conv2D(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=1,
                weight_attr=paddle.ParamAttr(
                    initializer=KaimingNormal(fan_in=fan_conv)))
            conv.skip_quant = True
            mask_conv = nn.Sequential()
            mask_conv.add_sublayer('conv', conv)
            mask_conv.add_sublayer('conv' + 'act', nn.ReLU())
            self.conv_res = mask_conv

    @classmethod
    def from_config(cls, cfg, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channel': input_shape.channels, }

    def out_channels(self):
        # return self.out_channel
        return self.num_classes

    def forward(self,
                feats,
                res_feat=None,
                return_logits=True,
                return_feat=True):
        if res_feat is not None:
            assert self.with_conv_res
            res_feat = self.conv_res(res_feat)
            feats = feats + res_feat
        x = self.convs(feats)
        res_feat = x
        outs = []
        if return_logits:
            x = self.upsample(x)
            x = self.upsampleact(x)
            x = self.conv_logits(x)
            outs.append(x)
        if return_feat:
            outs.append(res_feat)
        return outs if len(outs) > 1 else outs[0]


@register
class HybridTaskMaskFeat(nn.Layer):
    """
    Feature extraction in Mask head
    Args:
        in_channel (int): Input channels
        out_channel (int): Output channels
        num_convs (int): The number of conv layers, default 4
        norm_type (string | None): Norm type, bn, gn, sync_bn are available,
            default None
    """

    def __init__(self,
                 in_channel=256,
                 out_channel=256,
                 num_convs=4,
                 norm_type=None,
                 num_cascade_stages=3):
        super(HybridTaskMaskFeat, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_cascade_stages = num_cascade_stages

        self.upsample = []
        for stage in range(self.num_cascade_stages):
            head_per_stage = self.add_sublayer(
                str(stage),
                HybridTaskMaskFeatSub(
                    in_channel,
                    out_channel,
                    num_convs,
                    norm_type,
                    with_conv_res=(stage != 0)))
            self.upsample.append(head_per_stage)

    @classmethod
    def from_config(cls, cfg, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channel': input_shape.channels, }

    def out_channels(self):
        return self.out_channel

    def forward_train(self, feats, stage=0):
        last_feat = None
        for i in range(stage):
            last_feat = self.upsample[i](feats, last_feat, return_logits=False)
        mask_logits = self.upsample[stage](feats, last_feat, return_feat=False)
        # return self.upsample[stage](feats)
        return mask_logits

    def forward_test(self, feats, stage=0):
        result = []
        last_feat = None
        for i in range(stage):
            mask_logits, last_feat = self.upsample[i](feats, last_feat)
            result.append(mask_logits)
        mask_logits = self.upsample[stage](feats, last_feat, return_feat=False)
        # return self.upsample[stage](feats)
        result.append(mask_logits)
        return result

    def forward(self, feats, stage=0):
        if self.training:
            return self.forward_train(feats, stage)
        else:
            return self.forward_test(feats, stage)


@register
class HybridTaskMaskHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['mask_assigner']
    """
    RCNN mask head
    Args:
        head (nn.Layer): Extract feature in mask head
        roi_extractor (object): The module of RoI Extractor
        mask_assigner (object): The module of Mask Assigner, 
            label and sample the mask
        num_classes (int): The number of classes
        share_bbox_feat (bool): Whether to share the feature from bbox head,
            default false
    """

    def __init__(self,
                 head,
                 roi_extractor=RoIAlign().__dict__,
                 semantic_roi_extractor=RoIAlign().__dict__,
                 mask_assigner='MaskAssigner',
                 num_classes=80,
                 share_bbox_feat=False,
                 num_cascade_stages=3):
        super(HybridTaskMaskHead, self).__init__()
        self.num_classes = num_classes
        self.num_cascade_stages = num_cascade_stages

        self.roi_extractor = roi_extractor
        if isinstance(roi_extractor, dict):
            self.roi_extractor = RoIAlign(**roi_extractor)
        self.semantic_roi_extractor = semantic_roi_extractor
        if isinstance(semantic_roi_extractor, dict):
            self.semantic_roi_extractor = RoIAlign(**semantic_roi_extractor)
        self.head = head
        self.in_channels = head.out_channels()
        self.mask_assigner = mask_assigner
        self.share_bbox_feat = share_bbox_feat
        self.bbox_head = None

    @classmethod
    def from_config(cls, cfg, input_shape):
        roi_pooler = cfg['roi_extractor']
        assert isinstance(roi_pooler, dict)
        kwargs = RoIAlign.from_config(cfg, input_shape)
        roi_pooler.update(kwargs)
        kwargs = {'input_shape': input_shape}
        head = create(cfg['head'], **kwargs)
        res = {
            'roi_extractor': roi_pooler,
            'head': head,
        }
        if 'semantic_roi_extractor' in cfg:
            semantic_roi_extractor = cfg['semantic_roi_extractor']
            assert isinstance(semantic_roi_extractor, dict)
            semantic_roi_extractor.update({'spatial_scale': [0.125]})
            roi_pooler.update({'spatial_scale': [0.25, 0.125, 0.0625, 0.03125]})
            res['semantic_roi_extractor'] = semantic_roi_extractor
        return res

    def get_loss(self, mask_logits, mask_label, mask_target, mask_weight):
        mask_label = F.one_hot(mask_label, self.num_classes).unsqueeze([2, 3])
        mask_label = paddle.expand_as(mask_label, mask_logits)
        mask_label.stop_gradient = True
        mask_pred = paddle.gather_nd(mask_logits, paddle.nonzero(mask_label))
        shape = mask_logits.shape
        mask_pred = paddle.reshape(mask_pred, [shape[0], shape[2], shape[3]])

        mask_target = mask_target.cast('float32')
        mask_weight = mask_weight.unsqueeze([1, 2])
        loss_mask = F.binary_cross_entropy_with_logits(
            mask_pred, mask_target, weight=mask_weight, reduction="mean")
        return loss_mask

    def forward_train(self,
                      body_feats,
                      rois,
                      rois_num,
                      inputs,
                      targets,
                      bbox_feat,
                      semantic_feats,
                      stage=0):
        """
        body_feats (list[Tensor]): Multi-level backbone features
        rois (list[Tensor]): Proposals for each batch with shape [N, 4]
        rois_num (Tensor): The number of proposals for each batch
        inputs (dict): ground truth info
        """
        tgt_labels, _, tgt_gt_inds = targets
        rois, rois_num, tgt_classes, tgt_masks, mask_index, tgt_weights = self.mask_assigner(
            rois, tgt_labels, tgt_gt_inds, inputs)

        if self.share_bbox_feat:
            rois_feat = paddle.gather(bbox_feat, mask_index)
        else:
            rois_feat = self.roi_extractor(body_feats, rois, rois_num)

        if semantic_feats is not None:
            semantic_rois_feat = self.semantic_roi_extractor([semantic_feats],
                                                             rois, rois_num)
            if semantic_rois_feat.shape[-2:] != rois_feat.shape[-2:]:
                semantic_rois_feat = F.adaptive_avg_pool2d(semantic_rois_feat,
                                                           rois_feat.shape[-2:])
            rois_feat += semantic_rois_feat
        # mask_feat = self.head(rois_feat, stage)
        # mask_logits = self.mask_fcn_logits[stage](mask_feat)

        mask_logits = self.head(rois_feat, stage)

        loss_mask = self.get_loss(mask_logits, tgt_classes, tgt_masks,
                                  tgt_weights)
        # return {'loss_mask': loss_mask}
        return {"loss_mask_stage{}".format(stage): loss_mask}

    def forward_test(self,
                     body_feats,
                     rois,
                     rois_num,
                     scale_factor,
                     feat_func=None,
                     semantic_feats=None,
                     stage=2):
        """
        body_feats (list[Tensor]): Multi-level backbone features
        rois (Tensor): Prediction from bbox head with shape [N, 6]
        rois_num (Tensor): The number of prediction for each batch
        scale_factor (Tensor): The scale factor from origin size to input size
        """
        if rois.shape[0] == 0:
            mask_out = paddle.full([1, 1, 1, 1], -1)
        else:
            bbox = [rois[:, 2:]]
            labels = rois[:, 0].cast('int32')
            rois_feat = self.roi_extractor(body_feats, bbox, rois_num)

            if semantic_feats is not None:
                semantic_rois_feat = self.semantic_roi_extractor(
                    [semantic_feats], bbox, rois_num)
                if semantic_rois_feat.shape[-2:] != rois_feat.shape[-2:]:
                    semantic_rois_feat = F.adaptive_avg_pool2d(
                        semantic_rois_feat, rois_feat.shape[-2:])
                rois_feat += semantic_rois_feat

            if self.share_bbox_feat:
                assert feat_func is not None
                rois_feat = feat_func(rois_feat)

            # mask_feat = self.head(rois_feat)
            # mask_logit = self.mask_fcn_logits(mask_feat)
            mask_logit = self.head(rois_feat, stage)
            mask_logit = paddle.to_tensor(
                [F.sigmoid(logit) for logit in mask_logit])
            mask_logit = paddle.mean(mask_logit, axis=0)

            mask_num_class = mask_logit[0].shape[1]
            if mask_num_class == 1:
                mask_out = F.sigmoid(mask_logit)
            else:
                num_masks = mask_logit.shape[0]
                mask_out = []
                # TODO: need to optimize gather
                for i in range(mask_logit.shape[0]):
                    pred_masks = paddle.unsqueeze(
                        mask_logit[i, :, :, :], axis=0)
                    mask = paddle.gather(pred_masks, labels[i], axis=1)
                    mask_out.append(mask)
                mask_out = paddle.concat(mask_out)
                # mask_out = F.sigmoid(paddle.concat(mask_out))
        return mask_out

    def forward(self,
                body_feats,
                rois,
                rois_num,
                inputs,
                targets=None,
                bbox_feat=None,
                feat_func=None,
                semantic_feats=None,
                stage=0):
        if self.training:
            return self.forward_train(
                body_feats,
                rois,
                rois_num,
                inputs,
                targets,
                bbox_feat,
                semantic_feats=semantic_feats,
                stage=stage)
        else:
            im_scale = inputs['scale_factor']
            return self.forward_test(
                body_feats,
                rois,
                rois_num,
                im_scale,
                feat_func,
                semantic_feats=semantic_feats,
                stage=stage)


@register
class FusedSemanticHead(nn.Layer):
    def __init__(self, semantic_num_class=183, loss_weight=0.2):
        super(FusedSemanticHead, self).__init__()

        self.semantic_num_class = semantic_num_class
        self.loss_weight = loss_weight

        self.lateral_convs = []
        self.convs = []

        for i in range(5):
            lateral_name = 'lateral_convs.{}.conv'.format(i)
            lateral = self.add_sublayer(
                lateral_name,
                nn.Conv2D(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=1,
                    weight_attr=ParamAttr(initializer=KaimingUniform())))
            self.lateral_convs.append(lateral)

        for i in range(4):
            fpn_name = 'convs.{}.conv'.format(i)
            fpn_conv = self.add_sublayer(
                fpn_name,
                nn.Conv2D(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    padding=1,
                    weight_attr=ParamAttr(initializer=KaimingUniform())))
            self.convs.append(fpn_conv)

        self.conv_embedding = self.add_sublayer(
            'conv_embedding',
            nn.Conv2D(
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                weight_attr=ParamAttr(initializer=KaimingUniform())))

        self.conv_logits = self.add_sublayer(
            'conv_logits',
            nn.Conv2D(
                in_channels=256,
                out_channels=self.semantic_num_class,
                kernel_size=1,
                weight_attr=ParamAttr(initializer=KaimingUniform())))

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    @classmethod
    def from_config(cls, cfg, input_shape):
        s = input_shape
        s = s[0] if isinstance(s, (list, tuple)) else s
        return {'in_channel': s.channels}

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.out_channel, )]

    def forward(self, body_feats):
        x = F.relu(self.lateral_convs[1](body_feats[1]))
        fused_size = tuple(x.shape[-2:])
        for i, feat in enumerate(body_feats):
            if i != 1:
                feat = F.interpolate(
                    feat, size=fused_size, mode='bilinear', align_corners=True)
                x += F.relu(self.lateral_convs[i](feat))

        for i in range(4):
            x = F.relu(self.convs[i](x))

        mask_pred = self.conv_logits(x)
        x = F.relu(self.conv_embedding(x))
        return mask_pred, x

    def loss(self, mask_pred, labels):
        labels = paddle.transpose(labels, perm=[0, 2, 3, 1]).astype('int64')
        mask_pred = paddle.transpose(mask_pred, perm=[0, 2, 3, 1])
        loss_semantic_seg = self.criterion(mask_pred, labels)
        loss_semantic_seg *= self.loss_weight
        return loss_semantic_seg