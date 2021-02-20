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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register, create
from ppdet.modeling import ops
from ppdet.modeling.layers import ConvNormLayer

from .roi_extractor import RoIAlign


@register
class MaskFeat(nn.Layer):
    def __init__(self,
                 num_convs=4,
                 in_channels=256,
                 out_channels=256,
                 norm_type=None):
        super(MaskFeat, self).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
        fan_conv = out_channels * 3 * 3
        fan_deconv = out_channels * 2 * 2

        mask_conv = nn.Sequential()
        if norm_type == 'gn':
            for i in range(self.num_convs):
                conv_name = 'mask_inter_feat_{}'.format(i + 1)
                mask_conv.add_sublayer(
                    conv_name,
                    ConvNormLayer(
                        ch_in=in_channels if i == 0 else out_channels,
                        ch_out=out_channels,
                        filter_size=3,
                        stride=1,
                        norm_type=self.norm_type,
                        norm_name=conv_name + '_norm',
                        initializer=KaimingNormal(fan_in=fan_conv),
                        name=conv_name))
                mask_conv.add_sublayer(conv_name + 'act', nn.ReLU())
        else:
            for i in range(self.num_convs):
                conv_name = 'mask_inter_feat_{}'.format(i + 1)
                mask_conv.add_sublayer(
                    conv_name,
                    nn.Conv2D(
                        in_channels=in_channels if i == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        weight_attr=paddle.ParamAttr(
                            initializer=KaimingNormal(fan_in=fan_conv))))
                mask_conv.add_sublayer(conv_name + 'act', nn.ReLU())
        mask_conv.add_sublayer(
            'conv5_mask',
            nn.Conv2DTranspose(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=2,
                stride=2,
                weight_attr=paddle.ParamAttr(
                    initializer=KaimingNormal(fan_in=fan_deconv))))
        mask_conv.add_sublayer('conv5_mask' + 'act', nn.ReLU())
        self.upsample = mask_conv

    @classmethod
    def from_config(cls, cfg, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channels': input_shape.channels, }

    def out_channel(self):
        return self.out_channels

    def forward(self, feats):
        return self.upsample(feats)


@register
class MaskHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['mask_assigner']

    def __init__(self,
                 head,
                 roi_extractor=RoIAlign().__dict__,
                 mask_assigner='MaskAssigner',
                 num_classes=80,
                 share_bbox_feat=False):
        super(MaskHead, self).__init__()
        self.num_classes = num_classes

        self.roi_extractor = roi_extractor
        if isinstance(roi_extractor, dict):
            self.roi_extractor = RoIAlign(**roi_extractor)
        self.head = head
        self.in_channels = head.out_channel()
        self.mask_assigner = mask_assigner
        self.share_bbox_feat = share_bbox_feat
        self.bbox_head = None

        self.mask_fcn_logits = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            kernel_size=1,
            weight_attr=paddle.ParamAttr(initializer=KaimingNormal(
                fan_in=self.num_classes)))

    @classmethod
    def from_config(cls, cfg, input_shape):
        roi_pooler = cfg['roi_extractor']
        assert isinstance(roi_pooler, dict)
        kwargs = RoIAlign.from_config(cfg, input_shape)
        roi_pooler.update(kwargs)
        kwargs = {'input_shape': input_shape}
        head = create(cfg['head'], **kwargs)
        return {
            'roi_extractor': roi_pooler,
            'head': head,
        }

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

    def forward_train(self, body_feats, rois, rois_num, inputs, targets,
                      bbox_feat):
        """
        body_feats (list[Tensor]): Multi-level backbone features
        rois (list[Tensor]): Proposals for each batch with shape [N, 4]
        rois_num (Tensor): The number of proposals for each batch
        inputs (dict): ground truth info
        """
        #assert self.bbox_head
        tgt_labels, _, tgt_gt_inds = targets
        rois, rois_num, tgt_classes, tgt_masks, mask_index, tgt_weights = self.mask_assigner(
            rois, tgt_labels, tgt_gt_inds, inputs)

        if self.share_bbox_feat:
            rois_feat = paddle.gather(bbox_feat, mask_index)
        else:
            rois_feat = self.roi_extractor(body_feats, rois, rois_num)
        mask_feat = self.head(rois_feat)
        mask_logits = self.mask_fcn_logits(mask_feat)

        loss_mask = self.get_loss(mask_logits, tgt_classes, tgt_masks,
                                  tgt_weights)
        return {'loss_mask': loss_mask}

    def forward_test(self,
                     body_feats,
                     rois,
                     rois_num,
                     scale_factor,
                     feat_func=None):
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
            if self.share_bbox_feat:
                assert feat_func is not None
                rois_feat = feat_func(rois_feat)

            mask_feat = self.head(rois_feat)
            mask_logit = self.mask_fcn_logits(mask_feat)
            mask_num_class = mask_logit.shape[1]
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
                mask_out = F.sigmoid(paddle.concat(mask_out))
        return mask_out

    def forward(self,
                body_feats,
                rois,
                rois_num,
                inputs,
                targets=None,
                bbox_feat=None,
                feat_func=None):
        if self.training:
            return self.forward_train(body_feats, rois, rois_num, inputs,
                                      targets, bbox_feat)
        else:
            im_scale = inputs['scale_factor']
            return self.forward_test(body_feats, rois, rois_num, im_scale,
                                     feat_func)
