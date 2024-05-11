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

from ppdet.core.workspace import register, create
from ppdet.modeling.layers import ConvNormLayer
from .roi_extractor import RoIAlign
from ..cls_utils import _get_class_default_kwargs


@register
class MaskFeat(nn.Layer):
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
                 norm_type=None):
        super(MaskFeat, self).__init__()
        self.num_convs = num_convs
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.norm_type = norm_type
        fan_conv = out_channel * 3 * 3
        fan_deconv = out_channel * 2 * 2

        mask_conv = nn.Sequential()
        if norm_type == 'gn':
            for i in range(self.num_convs):
                conv_name = 'mask_inter_feat_{}'.format(i + 1)
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
                conv_name = 'mask_inter_feat_{}'.format(i + 1)
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
        mask_conv.add_sublayer(
            'conv5_mask',
            nn.Conv2DTranspose(
                in_channels=self.out_channel if num_convs > 0 else self.in_channel,
                out_channels=self.out_channel,
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
        return {'in_channel': input_shape.channels, }

    def out_channels(self):
        return self.out_channel

    def forward(self, feats):
        return self.upsample(feats)


@register
class MaskHead(nn.Layer):
    __shared__ = ['num_classes', 'export_onnx']
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
                 roi_extractor=_get_class_default_kwargs(RoIAlign),
                 mask_assigner='MaskAssigner',
                 num_classes=80,
                 share_bbox_feat=False,
                 export_onnx=False):
        super(MaskHead, self).__init__()
        self.num_classes = num_classes
        self.export_onnx = export_onnx

        self.roi_extractor = roi_extractor
        if isinstance(roi_extractor, dict):
            self.roi_extractor = RoIAlign(**roi_extractor)
        self.head = head
        self.in_channels = head.out_channels()
        self.mask_assigner = mask_assigner
        self.share_bbox_feat = share_bbox_feat
        self.bbox_head = None

        self.mask_fcn_logits = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            kernel_size=1,
            weight_attr=paddle.ParamAttr(initializer=KaimingNormal(
                fan_in=self.num_classes)))
        self.mask_fcn_logits.skip_quant = True

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
        if not self.export_onnx and rois.shape[0] == 0:
            mask_out = paddle.full([1, 1, 1], -1)
        else:
            bbox = [rois[:, 2:]]
            labels = rois[:, 0].cast('int32')
            rois_feat = self.roi_extractor(body_feats, bbox, rois_num)
            if self.share_bbox_feat:
                assert feat_func is not None
                rois_feat = feat_func(rois_feat)

            mask_feat = self.head(rois_feat)
            mask_logit = self.mask_fcn_logits(mask_feat)
            if self.num_classes == 1:
                mask_out = F.sigmoid(mask_logit)[:, 0, :, :]
            else:
                num_masks = paddle.shape(mask_logit)[0]
                index = paddle.arange(num_masks).cast('int32')
                mask_out = mask_logit[index, labels]
                mask_out_shape = paddle.shape(mask_out)
                mask_out = paddle.reshape(mask_out, [
                    paddle.shape(index), mask_out_shape[-2], mask_out_shape[-1]
                ])
                mask_out = F.sigmoid(mask_out)
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
