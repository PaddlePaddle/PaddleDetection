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
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import Layer, Sequential
from paddle.nn import Conv2D, Conv2DTranspose, ReLU
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from ppdet.modeling import ops


@register
class MaskFeat(Layer):
    __inject__ = ['mask_roi_extractor']

    def __init__(self,
                 mask_roi_extractor=None,
                 num_convs=0,
                 feat_in=2048,
                 feat_out=256,
                 mask_num_stages=1,
                 share_bbox_feat=False):
        super(MaskFeat, self).__init__()
        self.num_convs = num_convs
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.mask_roi_extractor = mask_roi_extractor
        self.mask_num_stages = mask_num_stages
        self.share_bbox_feat = share_bbox_feat
        self.upsample_module = []
        fan_conv = feat_out * 3 * 3
        fan_deconv = feat_out * 2 * 2
        for i in range(self.mask_num_stages):
            name = 'stage_{}'.format(i)
            mask_conv = Sequential()
            for j in range(self.num_convs):
                conv_name = 'mask_inter_feat_{}'.format(j + 1)
                mask_conv.add_sublayer(
                    conv_name,
                    Conv2D(
                        in_channels=feat_in if j == 0 else feat_out,
                        out_channels=feat_out,
                        kernel_size=3,
                        padding=1,
                        weight_attr=ParamAttr(
                            initializer=KaimingNormal(fan_in=fan_conv)),
                        bias_attr=ParamAttr(
                            learning_rate=2., regularizer=L2Decay(0.))))
                mask_conv.add_sublayer(conv_name + 'act', ReLU())
            mask_conv.add_sublayer(
                'conv5_mask',
                Conv2DTranspose(
                    in_channels=self.feat_in,
                    out_channels=self.feat_out,
                    kernel_size=2,
                    stride=2,
                    weight_attr=ParamAttr(
                        initializer=KaimingNormal(fan_in=fan_deconv)),
                    bias_attr=ParamAttr(
                        learning_rate=2., regularizer=L2Decay(0.))))
            mask_conv.add_sublayer('conv5_mask' + 'act', ReLU())
            upsample = self.add_sublayer(name, mask_conv)
            self.upsample_module.append(upsample)

    def forward(self,
                body_feats,
                bboxes,
                bbox_feat,
                mask_index,
                spatial_scale,
                stage=0,
                bbox_head_feat_func=None,
                mode='train'):
        if self.share_bbox_feat and mask_index is not None:
            rois_feat = paddle.gather(bbox_feat, mask_index)
        else:
            rois_feat = self.mask_roi_extractor(body_feats, bboxes,
                                                spatial_scale)
        if self.share_bbox_feat and bbox_head_feat_func is not None and mode == 'infer':
            rois_feat = bbox_head_feat_func(rois_feat)

        # upsample 
        mask_feat = self.upsample_module[stage](rois_feat)
        return mask_feat


@register
class MaskHead(Layer):
    __shared__ = ['num_classes', 'mask_num_stages']
    __inject__ = ['mask_feat']

    def __init__(self,
                 mask_feat,
                 feat_in=256,
                 num_classes=81,
                 mask_num_stages=1):
        super(MaskHead, self).__init__()
        self.mask_feat = mask_feat
        self.feat_in = feat_in
        self.num_classes = num_classes
        self.mask_num_stages = mask_num_stages
        self.mask_fcn_logits = []
        for i in range(self.mask_num_stages):
            name = 'mask_fcn_logits_{}'.format(i)
            self.mask_fcn_logits.append(
                self.add_sublayer(
                    name,
                    Conv2D(
                        in_channels=self.feat_in,
                        out_channels=self.num_classes,
                        kernel_size=1,
                        weight_attr=ParamAttr(initializer=KaimingNormal(
                            fan_in=self.num_classes)),
                        bias_attr=ParamAttr(
                            learning_rate=2., regularizer=L2Decay(0.0)))))

    def forward_train(self,
                      body_feats,
                      bboxes,
                      bbox_feat,
                      mask_index,
                      spatial_scale,
                      stage=0):
        # feat
        mask_feat = self.mask_feat(
            body_feats,
            bboxes,
            bbox_feat,
            mask_index,
            spatial_scale,
            stage,
            mode='train')
        # logits
        mask_head_out = self.mask_fcn_logits[stage](mask_feat)
        return mask_head_out

    def forward_test(self,
                     scale_factor,
                     body_feats,
                     bboxes,
                     bbox_feat,
                     mask_index,
                     spatial_scale,
                     stage=0,
                     bbox_head_feat_func=None):
        bbox, bbox_num = bboxes

        if bbox.shape[0] == 0:
            mask_head_out = paddle.full([1, 6], -1)
        else:
            scale_factor_list = []
            for idx in range(bbox_num.shape[0]):
                num = bbox_num[idx]
                scale = scale_factor[idx, 0]
                ones = paddle.ones(num)
                scale_expand = ones * scale
                scale_factor_list.append(scale_expand)
            scale_factor_list = paddle.cast(
                paddle.concat(scale_factor_list), 'float32')
            scale_factor_list = paddle.reshape(scale_factor_list, shape=[-1, 1])
            scaled_bbox = paddle.multiply(bbox[:, 2:], scale_factor_list)
            scaled_bboxes = (scaled_bbox, bbox_num)
            mask_feat = self.mask_feat(
                body_feats,
                scaled_bboxes,
                bbox_feat,
                mask_index,
                spatial_scale,
                stage,
                bbox_head_feat_func,
                mode='infer')
            mask_logit = self.mask_fcn_logits[stage](mask_feat)
            mask_head_out = F.sigmoid(mask_logit)
        return mask_head_out

    def forward(self,
                inputs,
                body_feats,
                bboxes,
                bbox_feat,
                mask_index,
                spatial_scale,
                bbox_head_feat_func=None,
                stage=0):
        if inputs['mode'] == 'train':
            mask_head_out = self.forward_train(body_feats, bboxes, bbox_feat,
                                               mask_index, spatial_scale, stage)
        else:
            scale_factor = inputs['scale_factor']
            mask_head_out = self.forward_test(
                scale_factor, body_feats, bboxes, bbox_feat, mask_index,
                spatial_scale, stage, bbox_head_feat_func)
        return mask_head_out

    def get_loss(self, mask_head_out, mask_target):
        mask_logits = paddle.flatten(mask_head_out, start_axis=1, stop_axis=-1)
        mask_label = paddle.cast(x=mask_target, dtype='float32')
        mask_label.stop_gradient = True
        loss_mask = ops.sigmoid_cross_entropy_with_logits(
            input=mask_logits,
            label=mask_label,
            ignore_index=-1,
            normalize=True)
        loss_mask = paddle.sum(loss_mask)

        return {'loss_mask': loss_mask}
