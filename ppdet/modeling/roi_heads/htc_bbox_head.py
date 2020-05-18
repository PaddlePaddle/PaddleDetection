# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Xavier
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import MSRA

from ppdet.modeling.ops import MultiClassNMS
from ppdet.modeling.ops import ConvNorm
from ppdet.modeling.losses import SmoothL1Loss
from ppdet.core.workspace import register
from .cascade_head import CascadeBBoxHead 

__all__ = ['HTCBBoxHead']


@register
class HTCBBoxHead(CascadeBBoxHead):
    """
    HTC bbox head

    Args:
        head (object): the head module instance
        nms (object): `MultiClassNMS` instance
        num_classes: number of output classes
    """
    __inject__ = ['head', 'nms', 'bbox_loss']
    __shared__ = ['num_classes']

    def __init__(
            self,
            head,
            nms=MultiClassNMS().__dict__,
            bbox_loss=SmoothL1Loss().__dict__,
            num_classes=81, ):
        super(CascadeBBoxHead, self).__init__()
        self.head = head
        self.nms = nms
        self.bbox_loss = bbox_loss
        self.num_classes = num_classes
        if isinstance(nms, dict):
            self.nms = MultiClassNMS(**nms)
        if isinstance(bbox_loss, dict):
            self.bbox_loss = SmoothL1Loss(**bbox_loss)

    def get_prediction(self,
                       im_info,
                       im_shape,
                       roi_feat_list,
                       rcnn_pred_list,
                       proposal_list,
                       cascade_bbox_reg_weights,
                       cls_agnostic_bbox_reg=2,
                       return_box_score=False):
        """
        Get prediction bounding box in test stage.
        :
        Args:
            im_info (Variable): A 2-D LoDTensor with shape [B, 3]. B is the
                number of input images, each element consists
                of im_height, im_width, im_scale.
            im_shape (Variable): Actual shape of original image with shape
                [B, 3]. B is the number of images, each element consists of
                original_height, original_width, 1
            rois_feat_list (List): RoI feature from RoIExtractor.
            rcnn_pred_list (Variable): Cascade rcnn's head's output
                including bbox_pred and cls_score
            proposal_list (List): RPN proposal boxes.
            cascade_bbox_reg_weights (List): BBox decode var.
            cls_agnostic_bbox_reg(Int): BBox regressor are class agnostic

        Returns:
            pred_result(Variable): Prediction result with shape [N, 6]. Each
               row has 6 values: [label, confidence, xmin, ymin, xmax, ymax].
               N is the total number of prediction.
        """
        self.im_scale = fluid.layers.slice(im_info, [1], starts=[2], ends=[3])
        boxes_cls_prob_l = []

        rcnn_pred = rcnn_pred_list[-1]  # stage 3
        repreat_num = 1
        repreat_num = 3
        bbox_reg_w = cascade_bbox_reg_weights[-1]
        for i in range(repreat_num):
            # cls score
            cls_score, _ = self.get_output(
                roi_feat_list[i],  # roi_feat_3
                name='_' + str(i + 1) if i > 0 else '')
            cls_prob = fluid.layers.softmax(cls_score, use_cudnn=False)
            boxes_cls_prob_l.append(cls_prob)

        boxes_cls_prob_mean = (
            boxes_cls_prob_l[0] + boxes_cls_prob_l[1] + boxes_cls_prob_l[2]
        ) / 3.0

        # bbox pred
        proposals_boxes = proposal_list[-1]
        im_scale_lod = fluid.layers.sequence_expand(self.im_scale,
                                                    proposals_boxes)
        proposals_boxes = proposals_boxes / im_scale_lod
        bbox_pred = rcnn_pred[1]
        bbox_pred_new = fluid.layers.reshape(bbox_pred,
                                             (-1, cls_agnostic_bbox_reg, 4))
        if cls_agnostic_bbox_reg == 2:
            # only use fg box delta to decode box
            bbox_pred_new = fluid.layers.slice(
                bbox_pred_new, axes=[1], starts=[1], ends=[2])
            bbox_pred_new = fluid.layers.expand(bbox_pred_new,
                                                [1, self.num_classes, 1])
        decoded_box = fluid.layers.box_coder(
            prior_box=proposals_boxes,
            prior_box_var=bbox_reg_w,
            target_box=bbox_pred_new,
            code_type='decode_center_size',
            box_normalized=False,
            axis=1)

        box_out = fluid.layers.box_clip(input=decoded_box, im_info=im_shape)
        if return_box_score:
            return {'bbox': box_out, 'score': boxes_cls_prob_mean}
        pred_result = self.nms(bboxes=box_out, scores=boxes_cls_prob_mean)
        return {"bbox": pred_result}

