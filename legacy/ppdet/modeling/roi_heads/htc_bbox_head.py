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
from paddle.fluid.initializer import Normal
from paddle.fluid.regularizer import L2Decay

from ppdet.modeling.ops import MultiClassNMS
from ppdet.modeling.losses import SmoothL1Loss
from ppdet.core.workspace import register

__all__ = ['HTCBBoxHead']


@register
class HTCBBoxHead(object):
    """
    HTC bbox head

    Args:
        head (object): the head module instance
        nms (object): `MultiClassNMS` instance
        num_classes: number of output classes
    """
    __inject__ = ['head', 'nms', 'bbox_loss']
    __shared__ = ['num_classes']

    def __init__(self,
                 head,
                 nms=MultiClassNMS().__dict__,
                 bbox_loss=SmoothL1Loss().__dict__,
                 num_classes=81,
                 lr_ratio=2.0):
        super(HTCBBoxHead, self).__init__()
        self.head = head
        self.nms = nms
        self.bbox_loss = bbox_loss
        self.num_classes = num_classes
        self.lr_ratio = lr_ratio

        if isinstance(nms, dict):
            self.nms = MultiClassNMS(**nms)
        if isinstance(bbox_loss, dict):
            self.bbox_loss = SmoothL1Loss(**bbox_loss)

    def get_output(self,
                   roi_feat,
                   cls_agnostic_bbox_reg=2,
                   wb_scalar=1.0,
                   name=''):
        """
        Get bbox head output.

        Args:
            roi_feat (Variable): RoI feature from RoIExtractor.
            cls_agnostic_bbox_reg(Int): BBox regressor are class agnostic.
            wb_scalar(Float): Weights and Bias's learning rate.
            name(String): Layer's name

        Returns:
            cls_score(Variable): cls score.
            bbox_pred(Variable): bbox regression.
        """
        head_feat = self.head(roi_feat, wb_scalar, name)
        cls_score = fluid.layers.fc(input=head_feat,
                                    size=self.num_classes,
                                    act=None,
                                    name='cls_score' + name,
                                    param_attr=ParamAttr(
                                        name='cls_score%s_w' % name,
                                        initializer=Normal(
                                            loc=0.0, scale=0.01),
                                        learning_rate=wb_scalar),
                                    bias_attr=ParamAttr(
                                        name='cls_score%s_b' % name,
                                        learning_rate=wb_scalar * self.lr_ratio,
                                        regularizer=L2Decay(0.)))
        bbox_pred = fluid.layers.fc(input=head_feat,
                                    size=4 * cls_agnostic_bbox_reg,
                                    act=None,
                                    name='bbox_pred' + name,
                                    param_attr=ParamAttr(
                                        name='bbox_pred%s_w' % name,
                                        initializer=Normal(
                                            loc=0.0, scale=0.001),
                                        learning_rate=wb_scalar),
                                    bias_attr=ParamAttr(
                                        name='bbox_pred%s_b' % name,
                                        learning_rate=wb_scalar * self.lr_ratio,
                                        regularizer=L2Decay(0.)))
        return cls_score, bbox_pred

    def get_loss(self, rcnn_pred_list, rcnn_target_list, rcnn_loss_weight_list):
        """
        Get bbox_head loss.

        Args:
            rcnn_pred_list(List): Cascade RCNN's head's output including
                bbox_pred and cls_score
            rcnn_target_list(List): Cascade rcnn's bbox and label target
            rcnn_loss_weight_list(List): The weight of location and class loss

        Return:
            loss_cls(Variable): bbox_head loss.
            loss_bbox(Variable): bbox_head loss.
        """
        loss_dict = {}
        for i, (rcnn_pred, rcnn_target
                ) in enumerate(zip(rcnn_pred_list, rcnn_target_list)):
            labels_int64 = fluid.layers.cast(x=rcnn_target[1], dtype='int64')
            labels_int64.stop_gradient = True

            loss_cls = fluid.layers.softmax_with_cross_entropy(
                logits=rcnn_pred[0],
                label=labels_int64,
                numeric_stable_mode=True, )
            loss_cls = fluid.layers.reduce_mean(
                loss_cls, name='loss_cls_' + str(i)) * rcnn_loss_weight_list[i]

            loss_bbox = self.bbox_loss(
                x=rcnn_pred[1],
                y=rcnn_target[2],
                inside_weight=rcnn_target[3],
                outside_weight=rcnn_target[4])
            loss_bbox = fluid.layers.reduce_mean(
                loss_bbox,
                name='loss_bbox_' + str(i)) * rcnn_loss_weight_list[i]

            loss_dict['loss_cls_%d' % i] = loss_cls
            loss_dict['loss_loc_%d' % i] = loss_bbox

        return loss_dict

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
        repeat_num = 3
        # cls score 
        boxes_cls_prob_l = []
        for i in range(repeat_num):
            cls_score = rcnn_pred_list[i][0]
            cls_prob = fluid.layers.softmax(cls_score, use_cudnn=False)
            boxes_cls_prob_l.append(cls_prob)

        boxes_cls_prob_mean = fluid.layers.sum(boxes_cls_prob_l) / float(
            len(boxes_cls_prob_l))

        # bbox pred
        im_scale = fluid.layers.slice(im_info, [1], starts=[2], ends=[3])
        bbox_pred_l = []
        for i in range(repeat_num):
            if i < 2:
                continue
            bbox_reg_w = cascade_bbox_reg_weights[i]
            proposals_boxes = proposal_list[i]
            im_scale_lod = fluid.layers.sequence_expand(im_scale,
                                                        proposals_boxes)
            proposals_boxes = proposals_boxes / im_scale_lod
            bbox_pred = rcnn_pred_list[i][1]
            bbox_pred_new = fluid.layers.reshape(bbox_pred,
                                                 (-1, cls_agnostic_bbox_reg, 4))
            bbox_pred_l.append(bbox_pred_new)

        bbox_pred_new = bbox_pred_l[-1]
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

    def get_prediction_cls_aware(self,
                                 im_info,
                                 im_shape,
                                 cascade_cls_prob,
                                 cascade_decoded_box,
                                 cascade_bbox_reg_weights,
                                 return_box_score=False):
        '''
        get_prediction_cls_aware: predict bbox for each class
        '''
        cascade_num_stage = 3
        cascade_eval_weight = [0.2, 0.3, 0.5]
        # merge 3 stages results
        sum_cascade_cls_prob = sum([
            prob * cascade_eval_weight[idx]
            for idx, prob in enumerate(cascade_cls_prob)
        ])
        sum_cascade_decoded_box = sum([
            bbox * cascade_eval_weight[idx]
            for idx, bbox in enumerate(cascade_decoded_box)
        ])
        self.im_scale = fluid.layers.slice(im_info, [1], starts=[2], ends=[3])
        im_scale_lod = fluid.layers.sequence_expand(self.im_scale,
                                                    sum_cascade_decoded_box)

        sum_cascade_decoded_box = sum_cascade_decoded_box / im_scale_lod

        decoded_bbox = sum_cascade_decoded_box
        decoded_bbox = fluid.layers.reshape(
            decoded_bbox, shape=(-1, self.num_classes, 4))

        box_out = fluid.layers.box_clip(input=decoded_bbox, im_info=im_shape)
        if return_box_score:
            return {'bbox': box_out, 'score': sum_cascade_cls_prob}
        pred_result = self.nms(bboxes=box_out, scores=sum_cascade_cls_prob)
        return {"bbox": pred_result}
