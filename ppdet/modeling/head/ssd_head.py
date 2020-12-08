import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from ppdet.core.workspace import register
from ..ops import bipartite_match, label_target_assign, bbox_target_assign, mine_hard_example, box_coder, iou_similarity


@register
class SSDHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['anchor_generator']

    def __init__(self,
                 num_classes=81,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_generator='AnchorGeneratorSSD'):
        super(SSDHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.anchor_generator = anchor_generator
        self.num_priors = self.anchor_generator.num_priors

        self.box_convs = []
        self.score_convs = []
        for i, num_prior in enumerate(self.num_priors):
            self.box_convs.append(
                self.add_sublayer(
                    "boxes{}".format(i),
                    nn.Conv2D(
                        in_channels=in_channels[i],
                        out_channels=num_prior * 4,
                        kernel_size=3,
                        padding=1)))
            self.score_convs.append(
                self.add_sublayer(
                    "scores{}".format(i),
                    nn.Conv2D(
                        in_channels=in_channels[i],
                        out_channels=num_prior * num_classes,
                        kernel_size=3,
                        padding=1)))

    def forward(self, feats, image):
        box_preds = []
        cls_scores = []
        prior_boxes = []
        for feat, box_conv, score_conv in zip(feats, self.box_convs,
                                              self.score_convs):
            box_pred = box_conv(feat)
            box_pred = paddle.transpose(box_pred, [0, 2, 3, 1])
            box_pred = paddle.reshape(box_pred, [0, -1, 4])
            box_preds.append(box_pred)

            cls_score = score_conv(feat)
            cls_score = paddle.transpose(cls_score, [0, 2, 3, 1])
            cls_score = paddle.reshape(cls_score, [0, -1, self.num_classes])
            cls_scores.append(cls_score)

        prior_boxes = self.anchor_generator(feats, image)

        outputs = {}
        outputs['boxes'] = box_preds
        outputs['scores'] = cls_scores
        return outputs, prior_boxes

    def get_loss(self, inputs, targets, prior_boxes):
        boxes = paddle.concat(inputs['boxes'], axis=1)
        scores = paddle.concat(inputs['scores'], axis=1)
        prior_boxes = paddle.concat(prior_boxes, axis=0)
        gt_box = targets['gt_bbox']
        gt_label = targets['gt_class'].unsqueeze(-1)
        batch_size, num_priors, num_classes = scores.shape

        def _reshape_to_2d(x):
            return paddle.flatten(x, start_axis=2)

        # 1. Find matched bounding box by prior box.
        #   1.1 Compute IOU similarity between ground-truth boxes and prior boxes.
        #   1.2 Compute matched bounding box by bipartite matching algorithm.
        matched_indices = []
        matched_dist = []
        for i in range(gt_box.shape[0]):
            iou = iou_similarity(gt_box[i], prior_boxes)
            matched_indice, matched_d = bipartite_match(iou, 'per_prediction',
                                                        0.5)
            matched_indices.append(matched_indice)
            matched_dist.append(matched_d)
        matched_indices = paddle.concat(matched_indices, axis=0)
        matched_dist = paddle.concat(matched_dist, axis=0)
        matched_dist.stop_gradient = True

        # 2. Compute confidence for mining hard examples
        # 2.1. Get the target label based on matched indices
        target_label, _ = label_target_assign(gt_label, matched_indices)
        confidence = _reshape_to_2d(scores)
        # 2.2. Compute confidence loss.
        # Reshape confidence to 2D tensor.
        target_label = _reshape_to_2d(target_label).astype('int64')
        conf_loss = F.softmax_with_cross_entropy(confidence, target_label)
        conf_loss = paddle.reshape(conf_loss, [batch_size, num_priors])

        # 3. Mining hard examples
        neg_mask = mine_hard_example(conf_loss, matched_indices, matched_dist)

        # 4. Assign classification and regression targets
        # 4.1. Encoded bbox according to the prior boxes.
        prior_box_var = paddle.to_tensor(
            np.array(
                [0.1, 0.1, 0.2, 0.2], dtype='float32')).reshape(
                    [1, 4]).expand_as(prior_boxes)
        encoded_bbox = []
        for i in range(gt_box.shape[0]):
            encoded_bbox.append(
                box_coder(
                    prior_box=prior_boxes,
                    prior_box_var=prior_box_var,
                    target_box=gt_box[i],
                    code_type='encode_center_size'))
        encoded_bbox = paddle.stack(encoded_bbox, axis=0)
        # 4.2. Assign regression targets
        target_bbox, target_loc_weight = bbox_target_assign(encoded_bbox,
                                                            matched_indices)
        # 4.3. Assign classification targets
        target_label, target_conf_weight = label_target_assign(
            gt_label, matched_indices, neg_mask=neg_mask)

        # 5. Compute loss.
        # 5.1 Compute confidence loss.
        target_label = _reshape_to_2d(target_label).astype('int64')
        conf_loss = F.softmax_with_cross_entropy(confidence, target_label)

        target_conf_weight = _reshape_to_2d(target_conf_weight)
        conf_loss = conf_loss * target_conf_weight

        # 5.2 Compute regression loss.
        location = _reshape_to_2d(boxes)
        target_bbox = _reshape_to_2d(target_bbox)

        loc_loss = F.smooth_l1_loss(location, target_bbox, reduction='none')
        loc_loss = paddle.sum(loc_loss, axis=-1, keepdim=True)
        target_loc_weight = _reshape_to_2d(target_loc_weight)
        loc_loss = loc_loss * target_loc_weight

        # 5.3 Compute overall weighted loss.
        loss = conf_loss + loc_loss
        loss = paddle.reshape(loss, [batch_size, num_priors])
        loss = paddle.sum(loss, axis=1, keepdim=True)
        normalizer = paddle.sum(target_loc_weight)
        loss = paddle.sum(loss / normalizer)

        return loss
