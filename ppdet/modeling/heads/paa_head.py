import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal

from ppdet.core.workspace import register
from .. import AnchorGenerator, RPNTargetAssign


@register
class PAAHead(nn.Layer):
    """
    Region Proposal Network

    Args:
        anchor_generator (dict): configure of anchor generation
        rpn_target_assign (dict): configure of rpn targets assignment
        train_proposal (dict): configure of proposals generation
            at the stage of training
        test_proposal (dict): configure of proposals generation
            at the stage of prediction
        in_channel (int): channel of input feature maps which can be
            derived by from_config
    """

    def __init__(self,
                 anchor_generator=AnchorGenerator().__dict__,
                 rpn_target_assign=RPNTargetAssign().__dict__,
                 in_channel=1024,
                 num_classes=80):
        super(PAAHead, self).__init__()
        self.anchor_generator = anchor_generator
        self.rpn_target_assign = rpn_target_assign
        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGenerator(**anchor_generator)
        if isinstance(rpn_target_assign, dict):
            self.rpn_target_assign = RPNTargetAssign(**rpn_target_assign)

        num_anchors = self.anchor_generator.num_anchors

        # classification scores
        self.cls_conv = nn.Conv2D(
            in_channels=in_channel,
            out_channels=num_anchors * num_classes,
            kernel_size=1,
            padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)))
        self.cls_conv.skip_quant = True

        # regression deltas
        self.reg_conv = nn.Conv2D(
            in_channels=in_channel,
            out_channels=4 * num_anchors,
            kernel_size=1,
            padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)))
        self.reg_conv.skip_quant = True

    @classmethod
    def from_config(cls, cfg, input_shape):
        # FPN share same rpn head
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channel': input_shape.channels}

    def forward(self, feats, inputs):
        scores = []
        deltas = []

        for feat in feats:
            scores.append(self.cls_conv(feat))
            deltas.append(self.reg_conv(feat))

        anchors = self.anchor_generator(feats)

        # rois, rois_num = self._gen_proposal(scores, deltas, anchors, inputs)
        if self.training:
            loss = self.get_loss(scores, deltas, anchors, inputs)
            return loss
        else:
            return None

    # only calculate score for positive targets
    def get_anchor_score(self, anchors, scores, deltas, score_tgt, bbox_tgt):
        pass

    def get_loss(self, pred_scores, pred_deltas, anchors, inputs):
        """
        pred_scores (list[Tensor]): Multi-level scores prediction
        pred_deltas (list[Tensor]): Multi-level deltas prediction
        anchors (list[Tensor]): Multi-level anchors
        inputs (dict): ground truth info, including im, gt_bbox, gt_score
        """
        anchors = [paddle.reshape(a, shape=(-1, 4)) for a in anchors]
        anchors = paddle.concat(anchors)

        scores = [
            paddle.reshape(
                paddle.transpose(
                    v, perm=[0, 2, 3, 1]),
                shape=(v.shape[0], -1, 1)) for v in pred_scores
        ]
        scores = paddle.concat(scores, axis=1)

        deltas = [
            paddle.reshape(
                paddle.transpose(
                    v, perm=[0, 2, 3, 1]),
                shape=(v.shape[0], -1, 4)) for v in pred_deltas
        ]
        deltas = paddle.concat(deltas, axis=1)

        score_tgt, bbox_tgt, loc_tgt, norm = self.rpn_target_assign(inputs,
                                                                    anchors)

        scores = paddle.reshape(x=scores, shape=(-1, ))
        deltas = paddle.reshape(x=deltas, shape=(-1, 4))

        score_tgt = paddle.concat(score_tgt)
        score_tgt.stop_gradient = True

        pos_mask = score_tgt == 1
        pos_ind = paddle.nonzero(pos_mask)

        valid_mask = score_tgt >= 0
        valid_ind = paddle.nonzero(valid_mask)

        # cls loss
        if valid_ind.shape[0] == 0:
            loss_rpn_cls = paddle.zeros([1], dtype='float32')
        else:
            score_pred = paddle.gather(scores, valid_ind)
            score_label = paddle.gather(score_tgt, valid_ind).cast('float32')
            score_label.stop_gradient = True
            loss_rpn_cls = F.binary_cross_entropy_with_logits(
                logit=score_pred, label=score_label, reduction="sum")

        # reg loss
        if pos_ind.shape[0] == 0:
            loss_rpn_reg = paddle.zeros([1], dtype='float32')
        else:
            loc_pred = paddle.gather(deltas, pos_ind)
            loc_tgt = paddle.concat(loc_tgt)
            loc_tgt = paddle.gather(loc_tgt, pos_ind)
            loc_tgt.stop_gradient = True
            loss_rpn_reg = paddle.abs(loc_pred - loc_tgt).sum()

        score_pred_pos = paddle.gather(scores, pos_ind)
        score_label_pos = paddle.gather(score_tgt, pos_ind)
        score = self.get_anchor_score(anchors, score_pred_pos, loc_pred, score_label_pos, loc_tgt)

        return {
            'loss_rpn_cls': loss_rpn_cls / norm,
            'loss_rpn_reg': loss_rpn_reg / norm
        }
