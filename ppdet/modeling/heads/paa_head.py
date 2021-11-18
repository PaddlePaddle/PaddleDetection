import sys

import numpy as np

from ..bbox_utils import delta2bbox, bbox_iou, bbox_overlaps

try:
    import sklearn.mixture as skm
except ImportError:
    skm = None

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal

from ppdet.core.workspace import register
from ...modeling.proposal_generator.anchor_generator import AnchorGenerator
from ...modeling.proposal_generator.target_layer import PAATargetAssign


@register
class PAAHead(nn.Layer):
    """
    Region Proposal Network

    Args:
        anchor_generator (dict): configure of anchor generation
        target_assign (dict): configure of paa targets assignment
        train_proposal (dict): configure of proposals generation
            at the stage of training
        test_proposal (dict): configure of proposals generation
            at the stage of prediction
        in_channel (int): channel of input feature maps which can be
            derived by from_config
    """

    def __init__(self,
                 head,
                 anchor_generator=AnchorGenerator().__dict__,
                 target_assign=PAATargetAssign().__dict__,
                 in_channel=1024,
                 num_classes=80,
                 topk=9,
                 stacked_convs=4,
                 covariance_type='diag'):
        super(PAAHead, self).__init__()
        self.stacked_convs = stacked_convs
        self.covariance_type = covariance_type
        self.head = head
        self.anchor_generator = anchor_generator
        self.target_assign = target_assign
        self.num_classes = num_classes
        self.topk = topk
        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGenerator(**anchor_generator)
        if isinstance(target_assign, dict):
            self.target_assign = PAATargetAssign(**target_assign)

        num_anchors = self.anchor_generator.num_anchors

        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        for i in range(self.stacked_convs):
            self.cls_convs.append(nn.Conv2D(256, 256, 3, padding=1))
            self.reg_convs.append(nn.Conv2D(256, 256, 3, padding=1))

        # classification scores
        self.paa_cls = nn.Conv2D(
            in_channels=in_channel,
            out_channels=num_anchors * (num_classes+1),
            kernel_size=1,
            padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)))
        self.paa_cls.skip_quant = True

        # regression deltas
        self.paa_reg = nn.Conv2D(
            in_channels=in_channel,
            out_channels=4 * num_anchors,
            kernel_size=1,
            padding=0,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0., std=0.01)))
        self.paa_reg.skip_quant = True

        self.iou_branch = nn.Conv2D(
            in_channels=in_channel,
            out_channels=1 * num_anchors,
            kernel_size=3,
            padding=1)

    @classmethod
    def from_config(cls, cfg, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channel': input_shape.channels}

    def forward(self, feats, inputs):
        scores = []
        deltas = []
        iou_preds = []

        cls_feats = []
        reg_feats = []

        for feat in feats:
            cls_feat = feat
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            cls_feats.append(cls_feat)

            reg_feat = feat
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            reg_feats.append(reg_feat)

        for cls_feat in cls_feats:
            scores.append(self.paa_cls(cls_feat))

        for reg_feat in reg_feats:
            deltas.append(self.paa_reg(reg_feat))
            iou_preds.append(self.iou_branch(reg_feat))

        anchors = self.anchor_generator(feats)

        # rois, rois_num = self._gen_proposal(scores, deltas, anchors, inputs)
        if self.training:
            loss = self.get_loss(scores, deltas, iou_preds, anchors, inputs)
            return loss
        else:
            return self.get_prediction(scores, deltas), anchors

    def get_prediction(self, score, delta):
        bbox_prob = F.softmax(score)
        return delta, bbox_prob

    # only calculate score for positive targets
    def get_anchor_score(self, anchors, scores, bbox_pred, label_tgt, bbox_tgt):
        loss_cls = F.binary_cross_entropy_with_logits(logit=scores, label=label_tgt, reduction="none")
        # TODO: original code is using bbox absolute distance loss, here we use percentage
        #  to calculate, so it's much lower than original version
        loss_reg = paddle.abs(bbox_pred - bbox_tgt).mean(axis=-1)

        return loss_cls+loss_reg

    def paa_reassign(self, pos_losses, label, pos_inds, pos_gt_inds, anchors):
        """Fit loss to GMM distribution and separate positive, ignore, negative
        samples again with GMM model.

        Args:
            pos_losses (Tensor): Losses of all positive samples in
                single image.
            label (Tensor): classification target of each anchor with
                shape (num_anchors,)
            label_weight (Tensor): Classification loss weight of each
                anchor with shape (num_anchors).
            bbox_weight (Tensor): Bbox weight of each anchor with shape
                (num_anchors, 4).
            pos_inds (Tensor): Index of all positive samples got from
                first assign process.
            pos_gt_inds (Tensor): Gt_index of all positive samples got
                from first assign process.
            anchors (list[Tensor]): Anchors of each scale.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - label (Tensor): classification target of each anchor after
                  paa assign, with shape (num_anchors,)
                - label_weight (Tensor): Classification loss weight of each
                  anchor after paa assign, with shape (num_anchors).
                - bbox_weight (Tensor): Bbox weight of each anchor with shape
                  (num_anchors, 4).
                - num_pos (int): The number of positive samples after paa
                  assign.
        """
        if not len(pos_inds):
            return label, 0
        label = label.clone()
        num_gt = pos_gt_inds.max() + 1
        num_level = len(anchors)
        num_anchors_each_level = [item.shape[0] for item in anchors]
        num_anchors_each_level.insert(0, 0)
        inds_level_interval = np.cumsum(num_anchors_each_level)
        pos_level_mask = []
        for i in range(num_level):
            mask = paddle.logical_and((pos_inds >= inds_level_interval[i]), (pos_inds < inds_level_interval[i + 1]))
            pos_level_mask.append(mask)
        pos_inds_after_paa = [paddle.Tensor(np.array([]))]
        ignore_inds_after_paa = [paddle.Tensor(np.array([]))]
        for gt_ind in range(num_gt):
            pos_inds_gmm = []
            pos_loss_gmm = []
            gt_mask = pos_gt_inds == gt_ind
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                level_gt_mask = paddle.nonzero(paddle.logical_and(level_mask, gt_mask))
                if level_gt_mask.sum() == 0 or len(level_gt_mask)==0:
                    value = paddle.Tensor()
                    topk_inds = paddle.Tensor()
                else:
                    value, topk_inds = pos_losses.gather(level_gt_mask).topk(min(len(level_gt_mask), self.topk), largest=False)

                pos_gts = pos_inds.gather(level_gt_mask)
                # print(level_gt_mask.shape, pos_gts.shape)
                if pos_gts.shape[0] == 0:
                    continue
                pos_inds_gmm.append(pos_gts.gather(topk_inds))
                pos_loss_gmm.append(value)

            # paddle.concat doesn't support empty sequence
            if len(pos_inds_gmm) == 0:
                continue

            pos_inds_gmm = paddle.concat(pos_inds_gmm)
            pos_loss_gmm = paddle.concat(pos_loss_gmm)
            # fix gmm need at least two sample
            if len(pos_inds_gmm) < 2:
                continue
            # device = pos_inds_gmm.device
            # pos_loss_gmm, sort_inds = pos_loss_gmm.sort()
            sort_inds = pos_loss_gmm.argsort()
            pos_loss_gmm = pos_loss_gmm.gather(sort_inds)
            pos_inds_gmm = pos_inds_gmm.gather(sort_inds)
            # pos_loss_gmm = pos_loss_gmm.view(-1, 1).cpu().numpy()
            pos_loss_gmm = pos_loss_gmm.reshape((-1, 1)).numpy()
            min_loss, max_loss = pos_loss_gmm.min(), pos_loss_gmm.max()
            means_init = np.array([min_loss, max_loss]).reshape(2, 1)
            weights_init = np.array([0.5, 0.5])
            precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)  # full
            if self.covariance_type == 'spherical':
                precisions_init = precisions_init.reshape(2)
            elif self.covariance_type == 'diag':
                precisions_init = precisions_init.reshape(2, 1)
            elif self.covariance_type == 'tied':
                precisions_init = np.array([[1.0]])
            if skm is None:
                raise ImportError('Please run "pip install sklearn" '
                                  'to install sklearn first.')
            gmm = skm.GaussianMixture(
                2,
                weights_init=weights_init,
                means_init=means_init,
                precisions_init=precisions_init,
                covariance_type=self.covariance_type,
                # reg_covar=1e-5
            )
            print(pos_loss_gmm, pos_loss_gmm.shape)
            try:
                gmm.fit(pos_loss_gmm)
            except:
                print(sys.exc_info())
                continue

            gmm_assignment = gmm.predict(pos_loss_gmm)
            scores = gmm.score_samples(pos_loss_gmm)
            # gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
            gmm_assignment = paddle.Tensor(gmm_assignment)
            # scores = torch.from_numpy(scores).to(device)
            scores = paddle.Tensor(scores)

            pos_inds_temp, ignore_inds_temp = self.gmm_separation_scheme(
                gmm_assignment, scores, pos_inds_gmm)
            pos_inds_after_paa.append(pos_inds_temp)
            ignore_inds_after_paa.append(ignore_inds_temp)

        num_pos = 0
        for inds in pos_inds_after_paa:
            print(inds)
            num_pos += len(inds)

        if num_pos>0:
            pos_inds_after_paa = paddle.concat(pos_inds_after_paa)
            # ignore_inds_after_paa = paddle.concat(ignore_inds_after_paa)
            reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_paa).all(1)
            reassign_ids = pos_inds.gather(reassign_mask.nonzero())
            # label[reassign_ids] = self.num_classes
            for id in reassign_ids:
                label[id] = self.num_classes

            # num_pos = len(pos_inds_after_paa)

        return label, num_pos

    def gmm_separation_scheme(self, gmm_assignment, scores, pos_inds_gmm):
        """A general separation scheme for gmm model.

        It separates a GMM distribution of candidate samples into three
        parts, 0 1 and uncertain areas, and you can implement other
        separation schemes by rewriting this function.

        Args:
            gmm_assignment (Tensor): The prediction of GMM which is of shape
                (num_samples,). The 0/1 value indicates the distribution
                that each sample comes from.
            scores (Tensor): The probability of sample coming from the
                fit GMM distribution. The tensor is of shape (num_samples,).
            pos_inds_gmm (Tensor): All the indexes of samples which are used
                to fit GMM model. The tensor is of shape (num_samples,)

        Returns:
            tuple[Tensor]: The indices of positive and ignored samples.

                - pos_inds_temp (Tensor): Indices of positive samples.
                - ignore_inds_temp (Tensor): Indices of ignore samples.
        """
        # The implementation is (c) in Fig.3 in origin paper instead of (b).
        # You can refer to issues such as
        # https://github.com/kkhoot/PAA/issues/8 and
        # https://github.com/kkhoot/PAA/issues/9.
        fgs = gmm_assignment == 0
        fgs = fgs.nonzero()
        # pos_inds_temp = fgs.new_tensor([], dtype=torch.long)
        pos_inds_temp = paddle.Tensor()
        # ignore_inds_temp = fgs.new_tensor([], dtype=torch.long)
        ignore_inds_temp = paddle.Tensor()
        if fgs.nonzero().numel():
            _, pos_thr_ind = scores.gather(fgs).topk(1)
            pos_inds_temp = pos_inds_gmm.gather(fgs)[:pos_thr_ind + 1]
            # ignore_inds_temp = pos_inds_gmm.new_tensor([])
            ignore_inds_temp = paddle.Tensor()
        return pos_inds_temp, ignore_inds_temp

    def get_loss(self, pred_scores, pred_deltas, iou_preds, anchors, inputs):
        """
        pred_scores (list[Tensor]): Multi-level scores prediction
        pred_deltas (list[Tensor]): Multi-level deltas prediction
        anchors (list[Tensor]): Multi-level anchors
        inputs (dict): ground truth info, including im, gt_bbox, gt_score
        """
        multi_level_anchors = [paddle.reshape(a, shape=(-1, 4)) for a in anchors]
        anchors = paddle.concat(multi_level_anchors)

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

        label_tgt, bbox_tgt, gt_inds, deltas_tgt, norm = self.target_assign(inputs, anchors, self.num_classes)

        scores = paddle.reshape(x=scores, shape=(-1, ))
        deltas = paddle.reshape(x=deltas, shape=(-1, 4))

        label_tgt = paddle.concat(label_tgt)
        label_tgt.stop_gradient = True

        gt_inds = paddle.concat(gt_inds)

        pos_mask = label_tgt != self.num_classes
        pos_ind = paddle.nonzero(pos_mask)
        pos_gt_inds = paddle.gather(gt_inds, pos_ind)

        loc_pred = paddle.gather(deltas, pos_ind)
        origin_deltas_tgt = paddle.concat(deltas_tgt)
        deltas_tgt = paddle.gather(origin_deltas_tgt, pos_ind)
        deltas_tgt.stop_gradient = True

        score_pred_pos = paddle.gather(scores, pos_ind)
        label_tgt_pos = paddle.gather(label_tgt, pos_ind).cast('float32')

        score = self.get_anchor_score(anchors, score_pred_pos, loc_pred, label_tgt_pos, deltas_tgt)

        reassign_labels, num_pos = self.paa_reassign(score, label_tgt, pos_ind.reshape((-1,)), pos_gt_inds, multi_level_anchors)

        cls_scores = [
            paddle.reshape(
                paddle.transpose(
                    v, perm=[0, 2, 3, 1]),
                shape=(v.shape[0], -1, self.num_classes+1)) for v in pred_scores
        ]
        cls_scores = paddle.concat(cls_scores, axis=1)
        cls_scores = [v for v in cls_scores]

        bbox_preds = [
            paddle.reshape(
                paddle.transpose(
                    v, perm=[0, 2, 3, 1]),
                shape=(v.shape[0], -1, 4)) for v in pred_deltas
        ]
        bbox_preds = paddle.concat(bbox_preds, axis=1)
        bbox_preds = [v for v in bbox_preds]
        # convert all tensor list to a flatten tensor
        # cls_scores = paddle.concat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        cls_scores = paddle.concat(cls_scores, 0)
        # bbox_preds = paddle.concat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
        bbox_preds = paddle.concat(bbox_preds, 0)

        # iou_preds = paddle.concat(iou_preds, 0).view(-1, iou_preds[0].size(-1))
        iou_preds = [
            paddle.reshape(
                paddle.transpose(
                    v, perm=[0, 2, 3, 1]),
                shape=(v.shape[0], -1, 1)) for v in iou_preds
        ]
        iou_preds = paddle.concat(iou_preds, axis=1).reshape([-1,1])

        # labels = paddle.concat(reassign_labels, 0).view(-1)
        labels = reassign_labels
        # flatten_anchors = paddle.concat([paddle.concat(item, 0) for item in multi_level_anchors])
        flatten_anchors = paddle.concat(multi_level_anchors)

        # pos_inds_flatten = ((labels >= 0)
        #                     &
        #                     (labels < self.num_classes)).nonzero().reshape(-1)
        # pos_inds_flatten = paddle.logical_and((labels >= 0), (labels < self.num_classes)).nonzero().reshape([-1])
        pos_inds_flatten = paddle.logical_and((labels >= 0), (labels < self.num_classes)).nonzero().reshape([-1])

        # losses_cls = self.loss_cls(
        #     cls_scores,
        #     labels,
        #     avg_factor=max(num_pos, len(img_metas)))  # avoid num_pos=0

        # losses_cls = F.cross_entropy(cls_scores, labels.cast('int64'))
        losses_cls = F.sigmoid_focal_loss(cls_scores, F.one_hot(labels, self.num_classes+1), reduction='mean')

        if num_pos:
            # pos_bbox_pred = self.bbox_coder.decode(
            #     flatten_anchors[pos_inds_flatten],
            #     bbox_preds[pos_inds_flatten])
            # pos_bbox_pred = bbox_preds[pos_inds_flatten]
            aligned_anchors = paddle.concat([anchors]*pred_scores[0].shape[0])
            pos_aligned_anchors = paddle.gather(aligned_anchors, pos_inds_flatten)
            pos_bbox_pred = bbox_preds.gather(pos_inds_flatten)
            pos_bbox_pred_absolute = delta2bbox(pos_bbox_pred, pos_aligned_anchors, [1,1,1,1]).reshape([-1,4])
            # pos_bbox_target = deltas_tgt[pos_inds_flatten]
            pos_bbox_target = origin_deltas_tgt.gather(pos_inds_flatten)
            pos_bbox_target_absolute = delta2bbox(pos_bbox_target, pos_aligned_anchors, [1, 1, 1, 1]).reshape([-1,4])
            # iou = bbox_overlaps(pos_bbox_pred_absolute.reshape([-1,4]), pos_bbox_target_absolute.reshape([-1,4]))
            iou = []
            for i in range(pos_bbox_pred_absolute.shape[0]):
                iou.append(bbox_iou(pos_bbox_pred_absolute[i], pos_bbox_target_absolute[i]))
            iou = paddle.concat(iou)

            iou_loss = F.binary_cross_entropy_with_logits(iou_preds.gather(pos_inds_flatten), iou.reshape([-1,1]))

            # iou_target = bbox_overlaps(
            #     pos_bbox_pred.detach(), pos_bbox_target, is_aligned=True)
            # losses_iou = self.loss_centerness(
            #     iou_preds[pos_inds_flatten],
            #     iou_target.unsqueeze(-1),
            #     avg_factor=num_pos)

            # losses_bbox = self.loss_bbox(
            #     pos_bbox_pred,
            #     pos_bbox_target,
            #     # iou_target.clamp(min=EPS),
            #     # avg_factor=iou_target.sum()
            # )
            losses_bbox = paddle.abs(pos_bbox_pred - pos_bbox_target).mean()
        else:
            iou_loss = iou_preds.sum() * 0
            losses_bbox = bbox_preds.sum() * 0

        return {
            # 'loss_bbox_cls': losses_cls / norm,
            'loss_bbox_cls': losses_cls,
            # 'loss_bbox_reg': losses_bbox / norm
            'loss_bbox_reg': losses_bbox,
            'loss_iou': iou_loss
        }

    def score_voting(self, det_bboxes, det_labels, mlvl_bboxes,
                     mlvl_nms_scores, score_thr):
        """Implementation of score voting method works on each remaining boxes
        after NMS procedure.

        Args:
            det_bboxes (Tensor): Remaining boxes after NMS procedure,
                with shape (k, 5), each dimension means
                (x1, y1, x2, y2, score).
            det_labels (Tensor): The label of remaining boxes, with shape
                (k, 1),Labels are 0-based.
            mlvl_bboxes (Tensor): All boxes before the NMS procedure,
                with shape (num_anchors,4).
            mlvl_nms_scores (Tensor): The scores of all boxes which is used
                in the NMS procedure, with shape (num_anchors, num_class)
            mlvl_iou_preds (Tensor): The predictions of IOU of all boxes
                before the NMS procedure, with shape (num_anchors, 1)
            score_thr (float): The score threshold of bboxes.

        Returns:
            tuple: Usually returns a tuple containing voting results.

                - det_bboxes_voted (Tensor): Remaining boxes after
                    score voting procedure, with shape (k, 5), each
                    dimension means (x1, y1, x2, y2, score).
                - det_labels_voted (Tensor): Label of remaining bboxes
                    after voting, with shape (num_anchors,).
        """
        candidate_mask = mlvl_nms_scores > score_thr
        candidate_mask_nonzeros = candidate_mask.nonzero()
        candidate_inds = candidate_mask_nonzeros[:, 0]
        candidate_labels = candidate_mask_nonzeros[:, 1]
        candidate_bboxes = mlvl_bboxes[candidate_inds]
        candidate_scores = mlvl_nms_scores[candidate_mask]
        det_bboxes_voted = []
        det_labels_voted = []
        for cls in range(self.cls_out_channels):
            candidate_cls_mask = candidate_labels == cls
            if not candidate_cls_mask.any():
                continue
            candidate_cls_scores = candidate_scores[candidate_cls_mask]
            candidate_cls_bboxes = candidate_bboxes[candidate_cls_mask]
            det_cls_mask = det_labels == cls
            det_cls_bboxes = det_bboxes[det_cls_mask].view(
                -1, det_bboxes.size(-1))
            det_candidate_ious = bbox_overlaps(det_cls_bboxes[:, :4],
                                               candidate_cls_bboxes)
            for det_ind in range(len(det_cls_bboxes)):
                single_det_ious = det_candidate_ious[det_ind]
                pos_ious_mask = single_det_ious > 0.01
                pos_ious = single_det_ious[pos_ious_mask]
                pos_bboxes = candidate_cls_bboxes[pos_ious_mask]
                pos_scores = candidate_cls_scores[pos_ious_mask]
                pis = (paddle.exp(-(1 - pos_ious)**2 / 0.025) *
                       pos_scores)[:, None]
                voted_box = paddle.sum(
                    pis * pos_bboxes, dim=0) / paddle.sum(
                        pis, dim=0)
                voted_score = det_cls_bboxes[det_ind][-1:][None, :]
                det_bboxes_voted.append(
                    paddle.concat((voted_box[None, :], voted_score), dim=1))
                det_labels_voted.append(cls)

        det_bboxes_voted = paddle.concat(det_bboxes_voted, dim=0)
        det_labels_voted = det_labels.new_tensor(det_labels_voted)
        return det_bboxes_voted, det_labels_voted

