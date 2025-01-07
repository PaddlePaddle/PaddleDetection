# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""
this code is base on https://github.com/Sense-X/Co-DETR/blob/main/projects/models/co_atss_head.py
"""
from functools import partial

import paddle
import paddle.nn as nn

from ppdet.modeling import bbox_utils
from ppdet.modeling.initializer import normal_, constant_, bias_init_with_prob
from ppdet.core.workspace import register

__all__ = ['CoATSSHead']


class Scale(nn.Layer):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = paddle.create_parameter(paddle.to_tensor(scale, dtype='float32').shape,dtype='float32')

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return x * self.scale
    
def reduce_mean(tensor):
    world_size = paddle.distributed.get_world_size()
    if world_size == 1:
        return tensor
    paddle.distributed.all_reduce(tensor)
    return tensor / world_size

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    res = tuple(map(list, zip(*map_results)))
    return res

@register
class CoATSSHead(nn.Layer):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
    """
    __inject__ = ['anchor_generator','loss_cls', 'loss_bbox', "assigner"]
    __shared__ = ['num_classes']
    def __init__(self,                  
                 in_channels,
                 num_classes=80,
                 stacked_convs=4,
                 feat_channels=256,
                 anchor_generator=None,
                 assigner='ATSSAssigner',
                 bbox_weight=[1.0, 1.0, 1.0, 1.0],
                 loss_cent_weight=1.0,
                 loss_cls=None,
                 loss_bbox=None,
                 reg_decoded_bbox=True,
                 pos_weight=-1
                 ):
        super().__init__()
        self.num_classes=num_classes
        self.in_channels=in_channels
        self.stacked_convs=stacked_convs
        self.feat_channels=feat_channels
        self.anchor_generator=anchor_generator
        self.num_levels=len(self.anchor_generator.strides)
        self.num_anchors = self.anchor_generator.num_anchors
        self.use_sigmoid_cls = True
        self.loss_cls=loss_cls
        self.loss_bbox=loss_bbox
        self.loss_centerness=nn.BCEWithLogitsLoss(reduction='sum')
        self.loss_cent_weight=loss_cent_weight
        self.assigner = assigner
        self.bbox_weight = bbox_weight
        self.reg_decoded_bbox=reg_decoded_bbox
        self.pos_weight=pos_weight
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self._init_layers()
        self.apply(self._init_weights)
        constant_(self.atss_cls.bias, bias_init_with_prob(0.01))
        self.ass_time = 0
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            normal_(m.weight, 0, 0.01)
            constant_(m.bias, 0)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                nn.Sequential(
                    nn.Conv2D(chn, self.feat_channels, 3, padding=1), 
                    nn.GroupNorm(32,self.feat_channels),
                    nn.ReLU()))
                
            self.reg_convs.append(
                nn.Sequential(
                    nn.Conv2D(chn, self.feat_channels, 3, padding=1), 
                    nn.GroupNorm(32,self.feat_channels),
                    nn.ReLU()))
        self.atss_cls = nn.Conv2D(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.atss_reg = nn.Conv2D(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.atss_centerness = nn.Conv2D(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)
        self.scales = nn.LayerList(
            [Scale(1.0) for _ in self.anchor_generator.strides])
    
    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale : Learnable scale module to resize the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
    
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).astype('float32')
        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness
        
    def loss_single(self, anchors, cls_score, bbox_pred, centerness, labels,
                    label_weights, bbox_targets, img_metas, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        anchors = anchors.reshape((-1, 4))
        cls_score = cls_score.transpose((0, 2, 3, 1)).reshape(
            (-1, self.cls_out_channels))
        bbox_pred = bbox_pred.transpose((0, 2, 3, 1)).reshape((-1, 4))
        centerness = centerness.transpose((0, 2, 3, 1)).reshape([-1])
        bbox_targets = bbox_targets.reshape((-1, 4))
        labels = labels.reshape([-1]).astype(paddle.int32)
        label_weights = label_weights.reshape([-1])
        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = paddle.nonzero(
                paddle.logical_and((labels >= 0), (labels < bg_class_ind)),
                as_tuple=False).squeeze(1)
        
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = bbox_utils.delta2bbox(
                pos_bbox_pred, pos_anchors, self.bbox_weight).squeeze()

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_bbox_targets,
                iou_weight=centerness_targets.unsqueeze(-1))

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                centerness_targets) / num_total_samples * self.loss_cent_weight
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = paddle.to_tensor(0., dtype=bbox_targets.dtype)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    def centerness_target(self, anchors, gts):
        # only calculate pos centerness targets, otherwise there may be nan
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = paddle.stack([l_, r_], axis=1)
        top_bottom = paddle.stack([t_, b_], axis=1)
        centerness = paddle.sqrt(
            (left_right.min(axis=-1) / left_right.max(axis=-1)) *
            (top_bottom.min(axis=-1) / top_bottom.max(axis=-1)))
        assert not paddle.isnan(centerness).any()
        return centerness
    
    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator(
            featmap_sizes)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['img_shape'])
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list
    
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.shape[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.num_levels
        anchor_list, valid_flag_list = self.get_anchors(
            cls_scores, img_metas)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg,
         ori_anchors, ori_labels, ori_bbox_targets) = cls_reg_targets
        num_total_samples = reduce_mean(
            paddle.to_tensor(num_total_pos, dtype=paddle.float32)).item()
        num_total_samples = max(num_total_samples, 1.0)
        new_img_metas = [img_metas for _ in range(len(anchor_list))]
        losses_cls, losses_bbox, loss_centerness,\
            bbox_avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                centernesses,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                new_img_metas,
                num_total_samples=num_total_samples)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clip_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))

        pos_coords = (ori_anchors, ori_labels, ori_bbox_targets, 'atss')
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_centerness=loss_centerness,
            pos_coords=pos_coords)
        
        
    def images_to_levels(self, target, num_level_anchors):
        """
        Convert targets by image to targets by feature level.
        """
        target = paddle.stack(target, 0)
        level_targets = []
        start = 0
        for n in num_level_anchors:
            end = start + n                             
            level_targets.append(target[:, start:end].squeeze(0))
            start = end
        return level_targets


    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.shape[0] for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = paddle.concat(anchor_list[i])
            valid_flag_list[i] = paddle.concat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        ori_anchors = all_anchors
        ori_labels = all_labels
        ori_bbox_targets = all_bbox_targets
        anchors_list = self.images_to_levels(all_anchors, num_level_anchors)
        labels_list = self.images_to_levels(all_labels, num_level_anchors)
        label_weights_list = self.images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = self.images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = self.images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg, ori_anchors, ori_labels, ori_bbox_targets)
        
    def anchor_inside_flags(self,flat_anchors,
                        valid_flags,
                        img_shape,
                        allowed_border=0):
        """Check whether the anchors are inside the border.

        Args:
            flat_anchors (torch.Tensor): Flatten anchors, shape (n, 4).
            valid_flags (torch.Tensor): An existing valid flags of anchors.
            img_shape (tuple(int)): Shape of current image.
            allowed_border (int, optional): The border to allow the valid anchor.
                Defaults to 0.

        Returns:
            torch.Tensor: Flags indicating whether the anchors are inside a \
                valid range.
        """
        img_h, img_w = img_shape[:2]
        if allowed_border >= 0:
            inside_flags = valid_flags & \
                (flat_anchors[:, 0] >= -allowed_border) & \
                (flat_anchors[:, 1] >= -allowed_border) & \
                (flat_anchors[:, 2] < img_w + allowed_border) & \
                (flat_anchors[:, 3] < img_h + allowed_border)
        else:
            inside_flags = valid_flags
        return inside_flags
    

    def unmap(self, data, count, inds, fill=0):
        """Unmap a subset of item (data) back to the original set of items (of size
        count)"""
        if data.dim() == 1:
            ret = paddle.full((count,1), fill, dtype=data.dtype)
            data=data.unsqueeze(0).transpose((1,0))
            ret[inds,:] = data
            ret=ret.transpose((1,0)).squeeze()
        else:
            new_size = (count, ) + tuple(data.shape[1:])
            ret = paddle.full(new_size, fill, dtype=data.dtype)
            ret[inds.astype(paddle.bool), :] = data
        return ret


    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = paddle.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
    
    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = self.anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           -1).astype(paddle.bool)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        
        bg_index = self.num_classes
        pad_gt_mask = paddle.ones_like(gt_labels.unsqueeze(0), dtype=paddle.float32)
        assigned_labels, assigned_bboxes, _ = self.assigner(
            anchors, num_level_anchors_inside, gt_labels.unsqueeze(0), 
            gt_bboxes.unsqueeze(0), pad_gt_mask, bg_index)
        labels, bbox_targets = assigned_labels.squeeze(0), assigned_bboxes.squeeze(0)
        num_valid_anchors = anchors.shape[0]
        bbox_weights = paddle.zeros_like(anchors)
        pos_inds = paddle.nonzero(paddle.logical_and((labels >= 0), (labels < bg_index)), as_tuple=False).squeeze(-1).unique()
        neg_inds = paddle.nonzero(labels == bg_index, as_tuple=False).squeeze(-1).unique()
        
        label_weights = paddle.zeros((num_valid_anchors, ), dtype=paddle.float32)
        if len(pos_inds) > 0:
            bbox_weights[pos_inds, :] = 1.0
            if self.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
            bbox_targets[neg_inds, :] = 0.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.shape[0]
            anchors = self.unmap(anchors, num_total_anchors, inside_flags)
            labels = self.unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)

            label_weights = self.unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = self.unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = self.unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)
        
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        return losses
