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

import numpy as np
from numbers import Integral

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from ppdet.core.workspace import register, serializable

__all__ = [
    'AnchorGenerator', 'RPNTargetAssign', 'GenerateProposals', 'MultiClassNMS',
    'BBoxAssigner', 'MaskAssigner', 'RoIAlign', 'RoIPool', 'MultiBoxHead',
    'SSDOutputDecoder', 'RetinaTargetAssign', 'RetinaOutputDecoder', 'ConvNorm',
    'MultiClassSoftNMS'
]


def ConvNorm(input,
             num_filters,
             filter_size,
             stride=1,
             groups=1,
             norm_decay=0.,
             norm_type='affine_channel',
             norm_groups=32,
             dilation=1,
             lr_scale=1,
             freeze_norm=False,
             act=None,
             norm_name=None,
             initializer=None,
             name=None):
    fan = num_filters
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=((filter_size - 1) // 2) * dilation,
        dilation=dilation,
        groups=groups,
        act=None,
        param_attr=ParamAttr(
            name=name + "_weights",
            initializer=initializer,
            learning_rate=lr_scale),
        bias_attr=False,
        name=name + '.conv2d.output.1')

    norm_lr = 0. if freeze_norm else 1.
    pattr = ParamAttr(
        name=norm_name + '_scale',
        learning_rate=norm_lr * lr_scale,
        regularizer=L2Decay(norm_decay))
    battr = ParamAttr(
        name=norm_name + '_offset',
        learning_rate=norm_lr * lr_scale,
        regularizer=L2Decay(norm_decay))

    if norm_type in ['bn', 'sync_bn']:
        global_stats = True if freeze_norm else False
        out = fluid.layers.batch_norm(
            input=conv,
            act=act,
            name=norm_name + '.output.1',
            param_attr=pattr,
            bias_attr=battr,
            moving_mean_name=norm_name + '_mean',
            moving_variance_name=norm_name + '_variance',
            use_global_stats=global_stats)
        scale = fluid.framework._get_var(pattr.name)
        bias = fluid.framework._get_var(battr.name)
    elif norm_type == 'gn':
        out = fluid.layers.group_norm(
            input=conv,
            act=act,
            name=norm_name + '.output.1',
            groups=norm_groups,
            param_attr=pattr,
            bias_attr=battr)
        scale = fluid.framework._get_var(pattr.name)
        bias = fluid.framework._get_var(battr.name)
    elif norm_type == 'affine_channel':
        scale = fluid.layers.create_parameter(
            shape=[conv.shape[1]],
            dtype=conv.dtype,
            attr=pattr,
            default_initializer=fluid.initializer.Constant(1.))
        bias = fluid.layers.create_parameter(
            shape=[conv.shape[1]],
            dtype=conv.dtype,
            attr=battr,
            default_initializer=fluid.initializer.Constant(0.))
        out = fluid.layers.affine_channel(
            x=conv, scale=scale, bias=bias, act=act)
    if freeze_norm:
        scale.stop_gradient = True
        bias.stop_gradient = True
    return out


@register
@serializable
class AnchorGenerator(object):
    __op__ = fluid.layers.anchor_generator
    __append_doc__ = True

    def __init__(self,
                 stride=[16.0, 16.0],
                 anchor_sizes=[32, 64, 128, 256, 512],
                 aspect_ratios=[0.5, 1., 2.],
                 variance=[1., 1., 1., 1.]):
        super(AnchorGenerator, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.variance = variance
        self.stride = stride


@register
@serializable
class RPNTargetAssign(object):
    __op__ = fluid.layers.rpn_target_assign
    __append_doc__ = True

    def __init__(self,
                 rpn_batch_size_per_im=256,
                 rpn_straddle_thresh=0.,
                 rpn_fg_fraction=0.5,
                 rpn_positive_overlap=0.7,
                 rpn_negative_overlap=0.3,
                 use_random=True):
        super(RPNTargetAssign, self).__init__()
        self.rpn_batch_size_per_im = rpn_batch_size_per_im
        self.rpn_straddle_thresh = rpn_straddle_thresh
        self.rpn_fg_fraction = rpn_fg_fraction
        self.rpn_positive_overlap = rpn_positive_overlap
        self.rpn_negative_overlap = rpn_negative_overlap
        self.use_random = use_random


@register
@serializable
class GenerateProposals(object):
    __op__ = fluid.layers.generate_proposals
    __append_doc__ = True

    def __init__(self,
                 pre_nms_top_n=6000,
                 post_nms_top_n=1000,
                 nms_thresh=.5,
                 min_size=.1,
                 eta=1.):
        super(GenerateProposals, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.eta = eta


@register
class MaskAssigner(object):
    __op__ = fluid.layers.generate_mask_labels
    __append_doc__ = True
    __shared__ = ['num_classes']

    def __init__(self, num_classes=81, resolution=14):
        super(MaskAssigner, self).__init__()
        self.num_classes = num_classes
        self.resolution = resolution


@register
@serializable
class MultiClassNMS(object):
    __op__ = fluid.layers.multiclass_nms
    __append_doc__ = True

    def __init__(self,
                 score_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 nms_threshold=.5,
                 normalized=False,
                 nms_eta=1.0,
                 background_label=0):
        super(MultiClassNMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.background_label = background_label

@register
@serializable
class MultiClassSoftNMS(object):
    def __init__(self,
                 score_threshold=0.01,
                 keep_top_k=300,
                 softnms_sigma=0.5,
                 normalized=False,
                 background_label=0,
                ):
        super(MultiClassSoftNMS, self).__init__()
        self.score_threshold = score_threshold
        self.keep_top_k = keep_top_k
        self.softnms_sigma = softnms_sigma
        self.normalized = normalized
        self.background_label = background_label
    
    def __call__( self, bboxes, scores ):
        
        def create_tmp_var(program, name, dtype, shape, lod_leval):
            return program.current_block().create_var(name=name, 
                                                      dtype=dtype, 
                                                      shape=shape, 
                                                      lod_leval=lod_leval)
        
        def _soft_nms_for_cls(dets, sigma, thres):
            """soft_nms_for_cls"""
            dets_final = []
            while len(dets) > 0:
                maxpos = np.argmax(dets[:, 0])
                dets_final.append(dets[maxpos].copy())
                ts, tx1, ty1, tx2, ty2 = dets[maxpos]
                scores = dets[:, 0]
                x1 = dets[:, 1]
                y1 = dets[:, 2]
                x2 = dets[:, 3]
                y2 = dets[:, 4]
                eta = 0 if self.normalized else 1
                areas = (x2 - x1 + eta) * (y2 - y1 + eta)
                xx1 = np.maximum(tx1, x1)
                yy1 = np.maximum(ty1, y1)
                xx2 = np.minimum(tx2, x2)
                yy2 = np.minimum(ty2, y2)
                w = np.maximum(0.0, xx2 - xx1 + eta)
                h = np.maximum(0.0, yy2 - yy1 + eta)
                inter = w * h
                ovr = inter / (areas + areas[maxpos] - inter) 
                weight = np.exp(-(ovr * ovr) / sigma)
                scores = scores * weight
                idx_keep = np.where(scores >= thres)
                dets[:, 0] = scores
                dets = dets[idx_keep]
            dets_final = np.array(dets_final).reshape(-1, 5)
            return dets_final 

        def _soft_nms(bboxes, scores):
            bboxes = np.array(bboxes)
            scores = np.array(scores)
            class_nums = scores.shape[-1]
            
            softnms_thres = self.score_threshold
            softnms_sigma = self.softnms_sigma
            keep_top_k = self.keep_top_k
            
            cls_boxes = [[] for _ in range(class_nums)]
            cls_ids = [[] for _ in range(class_nums)]
                        
            start_idx = 1 if self.background_label == 0 else 0
            for j in range(start_idx, class_nums):
                inds = np.where(scores[:, j] >= softnms_thres)[0]
                scores_j = scores[inds, j]
                rois_j = bboxes[inds, j, :]
                dets_j = np.hstack((scores_j[:, np.newaxis], rois_j)).astype(np.float32, copy=False)
                cls_rank = np.argsort(-dets_j[:, 0])
                dets_j = dets_j[cls_rank]

                cls_boxes[j] = _soft_nms_for_cls( dets_j, sigma=softnms_sigma, thres=softnms_thres )
                cls_ids[j] = np.array( [j]*cls_boxes[j].shape[0] ).reshape(-1,1)
            
            cls_boxes = np.vstack(cls_boxes[start_idx:])
            cls_ids = np.vstack(cls_ids[start_idx:])
            pred_result = np.hstack( [cls_ids, cls_boxes] )

            # Limit to max_per_image detections **over all classes**
            image_scores = cls_boxes[:,0]
            if len(image_scores) > keep_top_k:
                image_thresh = np.sort(image_scores)[-keep_top_k]
                keep = np.where(cls_boxes[:, 0] >= image_thresh)[0]
                pred_result = pred_result[keep, :]
            
            res = fluid.LoDTensor()
            res.set_lod([[0, pred_result.shape[0]]])
            if pred_result.shape[0] == 0:
                pred_result = np.array( [[1]], dtype=np.float32 )
            res.set(pred_result, fluid.CPUPlace())
            
            return res
        
        pred_result = create_tmp_var(fluid.default_main_program(), 
                                     name='softnms_pred_result', 
                                     dtype='float32', 
                                     shape=[6],
                                     lod_leval=1)
        fluid.layers.py_func(func=_soft_nms,
                        x=[bboxes, scores], out=pred_result)
        return pred_result


@register
class BBoxAssigner(object):
    __op__ = fluid.layers.generate_proposal_labels
    __append_doc__ = True
    __shared__ = ['num_classes']

    def __init__(self,
                 batch_size_per_im=512,
                 fg_fraction=.25,
                 fg_thresh=.5,
                 bg_thresh_hi=.5,
                 bg_thresh_lo=0.,
                 bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
                 num_classes=81,
                 shuffle_before_sample=True):
        super(BBoxAssigner, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo
        self.bbox_reg_weights = bbox_reg_weights
        self.class_nums = num_classes
        self.use_random = shuffle_before_sample


@register
class RoIAlign(object):
    __op__ = fluid.layers.roi_align
    __append_doc__ = True

    def __init__(self, resolution=7, spatial_scale=1. / 16, sampling_ratio=0):
        super(RoIAlign, self).__init__()
        if isinstance(resolution, Integral):
            resolution = [resolution, resolution]
        self.pooled_height = resolution[0]
        self.pooled_width = resolution[1]
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio


@register
class RoIPool(object):
    __op__ = fluid.layers.roi_pool
    __append_doc__ = True

    def __init__(self, resolution=7, spatial_scale=1. / 16):
        super(RoIPool, self).__init__()
        if isinstance(resolution, Integral):
            resolution = [resolution, resolution]
        self.pooled_height = resolution[0]
        self.pooled_width = resolution[1]
        self.spatial_scale = spatial_scale


@register
class MultiBoxHead(object):
    __op__ = fluid.layers.multi_box_head
    __append_doc__ = True

    def __init__(self,
                 min_ratio=20,
                 max_ratio=90,
                 base_size=300,
                 min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
                 max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
                 aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2., 3.],
                                [2., 3.]],
                 steps=None,
                 offset=0.5,
                 flip=True,
                 min_max_aspect_ratios_order=False,
                 kernel_size=1,
                 pad=0):
        super(MultiBoxHead, self).__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.base_size = base_size
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.aspect_ratios = aspect_ratios
        self.steps = steps
        self.offset = offset
        self.flip = flip
        self.min_max_aspect_ratios_order = min_max_aspect_ratios_order
        self.kernel_size = kernel_size
        self.pad = pad


@register
@serializable
class SSDOutputDecoder(object):
    __op__ = fluid.layers.detection_output
    __append_doc__ = True

    def __init__(self,
                 nms_threshold=0.45,
                 nms_top_k=400,
                 keep_top_k=200,
                 score_threshold=0.01,
                 nms_eta=1.0,
                 background_label=0):
        super(SSDOutputDecoder, self).__init__()
        self.nms_threshold = nms_threshold
        self.background_label = background_label
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.score_threshold = score_threshold
        self.nms_eta = nms_eta


@register
@serializable
class RetinaTargetAssign(object):
    __op__ = fluid.layers.retinanet_target_assign
    __append_doc__ = True

    def __init__(self, positive_overlap=0.5, negative_overlap=0.4):
        super(RetinaTargetAssign, self).__init__()
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap


@register
@serializable
class RetinaOutputDecoder(object):
    __op__ = fluid.layers.retinanet_detection_output
    __append_doc__ = True

    def __init__(self,
                 score_thresh=0.05,
                 nms_thresh=0.3,
                 pre_nms_top_n=1000,
                 detections_per_im=100,
                 nms_eta=1.0):
        super(RetinaOutputDecoder, self).__init__()
        self.score_threshold = score_thresh
        self.nms_threshold = nms_thresh
        self.nms_top_k = pre_nms_top_n
        self.keep_top_k = detections_per_im
        self.nms_eta = nms_eta
