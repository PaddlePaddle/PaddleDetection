import numpy as np
from numbers import Integral
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from ppdet.core.workspace import register, serializable
from ppdet.py_op.target import generate_rpn_anchor_target, generate_proposal_target, generate_mask_target


@register
@serializable
class AnchorGenerator(object):
    def __init__(self,
                 anchor_sizes=[32, 64, 128, 256, 512],
                 aspect_ratios=[0.5, 1.0, 2.0],
                 stride=[16.0, 16.0],
                 variance=[1.0, 1.0, 1.0, 1.0]):
        super(AnchorGenerator, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.stride = stride
        self.variance = variance

    def __call__(self, inputs):
        outs = fluid.layers.anchor_generator(
            input=inputs,
            anchor_sizes=self.anchor_sizes,
            aspect_ratios=self.aspect_ratios,
            stride=self.stride,
            variance=self.variance)
        return outs


@register
@serializable
class RPNAnchorTargetGenerator(object):
    def __init__(self,
                 rpn_batch_size_per_im=256,
                 rpn_straddle_thresh=0.,
                 rpn_fg_fraction=0.5,
                 rpn_positive_overlap=0.7,
                 rpn_negative_overlap=0.3,
                 use_random=True):
        super(RPNAnchorTargetGenerator, self).__init__()
        self.rpn_batch_size_per_im = rpn_batch_size_per_im
        self.rpn_straddle_thresh = rpn_straddle_thresh
        self.rpn_fg_fraction = rpn_fg_fraction
        self.rpn_positive_overlap = rpn_positive_overlap
        self.rpn_negative_overlap = rpn_negative_overlap
        self.use_random = use_random

    def __call__(self, cls_logits, bbox_pred, anchor_box, gt_boxes, is_crowd,
                 im_info):
        anchor_box = anchor_box.numpy()
        gt_boxes = gt_boxes.numpy()
        is_crowd = is_crowd.numpy()
        im_info = im_info.numpy()

        loc_indexes, score_indexes, tgt_labels, tgt_bboxes, bbox_inside_weights = generate_rpn_anchor_target(
            anchor_box, gt_boxes, is_crowd, im_info, self.rpn_straddle_thresh,
            self.rpn_batch_size_per_im, self.rpn_positive_overlap,
            self.rpn_negative_overlap, self.rpn_fg_fraction, self.use_random)

        loc_indexes = to_variable(loc_indexes)
        score_indexes = to_variable(score_indexes)
        tgt_labels = to_variable(tgt_labels)
        tgt_bboxes = to_variable(tgt_bboxes)
        bbox_inside_weights = to_variable(bbox_inside_weights)

        loc_indexes.stop_gradient = True
        score_indexes.stop_gradient = True
        tgt_labels.stop_gradient = True

        cls_logits = fluid.layers.reshape(x=cls_logits, shape=(-1, ))
        bbox_pred = fluid.layers.reshape(x=bbox_pred, shape=(-1, 4))
        pred_cls_logits = fluid.layers.gather(cls_logits, score_indexes)
        pred_bbox_pred = fluid.layers.gather(bbox_pred, loc_indexes)

        return pred_cls_logits, pred_bbox_pred, tgt_labels, tgt_bboxes, bbox_inside_weights


@register
@serializable
class ProposalGenerator(object):
    __append_doc__ = True

    def __init__(self,
                 train_pre_nms_top_n=12000,
                 train_post_nms_top_n=2000,
                 infer_pre_nms_top_n=6000,
                 infer_post_nms_top_n=1000,
                 nms_thresh=.5,
                 min_size=.1,
                 eta=1.,
                 return_rois_num=True):
        super(ProposalGenerator, self).__init__()
        self.train_pre_nms_top_n = train_pre_nms_top_n
        self.train_post_nms_top_n = train_post_nms_top_n
        self.infer_pre_nms_top_n = infer_pre_nms_top_n
        self.infer_post_nms_top_n = infer_post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.eta = eta
        self.return_rois_num = return_rois_num

    def __call__(self,
                 scores,
                 bbox_deltas,
                 anchors,
                 variances,
                 im_info,
                 mode='train'):
        pre_nms_top_n = self.train_pre_nms_top_n if mode == 'train' else self.infer_pre_nms_top_n
        post_nms_top_n = self.train_post_nms_top_n if mode == 'train' else self.infer_post_nms_top_n
        outs = fluid.layers.generate_proposals(
            scores,
            bbox_deltas,
            im_info,
            anchors,
            variances,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=self.nms_thresh,
            min_size=self.min_size,
            eta=self.eta,
            return_rois_num=self.return_rois_num)
        return outs


@register
@serializable
class ProposalTargetGenerator(object):
    __shared__ = ['num_classes']

    def __init__(self,
                 batch_size_per_im=512,
                 fg_fraction=.25,
                 fg_thresh=.5,
                 bg_thresh_hi=.5,
                 bg_thresh_lo=0.,
                 bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
                 num_classes=81,
                 shuffle_before_sample=True,
                 is_cls_agnostic=False,
                 is_cascade_rcnn=False):
        super(ProposalTargetGenerator, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo
        self.bbox_reg_weights = bbox_reg_weights
        self.num_classes = num_classes
        self.use_random = shuffle_before_sample
        self.is_cls_agnostic = is_cls_agnostic,
        self.is_cascade_rcnn = is_cascade_rcnn

    def __call__(self, rpn_rois, rpn_rois_nums, gt_classes, is_crowd, gt_boxes,
                 im_info):
        rpn_rois = rpn_rois.numpy()
        rpn_rois_nums = rpn_rois_nums.numpy()
        gt_classes = gt_classes.numpy()
        gt_boxes = gt_boxes.numpy()
        is_crowd = is_crowd.numpy()
        im_info = im_info.numpy()
        outs = generate_proposal_target(
            rpn_rois, rpn_rois_nums, gt_classes, is_crowd, gt_boxes, im_info,
            self.batch_size_per_im, self.fg_fraction, self.fg_thresh,
            self.bg_thresh_hi, self.bg_thresh_lo, self.bbox_reg_weights,
            self.num_classes, self.use_random, self.is_cls_agnostic,
            self.is_cascade_rcnn)

        outs = [to_variable(v) for v in outs]
        for v in outs:
            v.stop_gradient = True
        return outs


@register
@serializable
class MaskTargetGenerator(object):
    __shared__ = ['num_classes']

    def __init__(self, num_classes=81, resolution=14):
        super(MaskTargetGenerator, self).__init__()
        self.num_classes = num_classes
        self.resolution = resolution

    def __call__(self, im_info, gt_classes, is_crowd, gt_segms, rois, rois_nums,
                 labels_int32):
        im_info = im_info.numpy()
        gt_classes = gt_classes.numpy()
        is_crowd = is_crowd.numpy()
        gt_segms = gt_segms.numpy()
        rois = rois.numpy()
        rois_nums = rois_nums.numpy()
        labels_int32 = labels_int32.numpy()
        outs = generate_mask_target(im_info, gt_classes, is_crowd, gt_segms,
                                    rois, rois_nums, labels_int32,
                                    self.num_classes, self.resolution)

        outs = [to_variable(v) for v in outs]
        for v in outs:
            v.stop_gradient = True
        return outs


@register
class RoIAlign(object):
    def __init__(self, resolution=7, spatial_scale=1. / 16, sampling_ratio=0):
        super(RoIAlign, self).__init__()
        if isinstance(resolution, Integral):
            resolution = [resolution, resolution]
        self.pooled_height = resolution[0]
        self.pooled_width = resolution[1]
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def __call__(self, inputs):
        cur_l = 0
        new_nums = [cur_l]
        rois_nums_np = inputs['rois_nums'].numpy()
        for l in rois_nums_np:
            cur_l += l
            new_nums.append(cur_l)
        nums_t = to_variable(np.asarray(new_nums))
        rois_feat = fluid.layers.roi_align(
            inputs['res4'],
            inputs['rois'],
            self.pooled_height,
            self.pooled_width,
            self.spatial_scale,
            rois_lod=nums_t)

        return {'rois_feat': rois_feat}


@register
class RoIPool(object):
    def __init__(self, resolution=7, spatial_scale=1. / 16):
        super(RoIPool, self).__init__()
        if isinstance(resolution, Integral):
            resolution = [resolution, resolution]
        self.pooled_height = resolution[0]
        self.pooled_width = resolution[1]
        self.spatial_scale = spatial_scale

    def __call__(self, inputs):
        cur_l = 0
        new_nums = [cur_l]
        rois_nums_np = inputs['rois_nums'].numpy()
        for l in rois_nums_np:
            cur_l += l
            new_nums.append(cur_l)
        nums_t = to_variable(np.asarray(new_nums))
        rois_feat = fluid.layers.roi_pool(
            inputs['res4'],
            inputs['rois'],
            self.pooled_height,
            self.pooled_width,
            self.spatial_scale,
            rois_nums=nums_t)

        return {'rois_feat': rois_feat}


@register
@serializable
class AnchorGrid(object):
    """Generate anchor grid

    Args:
        image_size (int or list): input image size, may be a single integer or
            list of [h, w]. Default: 512
        min_level (int): min level of the feature pyramid. Default: 3
        max_level (int): max level of the feature pyramid. Default: 7
        anchor_base_scale: base anchor scale. Default: 4
        num_scales: number of anchor scales. Default: 3
        aspect_ratios: aspect ratios. default: [[1, 1], [1.4, 0.7], [0.7, 1.4]]
    """

    def __init__(self,
                 image_size=512,
                 min_level=3,
                 max_level=7,
                 anchor_base_scale=4,
                 num_scales=3,
                 aspect_ratios=[[1, 1], [1.4, 0.7], [0.7, 1.4]]):
        super(AnchorGrid, self).__init__()
        if isinstance(image_size, Integral):
            self.image_size = [image_size, image_size]
        else:
            self.image_size = image_size
        for dim in self.image_size:
            assert dim % 2 ** max_level == 0, \
                "image size should be multiple of the max level stride"
        self.min_level = min_level
        self.max_level = max_level
        self.anchor_base_scale = anchor_base_scale
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios

    @property
    def base_cell(self):
        if not hasattr(self, '_base_cell'):
            self._base_cell = self.make_cell()
        return self._base_cell

    def make_cell(self):
        scales = [2**(i / self.num_scales) for i in range(self.num_scales)]
        scales = np.array(scales)
        ratios = np.array(self.aspect_ratios)
        ws = np.outer(scales, ratios[:, 0]).reshape(-1, 1)
        hs = np.outer(scales, ratios[:, 1]).reshape(-1, 1)
        anchors = np.hstack((-0.5 * ws, -0.5 * hs, 0.5 * ws, 0.5 * hs))
        return anchors

    def make_grid(self, stride):
        cell = self.base_cell * stride * self.anchor_base_scale
        x_steps = np.arange(stride // 2, self.image_size[1], stride)
        y_steps = np.arange(stride // 2, self.image_size[0], stride)
        offset_x, offset_y = np.meshgrid(x_steps, y_steps)
        offset_x = offset_x.flatten()
        offset_y = offset_y.flatten()
        offsets = np.stack((offset_x, offset_y, offset_x, offset_y), axis=-1)
        offsets = offsets[:, np.newaxis, :]
        return (cell + offsets).reshape(-1, 4)

    def generate(self):
        return [
            self.make_grid(2**l)
            for l in range(self.min_level, self.max_level + 1)
        ]

    def __call__(self):
        if not hasattr(self, '_anchor_vars'):
            anchor_vars = []
            helper = LayerHelper('anchor_grid')
            for idx, l in enumerate(range(self.min_level, self.max_level + 1)):
                stride = 2**l
                anchors = self.make_grid(stride)
                var = helper.create_parameter(
                    attr=ParamAttr(name='anchors_{}'.format(idx)),
                    shape=anchors.shape,
                    dtype='float32',
                    stop_gradient=True,
                    default_initializer=NumpyArrayInitializer(anchors))
                anchor_vars.append(var)
                var.persistable = True
            self._anchor_vars = anchor_vars

        return self._anchor_vars
