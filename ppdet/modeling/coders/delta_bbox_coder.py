import paddle
import numpy as np
from ppdet.core.workspace import register

__all__ = ['DeltaBBoxCoder']

def delta2bbox(rois,
               deltas,
               means=(0.0,0.0,0.0,0.0),
               stds=(1.0,1.0,1.0,1.0),
               max_shape=None,
               wh_ratio_clip=16.0/1000.0,
               ctr_clip=None):
    """Transform network output(delta) to bboxes. 
    Based on https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/
             bbox/coder/delta_xywh_bbox_coder.py
    Args:
        rois (Tensor): shape [..., 4], base bboxes, typical examples include
            anchor and rois
        deltas (Tensor): shape [..., 4], offset relative to base bboxes
        means (list[float]): the mean that was used to normalize deltas, 
            must be of size 4
        stds (list[float]): the std that was used to normalize deltas, 
            must be of size 4
        max_shape (list[float] or None): height and width of image, will be
            used to clip bboxes if not None
        wh_ratio_clip (float): to clip delta wh of decoded bboxes
        ctr_clip (float or None): whether to clip delta xy of decoded bboxes
    """
    if rois.size == 0:
        return paddle.empty_like(rois)
    means = paddle.to_tensor(means)
    stds = paddle.to_tensor(stds)
    deltas = deltas * stds + means

    dxy = deltas[..., :2]
    dwh = deltas[..., 2:]

    pxy = (rois[..., :2] + rois[..., 2:]) * 0.5
    pwh = rois[..., 2:] - rois[..., :2]
    dxy_wh = pwh * dxy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if ctr_clip is not None:
        dxy_wh = paddle.clip(dxy_wh, max=ctr_clip, min=-ctr_clip)
        dwh = paddle.clip(dwh, max=max_ratio)
    else:
        dwh = dwh.clip(min=-max_ratio, max=max_ratio)

    gxy = pxy + dxy_wh
    gwh = pwh * dwh.exp()
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    bboxes = paddle.concat([x1y1, x2y2], axis=-1)
    if max_shape is not None:
        bboxes[..., 0::2] = bboxes[..., 0::2].clip(min=0, max=max_shape[1])
        bboxes[..., 1::2] = bboxes[..., 1::2].clip(min=0, max=max_shape[0])
    return bboxes


def bbox2delta(src_boxes,
               tgt_boxes,
               means=(0.0, 0.0, 0.0, 0.0),
               stds=(1.0, 1.0, 1.0, 1.0)):
    """Encode bboxes to deltas.
    Modified from ppdet.modeling.bbox_utils.bbox2delta.
    Args:
        src_boxes (Tensor[..., 4]): base bboxes
        tgt_boxes (Tensor[..., 4]): target bboxes
        means (list[float]): the mean that will be used to normalize delta
        stds (list[float]): the std that will be used to normalize delta
    """
    if src_boxes.size == 0:
        return paddle.empty_like(src_boxes)
    src_w = src_boxes[..., 2] - src_boxes[..., 0]
    src_h = src_boxes[..., 3] - src_boxes[..., 1]
    src_ctr_x = src_boxes[..., 0] + 0.5 * src_w
    src_ctr_y = src_boxes[..., 1] + 0.5 * src_h

    tgt_w = tgt_boxes[..., 2] - tgt_boxes[..., 0]
    tgt_h = tgt_boxes[..., 3] - tgt_boxes[..., 1]
    tgt_ctr_x = tgt_boxes[..., 0] + 0.5 * tgt_w
    tgt_ctr_y = tgt_boxes[..., 1] + 0.5 * tgt_h

    dx = (tgt_ctr_x - src_ctr_x) / src_w
    dy = (tgt_ctr_y - src_ctr_y) / src_h
    dw = paddle.log(tgt_w / src_w)
    dh = paddle.log(tgt_h / src_h)

    deltas = paddle.stack((dx, dy, dw, dh), axis=1) # [n, 4]
    means = paddle.to_tensor(means, place=src_boxes.place)
    stds = paddle.to_tensor(stds, place=src_boxes.place)
    deltas = (deltas - means) / stds
    return deltas



@register
class DeltaBBoxCoder:
    """Encode bboxes in terms of delta/offset of a reference bbox.
    Args:
        norm_mean (list[float]): the mean to normalize delta
        norm_std (list[float]): the std to normalize delta
        wh_ratio_clip (float): to clip delta wh of decoded bboxes
        ctr_clip (float or None): whether to clip delta xy of decoded bboxes
    """
    def __init__(self,
                 norm_mean=[0.0, 0.0, 0.0, 0.0],
                 norm_std=[1., 1., 1., 1.],
                 wh_ratio_clip=16/1000.0,
                 ctr_clip=None):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.wh_ratio_clip = wh_ratio_clip
        self.ctr_clip = ctr_clip

    def encode(self, bboxes, tar_bboxes):
        return bbox2delta(
            bboxes, tar_bboxes, means=self.norm_mean, stds=self.norm_std)

    def decode(self, bboxes, deltas, max_shape=None):
        return delta2bbox(
            bboxes,
            deltas,
            max_shape=max_shape,
            wh_ratio_clip=self.wh_ratio_clip,
            ctr_clip=self.ctr_clip,
            means=self.norm_mean,
            stds=self.norm_std)
