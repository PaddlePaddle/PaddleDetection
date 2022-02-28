import paddle
import numpy as np
from ppdet.core.workspace import register
from ppdet.modeling.bbox_utils import delta2bbox_v2, bbox2delta_v2

__all__ = ['DeltaBBoxCoder']


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
        return bbox2delta_v2(
            bboxes, tar_bboxes, means=self.norm_mean, stds=self.norm_std)

    def decode(self, bboxes, deltas, max_shape=None):
        return delta2bbox_v2(
            bboxes,
            deltas,
            max_shape=max_shape,
            wh_ratio_clip=self.wh_ratio_clip,
            ctr_clip=self.ctr_clip,
            means=self.norm_mean,
            stds=self.norm_std)
