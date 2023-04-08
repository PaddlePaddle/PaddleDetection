from .operators import BaseOperator, register_op
from ppdet.utils.logger import setup_logger
import math
import numpy as np
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from scipy.interpolate import InterpolatedUnivariateSpline
import paddle
import collections

logger = setup_logger(__name__)

__all__ = ["CULaneToTensor"]


@register_op
class CULaneToTensor(object):
    def __init__(self):
        super(CULaneToTensor, self).__init__()

    def __call__(self, sample):
        # print(len(sample))
        for key in sample.keys():
            sample[key] = paddle.to_tensor(sample[key])
        return sample
    
    # def apply(self, sample, context=None):
    #     for key in sample.keys():
    #         sample[key] = paddle.to_tensor(sample[key])
    #     return sample