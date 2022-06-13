#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from ppdet.core.workspace import register
from ppdet.modeling import ops


def _to_list(v):
    if not isinstance(v, (list, tuple)):
        return [v]
    return v


@register
class RoIAlign(object):
    """
    RoI Align module

    For more details, please refer to the document of roi_align in
    in https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/vision/ops.py

    Args:
        resolution (int): The output size, default 14
        spatial_scale (float): Multiplicative spatial scale factor to translate
            ROI coords from their input scale to the scale used when pooling.
            default 0.0625
        sampling_ratio (int): The number of sampling points in the interpolation
            grid, default 0
        canconical_level (int): The referring level of FPN layer with 
            specified level. default 4
        canonical_size (int): The referring scale of FPN layer with 
            specified scale. default 224
        start_level (int): The start level of FPN layer to extract RoI feature,
            default 0
        end_level (int): The end level of FPN layer to extract RoI feature,
            default 3
        aligned (bool): Whether to add offset to rois' coord in roi_align.
            default false
    """

    def __init__(self,
                 resolution=14,
                 spatial_scale=0.0625,
                 sampling_ratio=0,
                 canconical_level=4,
                 canonical_size=224,
                 start_level=0,
                 end_level=3,
                 aligned=False):
        super(RoIAlign, self).__init__()
        self.resolution = resolution
        self.spatial_scale = _to_list(spatial_scale)
        self.sampling_ratio = sampling_ratio
        self.canconical_level = canconical_level
        self.canonical_size = canonical_size
        self.start_level = start_level
        self.end_level = end_level
        self.aligned = aligned

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'spatial_scale': [1. / i.stride for i in input_shape]}

    def __call__(self, feats, roi, rois_num):
        roi = paddle.concat(roi) if len(roi) > 1 else roi[0]
        if len(feats) == 1:
            rois_feat = paddle.vision.ops.roi_align(
                x=feats[self.start_level],
                boxes=roi,
                boxes_num=rois_num,
                output_size=self.resolution,
                spatial_scale=self.spatial_scale[0],
                aligned=self.aligned)
        else:
            offset = 2
            k_min = self.start_level + offset
            k_max = self.end_level + offset
            rois_dist, restore_index, rois_num_dist = ops.distribute_fpn_proposals(
                roi,
                k_min,
                k_max,
                self.canconical_level,
                self.canonical_size,
                rois_num=rois_num)
            rois_feat_list = []
            for lvl in range(self.start_level, self.end_level + 1):
                roi_feat = paddle.vision.ops.roi_align(
                    x=feats[lvl],
                    boxes=rois_dist[lvl],
                    boxes_num=rois_num_dist[lvl],
                    output_size=self.resolution,
                    spatial_scale=self.spatial_scale[lvl],
                    sampling_ratio=self.sampling_ratio,
                    aligned=self.aligned)
                rois_feat_list.append(roi_feat)
            rois_feat_shuffle = paddle.concat(rois_feat_list)
            rois_feat = paddle.gather(rois_feat_shuffle, restore_index)

        return rois_feat
