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


@register
class RoIAlign(object):
    def __init__(self,
                 resolution=14,
                 sampling_ratio=0,
                 canconical_level=4,
                 canonical_size=224,
                 start_level=0,
                 end_level=3):
        super(RoIAlign, self).__init__()
        self.resolution = resolution
        self.sampling_ratio = sampling_ratio
        self.canconical_level = canconical_level
        self.canonical_size = canonical_size
        self.start_level = start_level
        self.end_level = end_level

    def __call__(self, feats, rois, spatial_scale):
        roi, rois_num = rois
        if self.start_level == self.end_level:
            rois_feat = ops.roi_align(
                feats[self.start_level],
                roi,
                self.resolution,
                spatial_scale,
                rois_num=rois_num)
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
                roi_feat = ops.roi_align(
                    feats[lvl],
                    rois_dist[lvl],
                    self.resolution,
                    spatial_scale[lvl],
                    sampling_ratio=self.sampling_ratio,
                    rois_num=rois_num_dist[lvl])
                rois_feat_list.append(roi_feat)
            rois_feat_shuffle = paddle.concat(rois_feat_list)
            rois_feat = paddle.gather(rois_feat_shuffle, restore_index)

        return rois_feat
