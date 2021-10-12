#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle
import os
import sys

from ppdet.modeling import BBoxHead, TwoFCHead
from ppdet.modeling.proposal_generator.target_layer import OHEMBBoxAssigner

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import numpy as np


class TestOHEMBBoxAssigner(unittest.TestCase):
    def setUp(self):
        self.im_width=600
        self.im_height=600

    def test_assign(self):

        bbox_head = BBoxHead(TwoFCHead(), 1024,
                             roi_extractor={
                                 'resolution': 7,
                                 'sampling_ratio': 0,
                                 'aligned': True,
                                 'spatial_scale': [0.25, 0.125, 0.0625, 0.03125, 0.015625]
                             })
        assigner = OHEMBBoxAssigner()
        rois = [paddle.to_tensor(np.random.random((1000,4)).astype(np.float32))]
        rois_num = paddle.to_tensor(np.array([1000,]))
        body_feats = [
            paddle.to_tensor(np.random.random((1,256,256,256)).astype(np.float32)),
            paddle.to_tensor(np.random.random((1,256,128,128)).astype(np.float32)),
            paddle.to_tensor(np.random.random((1,256,64,64)).astype(np.float32)),
            paddle.to_tensor(np.random.random((1,256,32,32)).astype(np.float32)),
            paddle.to_tensor(np.random.random((1,256,16,16)).astype(np.float32)),
        ]

        inputs = {
            'im_id': paddle.to_tensor(np.array([3576])),
            'is_crowd': [paddle.to_tensor(np.array([0]))],
            'gt_class': [paddle.to_tensor(np.array([1]).reshape(-1,1))],
            'gt_bbox': [paddle.to_tensor(np.array([200,150,230,180], dtype=np.float32).reshape(1,4))],
            'image': paddle.to_tensor(np.random.random((1, 3, self.im_height, self.im_width))),
        }

        rois, rois_num, targets = assigner(rois, rois_num, inputs, bbox_head, body_feats)
        print(rois.shape, rois_num, targets.shape)


if __name__ == "__main__":
    unittest.main()
