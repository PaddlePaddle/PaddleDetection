# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

from ppdet.data.culane_utils import clip_lanes
from ppdet.data.transform.culane_operators import (CULaneAffine, CULaneResize,
                                                   HorizontalFlip)


class TestCULaneGeometry(unittest.TestCase):
    def setUp(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 1] = 3
        self.sample = {
            'image': np.repeat(mask[:, :, None], 3, axis=2),
            'mask': mask,
            'lanes': [np.array([[1, 2], [1, 5]], dtype=np.float32)],
        }

    def test_resize_then_flip(self):
        resized = CULaneResize(img_h=4, img_w=16, prob=1).apply(self.sample)
        np.testing.assert_array_equal(resized['lanes'][0], [[2, 1], [2, 2.5]])
        expected_mask = np.zeros((4, 16), dtype=np.uint8)
        expected_mask[1:3, 2:4] = 3
        np.testing.assert_array_equal(resized['mask'], expected_mask)
        np.testing.assert_array_equal(np.unique(resized['mask']), [0, 3])
        image = resized['image'].copy()

        flipped = HorizontalFlip(prob=1).apply(resized)
        np.testing.assert_array_equal(flipped['image'], image[:, ::-1])
        np.testing.assert_array_equal(flipped['mask'], expected_mask[:, ::-1])
        np.testing.assert_array_equal(flipped['lanes'][0], [[14, 1], [14, 2.5]])

    def test_affine_pixel_and_lane_centers(self):
        mask = self.sample['mask'].copy()
        transformed = CULaneAffine(
            translate_percent_x=(0, 0), translate_percent_y=(0, 0),
            rotate=(90, 90), scale=(1, 1), prob=1).apply(self.sample)
        expected_mask = np.rot90(mask, -1)
        np.testing.assert_array_equal(transformed['image'][:, :, 0],
                                      expected_mask)
        np.testing.assert_array_equal(transformed['mask'], expected_mask)
        np.testing.assert_allclose(transformed['lanes'][0], [[6, 1], [3, 1]],
                                   atol=1e-6)

    def test_clip_retains_separate_visible_segments(self):
        lane = np.array([[4, 1], [12, 3], [12, 5], [4, 7]], dtype=np.float32)
        first, second = clip_lanes([lane], height=8, width=8)
        np.testing.assert_allclose(first, [[4, 1], [7.999, 1.99975]])
        np.testing.assert_allclose(second, [[7.999, 6.00025], [4, 7]])


if __name__ == '__main__':
    unittest.main()
