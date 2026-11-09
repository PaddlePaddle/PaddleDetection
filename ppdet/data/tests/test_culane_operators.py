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

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from ppdet.data.culane_utils import clip_lanes
from ppdet.data.source.culane import CULaneDataSet
from ppdet.data.transform import culane_operators
from ppdet.data.transform.culane_operators import (
    AddToHueAndSaturation, ChannelShuffle, CULaneAffine, CULaneDataProcess,
    CULaneResize, HorizontalFlip, MultiplyAndAddToBrightness, OneOfBlur)


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
        with patch.object(np.random, 'random', return_value=0.75):
            resized = CULaneResize(img_h=4, img_w=16).apply(self.sample)
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
        segments = clip_lanes([lane], height=8, width=8)
        first, second = sorted(segments, key=lambda segment: segment[:, 1].min())
        np.testing.assert_allclose(first, [[4, 1], [7.999, 1.99975]])
        np.testing.assert_allclose(second, [[7.999, 6.00025], [4, 7]])

    def test_data_process_keeps_training_mask(self):
        self.sample.update(full_img_path='lane.png', img_name='lane.png', im_id=0)
        mask = self.sample['mask']
        process = CULaneDataProcess(img_w=8, img_h=8, num_points=5, max_lanes=1)
        result = process.apply(self.sample.copy())
        np.testing.assert_array_equal(result['seg'], mask)
        self.sample.pop('mask')
        self.assertNotIn('seg', process.apply(self.sample))

    def test_annotation_error_propagates_once(self):
        self.sample['full_img_path'] = 'broken-lane.png'
        process = CULaneDataProcess(img_w=8, img_h=8, num_points=5, max_lanes=1)
        error = ValueError('invalid lane points')
        with patch.object(culane_operators, 'transform_annotation',
                          side_effect=error) as transform:
            with self.assertLogs(culane_operators.logger, level='ERROR') as logs:
                with self.assertRaises(ValueError) as raised:
                    process.apply(self.sample)
        transform.assert_called_once()
        self.assertIs(raised.exception, error)
        self.assertIn('broken-lane.png', logs.output[0])


class TestCULaneAugmentation(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(11).integers(
            0, 256, size=(12, 16, 3), dtype=np.uint8)

    def test_channel_shuffle(self):
        sample = {'image': self.image, 'lanes': [], 'mask': self.image[:, :, 0]}
        mask = sample['mask'].copy()
        with patch.object(np.random, 'permutation', return_value=[2, 0, 1]):
            result = ChannelShuffle(prob=1).apply(sample)
        np.testing.assert_array_equal(result['image'], self.image[:, :, [2, 0, 1]])
        np.testing.assert_array_equal(result['mask'], mask)
        self.assertEqual(result['lanes'], [])

    def test_brightness_in_each_colorspace_and_order(self):
        colorspaces = [
            (cv2.COLOR_RGB2YCrCb, cv2.COLOR_YCrCb2RGB, 0),
            (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB, 2),
            (cv2.COLOR_RGB2HLS, cv2.COLOR_HLS2RGB, 1),
            (cv2.COLOR_RGB2Lab, cv2.COLOR_Lab2RGB, 0),
            (cv2.COLOR_RGB2Luv, cv2.COLOR_Luv2RGB, 0),
            (cv2.COLOR_RGB2YUV, cv2.COLOR_YUV2RGB, 0),
        ]
        transform = MultiplyAndAddToBrightness(
            mul=(1.25, 1.25), add=(-2.6, -2.6), prob=1)
        for index, (forward, inverse, channel) in enumerate(colorspaces):
            for multiply_first in (True, False):
                with self.subTest(colorspace=index, multiply_first=multiply_first):
                    converted = cv2.cvtColor(self.image, forward)
                    values = converted[:, :, channel].astype(np.int16)
                    if multiply_first:
                        values = np.minimum(values * 1.25, 255).astype(np.int16)
                        values = np.maximum(values - 3, 0)
                    else:
                        values = np.minimum(np.maximum(values - 3, 0) * 1.25, 255)
                    converted[:, :, channel] = values
                    expected = cv2.cvtColor(converted, inverse)
                    order = 0.0 if multiply_first else 0.9
                    with patch.object(np.random, 'randint', return_value=index), \
                            patch.object(np.random, 'random', side_effect=[0.0, order]):
                        actual = transform.apply({'image': self.image})['image']
                    np.testing.assert_array_equal(actual, expected)

    def test_hue_wraps_and_saturation_clips(self):
        hsv = np.array([[[178, 250, 240], [2, 253, 220], [12, 6, 255]]],
                       dtype=np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        for delta, hue_delta in ((11, 7), (-11, -7)):
            with self.subTest(delta=delta):
                expected = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                expected[:, :, 0] = (expected[:, :, 0].astype(int) + hue_delta) % 180
                expected[:, :, 1] = np.clip(expected[:, :, 1].astype(int) + delta,
                                            0, 255)
                expected = cv2.cvtColor(expected, cv2.COLOR_HSV2RGB)
                actual = AddToHueAndSaturation(value=(delta, delta), prob=1).apply(
                    {'image': image})['image']
                np.testing.assert_array_equal(actual, expected)

    def test_motion_blur_uses_normalized_odd_kernel(self):
        transform = OneOfBlur(MotionBlur_k=(4, 4), prob=1)
        kernel = np.zeros((5, 5), dtype=np.float32)
        kernel[:, 2] = 0.2
        expected = cv2.filter2D(self.image, -1, kernel)
        with patch.object(np.random, 'random', side_effect=[0.0, 0.0, 0.5]), \
                patch.object(np.random, 'uniform', return_value=0.0):
            actual = transform.apply({'image': self.image})['image']
        np.testing.assert_array_equal(actual, expected)

    def test_median_blur_rounds_even_kernel_up(self):
        transform = OneOfBlur(MedianBlur_k=(4, 4), prob=1)
        with patch.object(np.random, 'random', side_effect=[0.0, 0.9]):
            actual = transform.apply({'image': self.image})['image']
        np.testing.assert_array_equal(actual, cv2.medianBlur(self.image, 5))


class TestCULaneDeploy(unittest.TestCase):
    def test_deploy_matches_reader_for_path_and_rgb_array(self):
        deploy_dir = Path(__file__).resolve().parents[3] / 'deploy' / 'python'
        with patch.object(sys, 'path', [str(deploy_dir), *sys.path]):
            import preprocess as deploy_preprocess

        bgr = np.random.default_rng(12).integers(
            0, 256, size=(12, 16, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            image_path = str(Path(directory) / 'lane.png')
            self.assertTrue(cv2.imwrite(image_path, bgr))
            dataset = CULaneDataSet(dataset_dir=directory, cut_height=2,
                                   list_path='unused.txt', split='test')
            dataset.data_infos = [dict(img_path=image_path, img_name='lane.png',
                                       lanes=[])]
            sample = dataset[0]
            self.assertNotIn('seg', sample)
            sample = CULaneResize(img_h=8, img_w=10, prob=1).apply(sample)
            expected = CULaneDataProcess(
                img_w=10, img_h=8, num_points=5, max_lanes=1).apply(sample)['image']

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for op in [deploy_preprocess.CULaneResize(8, 10, 2),
                       deploy_preprocess.CULaneResize(8, 10, 2, prob=0)]:
                for image in (image_path, rgb):
                    with patch.object(np.random, 'random',
                                      side_effect=AssertionError('random resize')):
                        actual, _ = deploy_preprocess.preprocess(image, [op])
                    np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
