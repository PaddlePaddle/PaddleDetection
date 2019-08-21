#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import os
import unittest
import logging
import numpy as np
import set_env
import ppdet.data.transform as tf
logging.basicConfig(level=logging.INFO)


class TestBase(unittest.TestCase):
    """Test cases for dataset.transform.operator
    """

    @classmethod
    def setUpClass(cls, with_mixup=False):
        """ setup
        """
        roidb_fname = set_env.coco_data['TRAIN']['ANNO_FILE']
        image_dir = set_env.coco_data['TRAIN']['IMAGE_DIR']
        import pickle as pkl
        with open(roidb_fname, 'rb') as f:
            roidb = f.read()
            roidb = pkl.loads(roidb)
        fn = os.path.join(image_dir, roidb[0][0]['im_file'])
        with open(fn, 'rb') as f:
            roidb[0][0]['image'] = f.read()
        if with_mixup:
            mixup_fn = os.path.join(image_dir, roidb[0][1]['im_file'])
            roidb[0][0]['mixup'] = roidb[0][1]
            with open(fn, 'rb') as f:
                roidb[0][0]['mixup']['image'] = f.read()
        cls.sample = roidb[0][0]

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_ops_all(self):
        """ test operators
        """
        # ResizeImage
        ops_conf = [{
            'op': 'DecodeImage'
        }, {
            'op': 'ResizeImage',
            'target_size': 300,
            'max_size': 1333
        }]
        mapper = tf.build_mapper(ops_conf)
        self.assertTrue(mapper is not None)
        data = self.sample.copy()
        result0 = mapper(data)
        self.assertIsNotNone(result0['image'])
        self.assertEqual(len(result0['image'].shape), 3)
        # RandFlipImage
        ops_conf = [{'op': 'RandomFlipImage'}]
        mapper = tf.build_mapper(ops_conf)
        self.assertTrue(mapper is not None)
        result1 = mapper(result0)
        self.assertEqual(result1['image'].shape, result0['image'].shape)
        self.assertEqual(result1['gt_bbox'].shape, result0['gt_bbox'].shape)
        # NormalizeImage
        ops_conf = [{'op': 'NormalizeImage', 'is_channel_first': False}]
        mapper = tf.build_mapper(ops_conf)
        self.assertTrue(mapper is not None)
        result2 = mapper(result1)
        im1 = result1['image']
        count = np.where(im1 <= 1)[0]
        if im1.dtype == 'float64':
            self.assertEqual(count, im1.shape[0] * im1.shape[1], im1.shape[2])
        # ArrangeSample
        ops_conf = [{'op': 'ArrangeRCNN'}]
        mapper = tf.build_mapper(ops_conf)
        self.assertTrue(mapper is not None)
        result3 = mapper(result2)
        self.assertEqual(type(result3), tuple)

    def test_ops_part1(self):
        """test Crop and Resize
        """
        ops_conf = [{
            'op': 'DecodeImage'
        }, {
            'op': 'NormalizeBox'
        }, {
            'op': 'CropImage',
            'batch_sampler': [[1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                              [1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 0.0],
                              [1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 0.0],
                              [1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 0.0],
                              [1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 0.0],
                              [1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 0.0],
                              [1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0]]
        }]
        mapper = tf.build_mapper(ops_conf)
        self.assertTrue(mapper is not None)
        data = self.sample.copy()
        result = mapper(data)
        self.assertEqual(len(result['image'].shape), 3)

    def test_ops_part2(self):
        """test Expand and RandomDistort
        """
        ops_conf = [{
            'op': 'DecodeImage'
        }, {
            'op': 'NormalizeBox'
        }, {
            'op': 'ExpandImage',
            'max_ratio': 1.5,
            'prob': 1
        }]
        mapper = tf.build_mapper(ops_conf)
        self.assertTrue(mapper is not None)
        data = self.sample.copy()
        result = mapper(data)
        self.assertEqual(len(result['image'].shape), 3)
        self.assertGreater(result['gt_bbox'].shape[0], 0)

    def test_ops_part3(self):
        """test Mixup and RandomInterp
        """
        ops_conf = [{
            'op': 'DecodeImage',
            'with_mixup': True,
        }, {
            'op': 'MixupImage',
        }, {
            'op': 'RandomInterpImage',
            'target_size': 608
        }]
        mapper = tf.build_mapper(ops_conf)
        self.assertTrue(mapper is not None)
        data = self.sample.copy()
        result = mapper(data)
        self.assertEqual(len(result['image'].shape), 3)
        self.assertGreater(result['gt_bbox'].shape[0], 0)
        #self.assertGreater(result['gt_score'].shape[0], 0)


if __name__ == '__main__':
    unittest.main()
