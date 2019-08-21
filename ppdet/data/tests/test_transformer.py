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
import time
import unittest
import sys
import logging
import numpy as np

import set_env
import ppdet.data.transform as tf
from ppdet.data.source import build_source

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


class TestTransformer(unittest.TestCase):
    """Test cases for dataset.transform.transformer
    """

    @classmethod
    def setUpClass(cls):
        """ setup
        """

        prefix = os.path.dirname(os.path.abspath(__file__))
        # json data
        anno_path = set_env.coco_data['TRAIN']['ANNO_FILE']
        image_dir = set_env.coco_data['TRAIN']['IMAGE_DIR']
        cls.sc_config = {
            'anno_file': anno_path,
            'image_dir': image_dir,
            'samples': 200
        }

        cls.ops = [{
            'op': 'DecodeImage',
            'to_rgb': True
        }, {
            'op': 'ResizeImage',
            'target_size': 800,
            'max_size': 1333
        }, {
            'op': 'ArrangeRCNN',
            'is_mask': False
        }]

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_map(self):
        """ test transformer.map
        """
        mapper = tf.build_mapper(self.ops)
        ds = build_source(self.sc_config)
        mapped_ds = tf.map(ds, mapper)
        ct = 0
        for sample in mapped_ds:
            self.assertTrue(type(sample[0]) is np.ndarray)
            ct += 1

        self.assertEqual(ct, mapped_ds.size())

    def test_parallel_map(self):
        """ test transformer.map with concurrent workers
        """
        mapper = tf.build_mapper(self.ops)
        ds = build_source(self.sc_config)
        worker_conf = {'WORKER_NUM': 2, 'use_process': True}
        mapped_ds = tf.map(ds, mapper, worker_conf)

        ct = 0
        for sample in mapped_ds:
            self.assertTrue(type(sample[0]) is np.ndarray)
            ct += 1

        self.assertTrue(mapped_ds.drained())
        self.assertEqual(ct, mapped_ds.size())
        mapped_ds.reset()

        ct = 0
        for sample in mapped_ds:
            self.assertTrue(type(sample[0]) is np.ndarray)
            ct += 1

        self.assertEqual(ct, mapped_ds.size())

    def test_batch(self):
        """ test batched dataset
        """
        batchsize = 2
        mapper = tf.build_mapper(self.ops)
        ds = build_source(self.sc_config)
        mapped_ds = tf.map(ds, mapper)
        batched_ds = tf.batch(mapped_ds, batchsize, True)
        for sample in batched_ds:
            out = sample
        self.assertEqual(len(out), batchsize)


if __name__ == '__main__':
    unittest.main()
