# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import time
import unittest
import sys
import logging
import numpy as np

import set_env


class TestLoader(unittest.TestCase):
    """Test cases for dataset.source.loader
    """

    @classmethod
    def setUpClass(cls):
        """ setup
        """
        cls.prefix = os.path.dirname(os.path.abspath(__file__))
        # json data
        cls.anno_path = os.path.join(cls.prefix,
                                     'data/coco/instances_val2017.json')
        cls.image_dir = os.path.join(cls.prefix, 'data/coco/val2017')
        cls.anno_path1 = os.path.join(cls.prefix,
                                      "data/voc/ImageSets/Main/train.txt")
        cls.image_dir1 = os.path.join(cls.prefix, "data/voc/JPEGImages")

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_load_coco_in_json(self):
        """ test loading COCO data in json file
        """
        from ppdet.data.source.coco_loader import load
        if not os.path.exists(self.anno_path):
            logging.warn('not found %s, so skip this test' % (self.anno_path))
            return
        samples = 10
        records, cname2id = load(self.anno_path, samples)
        self.assertEqual(len(records), samples)
        self.assertGreater(len(cname2id), 0)

    def test_load_coco_in_roidb(self):
        """ test loading COCO data in pickled records
        """
        anno_path = os.path.join(self.prefix,
                                 'data/roidbs/instances_val2017.roidb')

        if not os.path.exists(anno_path):
            logging.warn('not found %s, so skip this test' % (anno_path))
            return

        samples = 10
        from ppdet.data.source.loader import load_roidb
        records, cname2cid = load_roidb(anno_path, samples)
        self.assertEqual(len(records), samples)
        self.assertGreater(len(cname2cid), 0)

    def test_load_voc_in_xml(self):
        """ test loading VOC data in xml files
        """
        from ppdet.data.source.voc_loader import load
        if not os.path.exists(self.anno_path1):
            logging.warn('not found %s, so skip this test' % (self.anno_path1))
            return
        samples = 3
        records, cname2cid = load(self.anno_path1, samples)
        self.assertEqual(len(records), samples)
        self.assertGreater(len(cname2cid), 0)

    def test_load_voc_in_roidb(self):
        """ test loading VOC data in pickled records
        """
        anno_path = os.path.join(self.prefix, 'data/roidbs/train.roidb')

        if not os.path.exists(anno_path):
            logging.warn('not found %s, so skip this test' % (anno_path))
            return

        samples = 3
        from ppdet.data.source.loader import load_roidb
        records, cname2cid = load_roidb(anno_path, samples)
        self.assertEqual(len(records), samples)
        self.assertGreater(len(cname2cid), 0)


if __name__ == '__main__':
    unittest.main()
