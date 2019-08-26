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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

import paddle.fluid as fluid

from ppdet.modeling.tests.decorator_helper import prog_scope
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.modeling.model_input import create_feed


class TestFasterRCNN(unittest.TestCase):
    def setUp(self):
        self.set_config()
        self.cfg = load_config(self.cfg_file)
        self.detector_type = self.cfg['architecture']

    def set_config(self):
        self.cfg_file = 'configs/faster_rcnn_r50_1x.yml'

    @prog_scope()
    def test_train(self):
        train_feed = create(self.cfg['train_feed'])
        model = create(self.detector_type)
        _, feed_vars = create_feed(train_feed)
        train_fetches = model.train(feed_vars)

    @prog_scope()
    def test_test(self):
        test_feed = create(self.cfg['eval_feed'])
        model = create(self.detector_type)
        _, feed_vars = create_feed(test_feed)
        test_fetches = model.eval(feed_vars)


class TestMaskRCNN(TestFasterRCNN):
    def set_config(self):
        self.cfg_file = 'configs/mask_rcnn_r50_1x.yml'


class TestCascadeRCNN(TestFasterRCNN):
    def set_config(self):
        self.cfg_file = 'configs/cascade_rcnn_r50_fpn_1x.yml'


class TestYolov3(TestFasterRCNN):
    def set_config(self):
        self.cfg_file = 'configs/yolov3_darknet.yml'


class TestRetinaNet(TestFasterRCNN):
    def set_config(self):
        self.cfg_file = 'configs/retinanet_r50_fpn_1x.yml'


class TestSSD(TestFasterRCNN):
    def set_config(self):
        self.cfg_file = 'configs/ssd/ssd_mobilenet_v1_voc.yml'


if __name__ == '__main__':
    unittest.main()
