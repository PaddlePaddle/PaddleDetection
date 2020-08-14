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

import paddle
import paddle.fluid as fluid
import os
import sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from ppdet.modeling.tests.decorator_helper import prog_scope
from ppdet.core.workspace import load_config, merge_config, create


class TestFasterRCNN(unittest.TestCase):
    def setUp(self):
        self.set_config()
        self.cfg = load_config(self.cfg_file)
        self.detector_type = self.cfg['architecture']

    def set_config(self):
        self.cfg_file = 'configs/faster_rcnn_r50_1x.yml'

    @prog_scope()
    def test_train(self):
        model = create(self.detector_type)
        inputs_def = self.cfg['TrainReader']['inputs_def']
        inputs_def['image_shape'] = [3, None, None]
        feed_vars, _ = model.build_inputs(**inputs_def)
        train_fetches = model.train(feed_vars)

    @prog_scope()
    def test_test(self):
        inputs_def = self.cfg['EvalReader']['inputs_def']
        inputs_def['image_shape'] = [3, None, None]
        model = create(self.detector_type)
        feed_vars, _ = model.build_inputs(**inputs_def)
        test_fetches = model.eval(feed_vars)


class TestMaskRCNN(TestFasterRCNN):
    def set_config(self):
        self.cfg_file = 'configs/mask_rcnn_r50_1x.yml'


@unittest.skip(
    reason="It should be fixed to adapt https://github.com/PaddlePaddle/Paddle/pull/23797"
)
class TestCascadeRCNN(TestFasterRCNN):
    def set_config(self):
        self.cfg_file = 'configs/cascade_rcnn_r50_fpn_1x.yml'


@unittest.skipIf(
    paddle.version.full_version < "1.8.4",
    "Paddle 2.0 should be used for YOLOv3 takes scale_x_y as inputs, "
    "disable this unittest for Paddle major version < 2")
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
