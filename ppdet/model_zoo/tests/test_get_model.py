#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import paddle
import ppdet
import unittest

# NOTE: weights downloading costs time, we choose
#       a small model for unittesting
MODEL_NAME = 'ppyolo/ppyolo_tiny_650e_coco'


class TestGetConfigFile(unittest.TestCase):
    def test_main(self):
        try:
            cfg_file = ppdet.model_zoo.get_config_file(MODEL_NAME)
            assert os.path.isfile(cfg_file)
        except:
            self.assertTrue(False)


class TestGetModel(unittest.TestCase):
    def test_main(self):
        try:
            model = ppdet.model_zoo.get_model(MODEL_NAME)
            assert isinstance(model, paddle.nn.Layer)
        except:
            self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
