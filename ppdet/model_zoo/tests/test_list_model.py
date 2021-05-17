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

import unittest
import ppdet


class TestListModel(unittest.TestCase):
    def setUp(self):
        self._filter = []

    def test_main(self):
        try:
            ppdet.model_zoo.list_model(self._filter)
            self.assertTrue(True)
        except:
            self.assertTrue(False)


class TestListModelYOLO(TestListModel):
    def setUp(self):
        self._filter = ['yolo']


class TestListModelRCNN(TestListModel):
    def setUp(self):
        self._filter = ['rcnn']


class TestListModelSSD(TestListModel):
    def setUp(self):
        self._filter = ['ssd']


class TestListModelMultiFilter(TestListModel):
    def setUp(self):
        self._filter = ['yolo', 'darknet']


class TestListModelError(unittest.TestCase):
    def setUp(self):
        self._filter = ['xxx']

    def test_main(self):
        try:
            ppdet.model_zoo.list_model(self._filter)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
