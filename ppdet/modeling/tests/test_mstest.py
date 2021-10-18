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
import unittest
from ppdet.core.workspace import load_config
from ppdet.engine import Trainer

class TestMultiScaleInference(unittest.TestCase):
    def setUp(self):
        self.set_config()

    def set_config(self):
        self.mstest_cfg_file = 'configs/faster_rcnn/faster_rcnn_r34_fpn_multiscaletest_1x_coco.yml'

    # test evaluation with multi scale test
    def test_eval_mstest(self):
        cfg = load_config(self.mstest_cfg_file)
        trainer = Trainer(cfg, mode='eval')

        if not os.path.exists(cfg.weights):
            cfg.weights = '/home/gbstack/Downloads/faster_rcnn_r34_fpn_1x_coco.pdparams'
        trainer.load_weights(cfg.weights)

        trainer.evaluate()

    # test inference with multi scale test
    def test_infer_mstest(self):
        cfg = load_config(self.mstest_cfg_file)
        trainer = Trainer(cfg, mode='test')

        if not os.path.exists(cfg.weights):
            cfg.weights = '/home/gbstack/Downloads/faster_rcnn_r34_fpn_1x_coco.pdparams'
        trainer.load_weights(cfg.weights)

        # input images to predict
        imgs = ['/dataset/coco2017/val2017/000000000139.jpg', '/dataset/coco2017/val2017/000000000724.jpg']
        trainer.predict(imgs,
                        draw_threshold=0.5,
                        output_dir='output',
                        save_txt=True)


if __name__ == '__main__':
    unittest.main()
