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

import unittest
import os
import yaml
import logging
import sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from ppdet.utils.download import get_path
from ppdet.utils.download import DATASET_HOME
from ppdet.core.workspace import load_config, merge_config

from ppdet.data.reader import create_reader
from ppdet.utils.check import enable_static_mode

COCO_VAL_URL = 'http://images.cocodataset.org/zips/val2017.zip'
COCO_VAL_MD5SUM = '442b8da7639aecaf257c1dceb8ba8c80'
COCO_ANNO_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
COCO_ANNO_MD5SUM = 'f4bbac642086de4f52a3fdda2de5fa2c'

FORMAT = '[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class TestReaderYAML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ setup
        """
        root_path = os.path.join(DATASET_HOME, 'coco')
        _, _ = get_path(COCO_VAL_URL, root_path, COCO_VAL_MD5SUM)
        _, _ = get_path(COCO_ANNO_URL, root_path, COCO_ANNO_MD5SUM)
        cls.anno_path = 'annotations/instances_val2017.json'
        cls.image_dir = 'val2017'
        cls.root_path = root_path

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_loader_yaml(self):
        cfg_file = 'ppdet/data/tests/test.yml'
        cfg = load_config(cfg_file)
        data_cfg = '[!COCODataSet {{image_dir: {0}, dataset_dir: {1}, ' \
            'anno_path: {2}, sample_num: 10}}]'.format(
                self.image_dir, self.root_path, self.anno_path)
        dataset_ins = yaml.load(data_cfg, Loader=yaml.Loader)
        update_train_cfg = {'TrainReader': {'dataset': dataset_ins[0]}}
        update_test_cfg = {'EvalReader': {'dataset': dataset_ins[0]}}
        merge_config(update_train_cfg)
        merge_config(update_test_cfg)

        reader = create_reader(cfg['TrainReader'], 10)()
        for samples in reader:
            for sample in samples:
                im_shape = sample[0].shape
                self.assertEqual(im_shape[0], 3)
                self.assertEqual(im_shape[1] % 32, 0)
                self.assertEqual(im_shape[2] % 32, 0)

                im_info_shape = sample[1].shape
                self.assertEqual(im_info_shape[-1], 3)

                im_id_shape = sample[2].shape
                self.assertEqual(im_id_shape[-1], 1)

                gt_bbox_shape = sample[3].shape
                self.assertEqual(gt_bbox_shape[-1], 4)

                gt_class_shape = sample[4].shape
                self.assertEqual(gt_class_shape[-1], 1)
                self.assertEqual(gt_class_shape[0], gt_bbox_shape[0])

                is_crowd_shape = sample[5].shape
                self.assertEqual(is_crowd_shape[-1], 1)
                self.assertEqual(is_crowd_shape[0], gt_bbox_shape[0])

                mask = sample[6]
                self.assertEqual(len(mask), gt_bbox_shape[0])
                self.assertEqual(mask[0][0].shape[-1], 2)

        reader = create_reader(cfg['EvalReader'], 10)()
        for samples in reader:
            for sample in samples:
                im_shape = sample[0].shape
                self.assertEqual(im_shape[0], 3)
                self.assertEqual(im_shape[1] % 32, 0)
                self.assertEqual(im_shape[2] % 32, 0)

                im_info_shape = sample[1].shape
                self.assertEqual(im_info_shape[-1], 3)

                im_id_shape = sample[2].shape
                self.assertEqual(im_id_shape[-1], 1)


if __name__ == '__main__':
    enable_static_mode()
    unittest.main()
