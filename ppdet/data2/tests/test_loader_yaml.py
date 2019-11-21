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

from ppdet.utils.download import get_path
from ppdet.utils.download import DATASET_HOME
from ppdet.core.workspace import load_config, merge_config

from ppdet.data2.loader import create_reader

COCO_VAL_URL = 'http://images.cocodataset.org/zips/val2017.zip'
COCO_VAL_MD5SUM = '442b8da7639aecaf257c1dceb8ba8c80'
COCO_ANNO_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
COCO_ANNO_MD5SUM = 'f4bbac642086de4f52a3fdda2de5fa2c'

FORMAT = '[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class TestLoaderYAML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """ setup
        """
        root_path = os.path.join(DATASET_HOME, 'coco')
        image_path, _ = get_path(COCO_VAL_URL, root_path, COCO_VAL_MD5SUM)

        anno_path, _ = get_path(COCO_ANNO_URL, root_path, COCO_ANNO_MD5SUM)

        # json data
        cls.anno_path = os.path.join(anno_path, 'instances_val2017.json')
        cls.image_dir = image_path

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_loader_yaml(self):
        cfg_file = 'ppdet/data2/tests/test.yml'
        cfg = load_config(cfg_file)
        data_cfg = '[!COCODataSet {{image_dir: {0}, dataset_dir: null, ' \
            'anno_path: {1}, sample_num: 10}}]'.format(
                self.image_dir, self.anno_path)
        dataset_ins = yaml.load(data_cfg, Loader=yaml.Loader)
        update_train_cfg = {'TrainLoader': {'dataset': dataset_ins[0]}}
        update_test_cfg = {'EvalLoader': {'dataset': dataset_ins[0]}}
        merge_config(update_train_cfg)
        merge_config(update_test_cfg)

        #reader = Loader(**cfg['TrainLoader'])()
        reader = create_reader(cfg['TrainLoader'], 10)()
        for samples in reader:
            for sample in samples:
                shape = sample['image'].shape
                self.assertEqual(shape[0], 3)
                self.assertEqual(shape[1] % 32, 0)
                self.assertEqual(shape[2] % 32, 0)

        #reader = Loader(**cfg['EvalLoader'])()
        reader = create_reader(cfg['EvalLoader'], 10)()
        for samples in reader:
            for sample in samples:
                shape = sample['image'].shape
                self.assertEqual(shape[0], 3)
                self.assertEqual(shape[1] % 32, 0)
                self.assertEqual(shape[2] % 32, 0)


if __name__ == '__main__':
    unittest.main()
