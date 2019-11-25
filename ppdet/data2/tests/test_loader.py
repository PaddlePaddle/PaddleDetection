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

from ppdet.data2.coco import COCODataSet
from ppdet.data2.loader import Loader
from ppdet.utils.download import get_path
from ppdet.utils.download import DATASET_HOME

from ppdet.data2.operators import DecodeImage, ResizeImage, Permute
from ppdet.data2.batch_operators import PadBatch

COCO_VAL_URL = 'http://images.cocodataset.org/zips/val2017.zip'
COCO_VAL_MD5SUM = '442b8da7639aecaf257c1dceb8ba8c80'
COCO_ANNO_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
COCO_ANNO_MD5SUM = 'f4bbac642086de4f52a3fdda2de5fa2c'


class TestLoader(unittest.TestCase):
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

    def test_loader(self):
        coco_loader = COCODataSet(self.image_dir, self.anno_path, 10)
        sample_trans = [
            DecodeImage(to_rgb=True), ResizeImage(
                target_size=800, max_size=1333, interp=1), Permute(to_bgr=False)
        ]
        batch_trans = [PadBatch(pad_to_stride=32, use_padded_im_info=True), ]

        data_loader = Loader(
            coco_loader,
            sample_transforms=sample_trans,
            batch_transforms=batch_trans,
            batch_size=2,
            fields=[
                'image', 'im_info', 'im_id', 'gt_bbox', 'gt_class', 'is_crowd',
                'gt_mask'
            ],
            shuffle=True,
            drop_empty=True)

        for i in range(2):
            for samples in data_loader:
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
            data_loader.reset()

    def test_loader_multi_threads(self):
        coco_loader = COCODataSet(self.image_dir, self.anno_path, 10)
        sample_trans = [
            DecodeImage(to_rgb=True), ResizeImage(
                target_size=800, max_size=1333, interp=1), Permute(to_bgr=False)
        ]
        batch_trans = [PadBatch(pad_to_stride=32, use_padded_im_info=True), ]

        data_loader = Loader(
            coco_loader,
            sample_transforms=sample_trans,
            batch_transforms=batch_trans,
            batch_size=2,
            fields=[
                'image', 'im_info', 'im_id', 'gt_bbox', 'gt_class', 'is_crowd',
                'gt_mask'
            ],
            shuffle=True,
            drop_empty=True,
            worker_num=2,
            use_process=False,
            bufsize=8)()
        for i in range(2):
            for samples in data_loader:
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
            data_loader.reset()


if __name__ == '__main__':
    unittest.main()
