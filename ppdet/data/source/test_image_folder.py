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

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from ppdet.data.source.dataset import ImageFolder


class TestImageFolder(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root / 'sample.jpg'
        Image.new('RGB', (8, 8)).save(self.image)

    def test_inference_ignores_missing_annotations(self):
        dataset = ImageFolder(dataset_dir=str(self.root),
                              anno_path='missing.json')
        dataset.set_images([str(self.image)])
        self.assertEqual(dataset.roidbs[0]['im_id'].tolist(), [0])
        self.assertEqual(dataset.get_imid2path(), {0: str(self.image)})

    def test_parse_dataset_ignores_invalid_annotations(self):
        annotation = self.root / 'invalid.json'
        annotation.write_text('not JSON')
        dataset = ImageFolder(image_dir=[str(self.image)],
                              anno_path=str(annotation))
        dataset.parse_dataset()
        self.assertEqual(len(dataset.roidbs), 1)

    def test_evaluation_preserves_annotation_image_id(self):
        annotation = self.root / 'annotations.json'
        annotation.write_text(
            json.dumps({
                'images': [{
                    'id': 42,
                    'file_name': 'sample.jpg'
                }],
                'annotations': [],
                'categories': []
            }))
        dataset = ImageFolder(anno_path=str(annotation))
        dataset.set_images([str(self.image)], do_eval=True)
        self.assertEqual(dataset.roidbs[0]['im_id'].tolist(), [42])
        self.assertEqual(dataset.get_imid2path(), {42: str(self.image)})

    def test_evaluation_requires_annotations(self):
        dataset = ImageFolder(dataset_dir=str(self.root),
                              anno_path='missing.json')
        with self.assertRaises(FileNotFoundError):
            dataset.set_images([str(self.image)], do_eval=True)


if __name__ == '__main__':
    unittest.main()
