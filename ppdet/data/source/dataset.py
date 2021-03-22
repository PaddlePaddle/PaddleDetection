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

import os
import numpy as np
from collections import OrderedDict
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from paddle.io import Dataset
from ppdet.core.workspace import register, serializable
from ppdet.utils.download import get_dataset_path
import copy


@serializable
class DetDataset(Dataset):
    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 use_default_label=None,
                 **kwargs):
        super(DetDataset, self).__init__()
        self.dataset_dir = dataset_dir if dataset_dir is not None else ''
        self.anno_path = anno_path
        self.image_dir = image_dir if image_dir is not None else ''
        self.data_fields = data_fields
        self.sample_num = sample_num
        self.use_default_label = use_default_label
        self._epoch = 0

    def __len__(self, ):
        return len(self.roidbs)

    def __getitem__(self, idx):
        # data batch
        roidb = copy.deepcopy(self.roidbs[idx])
        if self.mixup_epoch == 0 or self._epoch < self.mixup_epoch:
            n = len(self.roidbs)
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.cutmix_epoch == 0 or self._epoch < self.cutmix_epoch:
            n = len(self.roidbs)
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.mosaic_epoch == 0 or self._epoch < self.mosaic_epoch:
            n = len(self.roidbs)
            roidb = [roidb, ] + [
                copy.deepcopy(self.roidbs[np.random.randint(n)])
                for _ in range(3)
            ]

        return self.transform(roidb)

    def set_kwargs(self, **kwargs):
        self.mixup_epoch = kwargs.get('mixup_epoch', -1)
        self.cutmix_epoch = kwargs.get('cutmix_epoch', -1)
        self.mosaic_epoch = kwargs.get('mosaic_epoch', -1)

    def set_transform(self, transform):
        self.transform = transform

    def set_epoch(self, epoch_id):
        self._epoch = epoch_id

    def parse_dataset(self, ):
        raise NotImplementedError(
            "Need to implement parse_dataset method of Dataset")

    def get_anno(self):
        if self.anno_path is None:
            return
        return os.path.join(self.dataset_dir, self.anno_path)


def _is_valid_file(f, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    return f.lower().endswith(extensions)


def _make_dataset(dir):
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        raise ('{} should be a dir'.format(dir))
    images = []
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if _is_valid_file(path):
                images.append(path)
    return images


@register
@serializable
class ImageFolder(DetDataset):
    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 use_default_label=None,
                 **kwargs):
        super(ImageFolder, self).__init__(
            dataset_dir,
            image_dir,
            anno_path,
            sample_num=sample_num,
            use_default_label=use_default_label)
        self._imid2path = {}
        self.roidbs = None
        self.sample_num = sample_num

    def parse_dataset(self, ):
        if not self.roidbs:
            self.roidbs = self._load_images()

    def _parse(self):
        image_dir = self.image_dir
        if not isinstance(image_dir, Sequence):
            image_dir = [image_dir]
        images = []
        for im_dir in image_dir:
            if os.path.isdir(im_dir):
                im_dir = os.path.join(self.dataset_dir, im_dir)
                images.extend(_make_dataset(im_dir))
            elif os.path.isfile(im_dir) and _is_valid_file(im_dir):
                images.append(im_dir)
        return images

    def _load_images(self):
        images = self._parse()
        ct = 0
        records = []
        for image in images:
            assert image != '' and os.path.isfile(image), \
                    "Image {} not found".format(image)
            if self.sample_num > 0 and ct >= self.sample_num:
                break
            rec = {'im_id': np.array([ct]), 'im_file': image}
            self._imid2path[ct] = image
            ct += 1
            records.append(rec)
        assert len(records) > 0, "No image file found"
        return records

    def get_imid2path(self):
        return self._imid2path

    def set_images(self, images):
        self.image_dir = images
        self.roidbs = self._load_images()
