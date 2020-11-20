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


@serializable
class DetDataset(Dataset):
    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 use_default_label=None,
                 **kwargs):
        super(DetDataset, self).__init__()
        self.dataset_dir = dataset_dir if dataset_dir is not None else ''
        self.anno_path = anno_path
        self.image_dir = image_dir if image_dir is not None else ''
        self.sample_num = sample_num
        self.use_default_label = use_default_label
        self.epoch = 0
        # -1 for not use, 0 for always use, otherwise for specified epoch
        self.mixup_epoch = kwargs.get('mixup_epoch', -1)
        self.cutmix_epoch = kwargs.get('cutmix_epoch', -1)
        self.mosaic_epoch = kwargs.get('mosaic_epoch', -1)

    def __len__(self, ):
        return len(self.roidbs)

    def __getitem__(self, idx):
        # data batch
        roidb = self.roidbs[idx]
        if self.epoch < self.mixup_epoch:
            n = len(self.roidbs)
            roidb = [roidb, self.roidbs[np.random.randint(n)]]
        elif self.epoch < self.cutmix_epoch:
            n = len(self.roidbs)
            roidb = [roidb, self.roidbs[np.random.randint(n)]]
        elif self.epoch < self.mosaic_epoch:
            n = len(self.roidbs)
            roidb = [roidb,
                     ] + [self.roidbs[np.random.randint(n)] for _ in range(3)]

        # data augment
        roidb = self.transform(roidb)
        # data item 
        out = OrderedDict()
        for k in self.fields:
            out[k] = roidb[k]
        return out.values()

    def set_out(self, sample_transform, fields):
        self.transform = sample_transform
        self.fields = fields

    def parse_dataset(self, with_background=True):
        raise NotImplemented(
            "Need to implement parse_dataset method of Dataset")

    def get_anno(self):
        if self.anno_path is None:
            return
        return os.path.join(self.dataset_dir, self.anno_path)


@register
@serializable
class ImageFolder(DetDataset):
    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 **kwargs):
        super(ImageFolder, self).__init__(dataset_dir, image_dir, anno_path,
                                          sample_num)

    def parse_dataset(self):
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
        self.roidbs = images
