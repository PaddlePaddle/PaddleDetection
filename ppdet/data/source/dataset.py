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

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from ppdet.core.workspace import register, serializable
from ppdet.utils.download import get_dataset_path


@serializable
class DataSet(object):
    """
    Dataset, e.g., coco, pascal voc

    Args:
        annotation (str): annotation file path
        image_dir (str): directory where image files are stored
        shuffle (bool): shuffle samples
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 with_background=True,
                 use_default_label=False,
                 **kwargs):
        super(DataSet, self).__init__()
        self.anno_path = anno_path
        self.image_dir = image_dir if image_dir is not None else ''
        self.dataset_dir = dataset_dir if dataset_dir is not None else ''
        self.sample_num = sample_num
        self.with_background = with_background
        self.use_default_label = use_default_label

        self.cname2cid = None
        self._imid2path = None

    def load_roidb_and_cname2cid(self):
        """load dataset"""
        raise NotImplementedError('%s.load_roidb_and_cname2cid not available' %
                                  (self.__class__.__name__))

    def get_roidb(self):
        if not self.roidbs:
            data_dir = get_dataset_path(self.dataset_dir, self.anno_path,
                                        self.image_dir)
            if data_dir:
                self.dataset_dir = data_dir
            self.load_roidb_and_cname2cid()

        return self.roidbs

    def get_cname2cid(self):
        if not self.cname2cid:
            self.load_roidb_and_cname2cid()
        return self.cname2cid

    def get_anno(self):
        if self.anno_path is None:
            return
        return os.path.join(self.dataset_dir, self.anno_path)

    def get_imid2path(self):
        return self._imid2path


def _is_valid_file(f, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    return f.lower().endswith(extensions)


def _make_dataset(data_dir):
    data_dir = os.path.expanduser(data_dir)
    if not os.path.isdir(data_dir):
        raise ('{} should be a dir'.format(data_dir))
    images = []
    for root, _, fnames in sorted(os.walk(data_dir, followlinks=True)):
        for fname in sorted(fnames):
            file_path = os.path.join(root, fname)
            if _is_valid_file(file_path):
                images.append(file_path)
    return images


@register
@serializable
class ImageFolder(DataSet):
    """
    Args:
        dataset_dir (str): root directory for dataset.
        image_dir(list|str): list of image folders or list of image files
        anno_path (str): annotation file path.
        samples (int): number of samples to load, -1 means all
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 with_background=True,
                 use_default_label=False,
                 **kwargs):
        super(ImageFolder, self).__init__(dataset_dir, image_dir, anno_path,
                                          sample_num, with_background,
                                          use_default_label)
        self.roidbs = None
        self._imid2path = {}

    def get_roidb(self):
        if not self.roidbs:
            self.roidbs = self._load_images()
        return self.roidbs

    def set_images(self, images):
        self.image_dir = images
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
