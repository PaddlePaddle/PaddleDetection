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

#function:
#    interface to load data from local files and parse it for samples, 
#    eg: roidb data in pickled files

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random

import copy
import pickle as pkl
from ..dataset import Dataset


class RoiDbSource(Dataset):
    """ interface to load roidb data from files
    """

    def __init__(self,
                 anno_file,
                 image_dir=None,
                 samples=-1,
                 is_shuffle=True,
                 load_img=False,
                 cname2cid=None,
                 use_default_label=None,
                 mixup_epoch=-1,
                 with_background=True):
        """ Init

        Args:
            fname (str): label file path
            image_dir (str): root dir for images
            samples (int): samples to load, -1 means all
            is_shuffle (bool): whether to shuffle samples
            load_img (bool): whether load data in this class
            cname2cid (dict): the label name to id dictionary
            use_default_label (bool):whether use the default mapping of label to id
            mixup_epoch (int): parse mixup in first n epoch
            with_background (bool): whether load background 
                                    as a class
        """
        super(RoiDbSource, self).__init__()
        self._epoch = -1
        assert os.path.isfile(anno_file) or os.path.isdir(
            anno_file), 'invalid file[%s] for RoiDbSource' % (anno_file)
        self._fname = anno_file
        self._image_dir = image_dir
        if image_dir is not None:
            assert os.path.isdir(image_dir), 'invalid image directory[%s]' % (
                image_dir)
        self._roidb = None
        self._pos = -1
        self._drained = False
        self._samples = samples
        self._is_shuffle = is_shuffle
        self._load_img = load_img
        self.use_default_label = use_default_label
        self._mixup_epoch = mixup_epoch
        self._with_background = with_background
        self.cname2cid = cname2cid
        self._imid2path = None

    def __str__(self):
        return 'RoiDbSource(fname:%s,epoch:%d,size:%d,pos:%d)' \
            % (self._fname, self._epoch, self.size(), self._pos)

    def next(self):
        """ load next sample
        """
        if self._epoch < 0:
            self.reset()
        if self._pos >= self._samples:
            self._drained = True
            raise StopIteration('%s no more data' % (str(self)))
        sample = copy.deepcopy(self._roidb[self._pos])
        if self._load_img:
            sample['image'] = self._load_image(sample['im_file'])
        else:
            sample['im_file'] = os.path.join(self._image_dir, sample['im_file'])

        if self._epoch < self._mixup_epoch:
            mix_idx = random.randint(1, self._samples - 1)
            mix_pos = (mix_idx + self._pos) % self._samples
            sample['mixup'] = copy.deepcopy(self._roidb[mix_pos])
            if self._load_img:
                sample['mixup']['image'] = \
                        self._load_image(sample['mixup']['im_file'])
            else:
                sample['mixup']['im_file'] = \
                        os.path.join(self._image_dir, sample['mixup']['im_file'])
        self._pos += 1
        return sample

    def _load(self):
        """ load data from file
        """
        from . import loader
        records, cname2cid = loader.load(self._fname, self._samples,
                                         self._with_background, True,
                                         self.use_default_label, self.cname2cid)
        self.cname2cid = cname2cid
        return records

    def _load_image(self, where):
        fn = os.path.join(self._image_dir, where)
        with open(fn, 'rb') as f:
            return f.read()

    def reset(self):
        """ implementation of Dataset.reset
        """
        if self._roidb is None:
            self._roidb = self._load()

        self._samples = len(self._roidb)
        if self._is_shuffle:
            random.shuffle(self._roidb)

        if self._epoch < 0:
            self._epoch = 0
        else:
            self._epoch += 1

        self._pos = 0
        self._drained = False

    def size(self):
        """ implementation of Dataset.size
        """
        return len(self._roidb)

    def drained(self):
        """ implementation of Dataset.drained
        """
        assert self._epoch >= 0, 'The first epoch has not begin!'
        return self._pos >= self.size()

    def epoch_id(self):
        """ return epoch id for latest sample
        """
        return self._epoch

    def get_imid2path(self):
        """return image id to image path map"""
        if self._imid2path is None:
            self._imid2path = {}
            for record in self._roidb:
                im_id = record['im_id']
                im_id = im_id if isinstance(im_id, int) else im_id[0]
                im_path = os.path.join(self._image_dir, record['im_file'])
                self._imid2path[im_id] = im_path
        return self._imid2path
