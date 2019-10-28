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
import collections
import pickle as pkl
import numpy as np
from .roidb_source import RoiDbSource


class ClassAwareSamplingRoiDbSource(RoiDbSource):
    """ interface to load class aware sampling roidb data from files
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
        super(ClassAwareSamplingRoiDbSource, self).__init__(
            anno_file=anno_file,
            image_dir=image_dir,
            samples=samples,
            is_shuffle=is_shuffle,
            load_img=load_img,
            cname2cid=cname2cid,
            use_default_label=use_default_label,
            mixup_epoch=mixup_epoch,
            with_background=with_background)
        self._img_weights = None

    def __str__(self):
        return 'ClassAwareSamplingRoidbSource(fname:%s,epoch:%d,size:%d)' \
            % (self._fname, self._epoch, self.size())

    def next(self):
        """ load next sample
        """
        if self._epoch < 0:
            self.reset()

        _pos = np.random.choice(
            self._samples, 1, replace=False, p=self._img_weights)[0]
        sample = copy.deepcopy(self._roidb[_pos])

        if self._load_img:
            sample['image'] = self._load_image(sample['im_file'])
        else:
            sample['im_file'] = os.path.join(self._image_dir, sample['im_file'])

        return sample

    def _calc_img_weights(self):
        """ calculate the probabilities of each sample
        """
        imgs_cls = []
        num_per_cls = {}
        img_weights = []
        for i, roidb in enumerate(self._roidb):
            img_cls = set(
                [k for cls in self._roidb[i]['gt_class'] for k in cls])
            imgs_cls.append(img_cls)
            for c in img_cls:
                if c not in num_per_cls:
                    num_per_cls[c] = 1
                else:
                    num_per_cls[c] += 1

        for i in range(len(self._roidb)):
            weights = 0
            for c in imgs_cls[i]:
                weights += 1 / num_per_cls[c]
            img_weights.append(weights)
        # Probabilities sum to 1
        img_weights = img_weights / np.sum(img_weights)
        return img_weights

    def reset(self):
        """ implementation of Dataset.reset
        """
        if self._roidb is None:
            self._roidb = self._load()

        if self._img_weights is None:
            self._img_weights = self._calc_img_weights()

        self._samples = len(self._roidb)

        if self._epoch < 0:
            self._epoch = 0
