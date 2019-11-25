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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import collections
import traceback
import numpy as np
import logging

from ppdet.core.workspace import register, serializable

from .parallel import ParallelMap
from .dataset import DataSet

__all__ = ['Loader', 'create_reader']

logger = logging.getLogger(__name__)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        ctx = {}
        for f in self.transforms:
            try:
                data = f(data, ctx)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.info("fail to map op [{}] with error: {} and stack:\n{}".
                            format(f, e, str(stack_info)))
                raise e
        return data


def _calc_img_weights(roidbs):
    """ calculate the probabilities of each sample
    """
    imgs_cls = []
    num_per_cls = {}
    img_weights = []
    for i, roidb in enumerate(roidbs):
        img_cls = set([k for cls in roidbs[i]['gt_class'] for k in cls])
        imgs_cls.append(img_cls)
        for c in img_cls:
            if c not in num_per_cls:
                num_per_cls[c] = 1
            else:
                num_per_cls[c] += 1

    for i in range(len(roidbs)):
        weights = 0
        for c in imgs_cls[i]:
            weights += 1 / num_per_cls[c]
        img_weights.append(weights)
    # probabilities sum to 1
    img_weights = img_weights / np.sum(img_weights)
    return img_weights


def has_empty(items):
    def empty(x):
        if isinstance(x, np.ndarray) and x.size == 0:
            return True
        elif isinstance(x, collections.Sequence) and len(x) == 0:
            return True
        else:
            return False

    if any(x is None for x in items):
        return True
    if any(empty(x) for x in items):
        return True
    return False


def batch_arrange(batch_samples, fields):
    def segm(samples):
        assert 'gt_poly' in samples
        assert 'is_crowd' in samples
        segms = samples['gt_poly']
        is_crowd = samples['is_crowd']
        if len(segms) != 0:
            assert len(segms) == is_crowd.shape[0]

        gt_masks = []
        valid = True
        for i in range(len(segms)):
            segm, iscrowd = segms[i], is_crowd[i]
            gt_segm = []
            if iscrowd:
                gt_segm.append([[0, 0]])
            else:
                for poly in segm:
                    if len(poly) == 0:
                        valid = False
                        break
                    gt_segm.append(np.array(poly).reshape(-1, 2))
            if (not valid) or len(gt_segm) == 0:
                break
            gt_masks.append(gt_segm)
        return gt_masks

    arrange_batch = []
    for samples in batch_samples:
        one_ins = ()
        for i, field in enumerate(fields):
            if field == 'gt_mask':
                one_ins += (segm(samples), )
            else:
                assert field in samples, '{} not in samples'.format(field)
                one_ins += (samples[field], )
        arrange_batch.append(one_ins)
    return arrange_batch


@register
@serializable
class Loader(object):
    def __init__(self,
                 dataset=None,
                 sample_transforms=None,
                 batch_transforms=None,
                 batch_size=None,
                 fields=None,
                 shuffle=False,
                 drop_last=False,
                 drop_empty=True,
                 mixup_epoch=-1,
                 class_aware_sampling=False,
                 worker_num=-1,
                 use_process=False,
                 bufsize=100,
                 memsize='3G',
                 inputs_def=None,
                 **kwargs):
        self._dataset = dataset
        self._roidbs = self._dataset.get_roidb()
        # transform
        self._sample_transforms = Compose(sample_transforms)
        self._batch_transforms = None
        if batch_transforms:
            self._batch_transforms = Compose(batch_transforms)

        # data
        self._fields = inputs_def['fields'] if inputs_def else None
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._drop_empty = drop_empty

        # sampling
        self._mixup_epoch = mixup_epoch
        self._class_aware_sampling = class_aware_sampling

        self._load_img = False
        self._sample_num = len(self._roidbs)

        if self._class_aware_sampling:
            self.img_weights = _calc_img_weights(self._roidbs)
        self._indexes = None

        self._pos = -1
        self._epoch = -1
        self._drained = False

        # multi-process
        self._worker_num = worker_num
        if self._worker_num > -1:
            task = functools.partial(self.worker, self._drop_empty)
            self._parallel = ParallelMap(self, task, worker_num, bufsize,
                                         use_process, memsize)

    def __call__(self):
        if self._worker_num > -1:
            return self._parallel
        else:
            return self

    def __iter__(self):
        return self

    def reset(self):
        """implementation of Dataset.reset
        """
        self.indexes = [i for i in range(self.size())]
        if self._class_aware_sampling:
            self.indexes = np.random.choice(
                self._sample_num,
                self._sample_num,
                replace=False,
                p=self.img_weights)

        if self._shuffle:
            np.random.shuffle(self.indexes)

        if self._drop_last:
            self.indexes = self.indexes[0:self.size() // self._batch_size *
                                        self._batch_size]

        if self._epoch < 0:
            self._epoch = 0
        else:
            self._epoch += 1

        self._pos = 0
        self._drained = False

    def __next__(self):
        return self.next()

    def next(self):
        if self._epoch < 0:
            self.reset()
        if self.drained():
            raise StopIteration('There is no more data in %s.' % (str(self)))
        batch = self._load_batch()
        if self._worker_num > -1:
            return batch
        else:
            return self.worker(self._drop_empty, batch)

    def _load_batch(self):
        batch = []
        for i in range(self._batch_size):
            if self._pos >= self.size():
                break
            pos = self.indexes[self._pos]
            sample = copy.deepcopy(self._roidbs[pos])

            if self._load_img:
                sample['image'] = self._load_image(sample['im_file'])

            if self._epoch < self._mixup_epoch:
                num = len(self.indexes)
                mix_idx = np.random.randint(1, num)
                mix_idx = (mix_idx + pos) % num
                sample['mixup'] = copy.deepcopy(self._roidbs[mix_idx])
                if self._load_img:
                    sample['mixup']['image'] = self._load_image(sample['mixup'][
                        'im_file'])

            batch.append(sample)
            self._pos += 1
        return batch

    def worker(self, drop_empty=True, batch_samples=None):
        """
        sample transform and data transform.
        """
        batch = []
        for sample in batch_samples:
            sample = self._sample_transforms(sample)
            if self._drop_empty and has_empty(sample):
                continue
            batch.append(sample)
        if len(batch) > 0 and self._batch_transforms:
            batch = self._batch_transforms(batch)
        if self._fields:
            batch = batch_arrange(batch, self._fields)
        return batch

    def size(self):
        """ implementation of Dataset.size
        """
        return self._sample_num

    def drained(self):
        """ implementation of Dataset.drained
        """
        assert self._epoch >= 0, 'The first epoch has not begin!'
        return self._pos >= self.size()


def create_reader(cfg, max_iter=0):
    """
    Return iterable data reader.

    Args:
        max_iter (int): number of iterations.
    """
    if not isinstance(cfg, dict):
        raise TypeError("The config should be a dict when creating reader.")

    def _reader():
        reader = Loader(**cfg)()
        n = 0
        while True:
            for _batch in reader:
                yield _batch
                n += 1
                if max_iter > 0 and n == max_iter:
                    return
            reader.reset()
            if max_iter <= 0:
                return

    return _reader
