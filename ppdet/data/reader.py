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

import os
import copy
import functools
import collections
import traceback
import numpy as np
import logging

from ppdet.core.workspace import register, serializable

from .parallel_map import ParallelMap
from .transform.batch_operators import Gt2YoloTarget

__all__ = ['Reader', 'create_reader']

logger = logging.getLogger(__name__)


class Compose(object):
    def __init__(self, transforms, ctx=None):
        self.transforms = transforms
        self.ctx = ctx

    def __call__(self, data):
        ctx = self.ctx if self.ctx else {}
        for f in self.transforms:
            try:
                data = f(data, ctx)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warn("fail to map op [{}] with error: {} and stack:\n{}".
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


def _has_empty(item):
    def empty(x):
        if isinstance(x, np.ndarray) and x.size == 0:
            return True
        elif isinstance(x, collections.Sequence) and len(x) == 0:
            return True
        else:
            return False

    if isinstance(item, collections.Sequence) and len(item) == 0:
        return True
    if item is None:
        return True
    if empty(item):
        return True
    return False


def _segm(samples):
    assert 'gt_poly' in samples
    segms = samples['gt_poly']
    if 'is_crowd' in samples:
        is_crowd = samples['is_crowd']
        if len(segms) != 0:
            assert len(segms) == is_crowd.shape[0]

    gt_masks = []
    valid = True
    for i in range(len(segms)):
        segm = segms[i]
        gt_segm = []
        if 'is_crowd' in samples and is_crowd[i]:
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


def batch_arrange(batch_samples, fields):
    def im_shape(samples, dim=3):
        # hard code
        assert 'h' in samples
        assert 'w' in samples
        if dim == 3:  # RCNN, ..
            return np.array((samples['h'], samples['w'], 1), dtype=np.float32)
        else:  # YOLOv3, ..
            return np.array((samples['h'], samples['w']), dtype=np.int32)

    arrange_batch = []
    for samples in batch_samples:
        one_ins = ()
        for i, field in enumerate(fields):
            if field == 'gt_mask':
                one_ins += (_segm(samples), )
            elif field == 'im_shape':
                one_ins += (im_shape(samples), )
            elif field == 'im_size':
                one_ins += (im_shape(samples, 2), )
            else:
                if field == 'is_difficult':
                    field = 'difficult'
                assert field in samples, '{} not in samples'.format(field)
                one_ins += (samples[field], )
        arrange_batch.append(one_ins)
    return arrange_batch


@register
@serializable
class Reader(object):
    """
    Args:
        dataset (DataSet): DataSet object
        sample_transforms (list of BaseOperator): a list of sample transforms
            operators.
        batch_transforms (list of BaseOperator): a list of batch transforms
            operators.
        batch_size (int): batch size.
        shuffle (bool): whether shuffle dataset or not. Default False.
        drop_last (bool): whether drop last batch or not. Default False.
        drop_empty (bool): whether drop sample when it's gt is empty or not.
            Default True.
        mixup_epoch (int): mixup epoc number. Default is -1, meaning
            not use mixup.
        cutmix_epoch (int): cutmix epoc number. Default is -1, meaning
            not use cutmix.
        class_aware_sampling (bool): whether use class-aware sampling or not.
            Default False.
        worker_num (int): number of working threads/processes.
            Default -1, meaning not use multi-threads/multi-processes.
        use_process (bool): whether use multi-processes or not.
            It only works when worker_num > 1. Default False.
        bufsize (int): buffer size for multi-threads/multi-processes,
            please note, one instance in buffer is one batch data.
        memsize (str): size of shared memory used in result queue when
            use_process is true. Default 3G.
        inputs_def (dict): network input definition use to get input fields,
            which is used to determine the order of returned data.
        devices_num (int): number of devices.
    """

    def __init__(self,
                 dataset=None,
                 sample_transforms=None,
                 batch_transforms=None,
                 batch_size=None,
                 shuffle=False,
                 drop_last=False,
                 drop_empty=True,
                 mixup_epoch=-1,
                 cutmix_epoch=-1,
                 class_aware_sampling=False,
                 worker_num=-1,
                 use_process=False,
                 use_fine_grained_loss=False,
                 num_classes=80,
                 bufsize=-1,
                 memsize='3G',
                 inputs_def=None,
                 devices_num=1):
        self._dataset = dataset
        self._roidbs = self._dataset.get_roidb()
        self._fields = copy.deepcopy(inputs_def[
            'fields']) if inputs_def else None

        # transform
        self._sample_transforms = Compose(sample_transforms,
                                          {'fields': self._fields})
        self._batch_transforms = None

        if use_fine_grained_loss:
            for bt in batch_transforms:
                if isinstance(bt, Gt2YoloTarget):
                    bt.num_classes = num_classes
        elif batch_transforms:
            batch_transforms = [
                bt for bt in batch_transforms
                if not isinstance(bt, Gt2YoloTarget)
            ]

        if batch_transforms:
            self._batch_transforms = Compose(batch_transforms,
                                             {'fields': self._fields})

        # data
        if inputs_def and inputs_def.get('multi_scale', False):
            from ppdet.modeling.architectures.input_helper import multiscale_def
            im_shape = inputs_def[
                'image_shape'] if 'image_shape' in inputs_def else [
                    3, None, None
                ]
            _, ms_fields = multiscale_def(im_shape, inputs_def['num_scales'],
                                          inputs_def['use_flip'])
            self._fields += ms_fields
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._drop_empty = drop_empty

        # sampling
        self._mixup_epoch = mixup_epoch
        self._cutmix_epoch = cutmix_epoch
        self._class_aware_sampling = class_aware_sampling

        self._load_img = False
        self._sample_num = len(self._roidbs)

        if self._class_aware_sampling:
            self.img_weights = _calc_img_weights(self._roidbs)
        self._indexes = None

        self._pos = -1
        self._epoch = -1

        self._curr_iter = 0

        # multi-process
        self._worker_num = worker_num
        self._parallel = None
        if self._worker_num > -1:
            task = functools.partial(self.worker, self._drop_empty)
            bufsize = devices_num * 2 if bufsize == -1 else bufsize
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
        if self._epoch < 0:
            self._epoch = 0
        else:
            self._epoch += 1

        self.indexes = [i for i in range(self.size())]
        if self._class_aware_sampling:
            self.indexes = np.random.choice(
                self._sample_num,
                self._sample_num,
                replace=True,
                p=self.img_weights)

        if self._shuffle:
            trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))
            np.random.seed(self._epoch + trainer_id)
            np.random.shuffle(self.indexes)

        if self._mixup_epoch > 0 and len(self.indexes) < 2:
            logger.debug("Disable mixup for dataset samples "
                         "less than 2 samples")
            self._mixup_epoch = -1
        if self._cutmix_epoch > 0 and len(self.indexes) < 2:
            logger.info("Disable cutmix for dataset samples "
                        "less than 2 samples")
            self._cutmix_epoch = -1

        self._pos = 0

    def __next__(self):
        return self.next()

    def next(self):
        if self._epoch < 0:
            self.reset()
        if self.drained():
            raise StopIteration
        batch = self._load_batch()
        self._curr_iter += 1
        if self._drop_last and len(batch) < self._batch_size:
            raise StopIteration
        if self._worker_num > -1:
            return batch
        else:
            return self.worker(self._drop_empty, batch)

    def _load_batch(self):
        batch = []
        bs = 0
        while bs != self._batch_size:
            if self._pos >= self.size():
                break
            pos = self.indexes[self._pos]
            sample = copy.deepcopy(self._roidbs[pos])
            sample["curr_iter"] = self._curr_iter
            self._pos += 1

            if self._drop_empty and self._fields and 'gt_mask' in self._fields:
                if _has_empty(_segm(sample)):
                    #logger.warn('gt_mask is empty or not valid in {}'.format(
                    #    sample['im_file']))
                    continue
            if self._drop_empty and self._fields and 'gt_bbox' in self._fields:
                if _has_empty(sample['gt_bbox']):
                    #logger.warn('gt_bbox {} is empty or not valid in {}, '
                    #   'drop this sample'.format(
                    #    sample['im_file'], sample['gt_bbox']))
                    continue

            if self._load_img:
                sample['image'] = self._load_image(sample['im_file'])

            if self._epoch < self._mixup_epoch:
                num = len(self.indexes)
                mix_idx = np.random.randint(1, num)
                mix_idx = self.indexes[(mix_idx + self._pos - 1) % num]
                sample['mixup'] = copy.deepcopy(self._roidbs[mix_idx])
                sample['mixup']["curr_iter"] = self._curr_iter
                if self._load_img:
                    sample['mixup']['image'] = self._load_image(sample['mixup'][
                        'im_file'])
            if self._epoch < self._cutmix_epoch:
                num = len(self.indexes)
                mix_idx = np.random.randint(1, num)
                sample['cutmix'] = copy.deepcopy(self._roidbs[mix_idx])
                sample['cutmix']["curr_iter"] = self._curr_iter
                if self._load_img:
                    sample['cutmix']['image'] = self._load_image(sample[
                        'cutmix']['im_file'])

            batch.append(sample)
            bs += 1
        return batch

    def worker(self, drop_empty=True, batch_samples=None):
        """
        sample transform and batch transform.
        """
        batch = []
        for sample in batch_samples:
            sample = self._sample_transforms(sample)
            if drop_empty and 'gt_bbox' in sample:
                if _has_empty(sample['gt_bbox']):
                    #logger.warn('gt_bbox {} is empty or not valid in {}, '
                    #   'drop this sample'.format(
                    #    sample['im_file'], sample['gt_bbox']))
                    continue
            batch.append(sample)
        if len(batch) > 0 and self._batch_transforms:
            batch = self._batch_transforms(batch)
        if len(batch) > 0 and self._fields:
            batch = batch_arrange(batch, self._fields)
        return batch

    def _load_image(self, filename):
        with open(filename, 'rb') as f:
            return f.read()

    def size(self):
        """ implementation of Dataset.size
        """
        return self._sample_num

    def drained(self):
        """ implementation of Dataset.drained
        """
        assert self._epoch >= 0, 'The first epoch has not begin!'
        return self._pos >= self.size()

    def stop(self):
        if self._parallel:
            self._parallel.stop()


def create_reader(cfg, max_iter=0, global_cfg=None, devices_num=1):
    """
    Return iterable data reader.

    Args:
        max_iter (int): number of iterations.
    """
    if not isinstance(cfg, dict):
        raise TypeError("The config should be a dict when creating reader.")

    # synchornize use_fine_grained_loss/num_classes from global_cfg to reader cfg
    if global_cfg:
        cfg['use_fine_grained_loss'] = getattr(global_cfg,
                                               'use_fine_grained_loss', False)
        cfg['num_classes'] = getattr(global_cfg, 'num_classes', 80)
    cfg['devices_num'] = devices_num
    reader = Reader(**cfg)()

    def _reader():
        n = 0
        while True:
            for _batch in reader:
                if len(_batch) > 0:
                    yield _batch
                    n += 1
                if max_iter > 0 and n == max_iter:
                    return
            reader.reset()
            if max_iter <= 0:
                return

    return _reader
