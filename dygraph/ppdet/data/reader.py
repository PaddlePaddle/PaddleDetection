# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import copy
import traceback
import six
import sys
import multiprocessing as mp
if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue
import numpy as np

from paddle.io import DataLoader, DistributedBatchSampler
from paddle.fluid.dataloader.collate import default_collate_fn

from ppdet.core.workspace import register, serializable, create
from . import transform
from .shm_utils import _get_shared_memory_size_in_M

from ppdet.utils.logger import setup_logger
logger = setup_logger('reader')

MAIN_PID = os.getpid()


class Compose(object):
    def __init__(self, transforms, num_classes=80):
        self.transforms = transforms
        self.transforms_cls = []
        for t in self.transforms:
            for k, v in t.items():
                op_cls = getattr(transform, k)
                self.transforms_cls.append(op_cls(**v))
                if hasattr(op_cls, 'num_classes'):
                    op_cls.num_classes = num_classes

    def __call__(self, data):
        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warn("fail to map sample transform [{}] "
                            "with error: {} and stack:\n{}".format(
                                f, e, str(stack_info)))
                raise e

        return data


class BatchCompose(Compose):
    def __init__(self, transforms, num_classes=80):
        super(BatchCompose, self).__init__(transforms, num_classes)

    def __call__(self, data):
        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warn("fail to map batch transform [{}] "
                            "with error: {} and stack:\n{}".format(
                                f, e, str(stack_info)))
                raise e

        # remove keys which is not needed by model
        extra_key = ['h', 'w', 'flipped']
        for k in extra_key:
            for sample in data:
                if k in sample:
                    sample.pop(k)

        # batch data, if user-define batch function needed
        # use user-defined here
        data = default_collate_fn(data)

        return data


class BaseDataLoader(object):
    __share__ = ['num_classes']

    def __init__(self,
                 inputs_def=None,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 drop_empty=True,
                 num_classes=80,
                 **kwargs):
        # sample transform
        self._sample_transforms = Compose(
            sample_transforms, num_classes=num_classes)

        # batch transfrom 
        self._batch_transforms = BatchCompose(batch_transforms, num_classes)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.kwargs = kwargs

    def __call__(self,
                 dataset,
                 worker_num,
                 batch_sampler=None,
                 return_list=False):
        self.dataset = dataset
        self.dataset.parse_dataset()
        # get data
        self.dataset.set_transform(self._sample_transforms)
        # set kwargs
        self.dataset.set_kwargs(**self.kwargs)
        # batch sampler
        if batch_sampler is None:
            self._batch_sampler = DistributedBatchSampler(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last)
        else:
            self._batch_sampler = batch_sampler

        use_shared_memory = True
        # check whether shared memory size is bigger than 1G(1024M)
        shm_size = _get_shared_memory_size_in_M()
        if shm_size is not None and shm_size < 1024.:
            logger.warn("Shared memory size is less than 1G, "
                        "disable shared_memory in DataLoader")
            use_shared_memory = False

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_sampler=self._batch_sampler,
            collate_fn=self._batch_transforms,
            num_workers=worker_num,
            return_list=return_list,
            use_shared_memory=use_shared_memory)
        self.loader = iter(self.dataloader)

        return self

    def __len__(self):
        return len(self._batch_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        # pack {filed_name: field_data} here
        # looking forward to support dictionary
        # data structure in paddle.io.DataLoader
        try:
            return next(self.loader)
        except StopIteration:
            self.loader = iter(self.dataloader)
            six.reraise(*sys.exc_info())

    def next(self):
        # python2 compatibility
        return self.__next__()


@register
class TrainReader(BaseDataLoader):
    def __init__(self,
                 inputs_def=None,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=True,
                 drop_last=True,
                 drop_empty=True,
                 num_classes=80,
                 **kwargs):
        super(TrainReader, self).__init__(
            inputs_def, sample_transforms, batch_transforms, batch_size,
            shuffle, drop_last, drop_empty, num_classes, **kwargs)


@register
class EvalReader(BaseDataLoader):
    def __init__(self,
                 inputs_def=None,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=True,
                 drop_empty=True,
                 num_classes=80,
                 **kwargs):
        super(EvalReader, self).__init__(
            inputs_def, sample_transforms, batch_transforms, batch_size,
            shuffle, drop_last, drop_empty, num_classes, **kwargs)


@register
class TestReader(BaseDataLoader):
    def __init__(self,
                 inputs_def=None,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 drop_empty=True,
                 num_classes=80,
                 **kwargs):
        super(TestReader, self).__init__(
            inputs_def, sample_transforms, batch_transforms, batch_size,
            shuffle, drop_last, drop_empty, num_classes, **kwargs)
