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

from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler

from ppdet.core.workspace import register, serializable, create
from . import transform

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
                f = op_cls(**v)
                if hasattr(f, 'num_classes'):
                    f.num_classes = num_classes

                self.transforms_cls.append(f)

    def __call__(self, data):
        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warn("fail to map op [{}] with error: {} and stack:\n{}".
                            format(f, e, str(stack_info)))
                raise e

        return data


class BatchCompose(Compose):
    def __init__(self, transforms, num_classes=80):
        super(BatchCompose, self).__init__(transforms, num_classes)
        self.output_fields = mp.Manager().list([])
        self.lock = mp.Lock()

    def __call__(self, data):
        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warn("fail to map op [{}] with error: {} and stack:\n{}".
                            format(f, e, str(stack_info)))
                raise e

        # accessing ListProxy in main process (no worker subprocess)
        # may incur errors in some enviroments, ListProxy back to
        # list if no worker process start, while this `__call__`
        # will be called in main process
        global MAIN_PID
        if os.getpid() == MAIN_PID and \
            isinstance(self.output_fields, mp.managers.ListProxy):
            self.output_fields = []

        # parse output fields by first sample
        # **this shoule be fixed if paddle.io.DataLoader support**
        # For paddle.io.DataLoader not support dict currently,
        # we need to parse the key from the first sample,
        # BatchCompose.__call__ will be called in each worker
        # process, so lock is need here.
        if len(self.output_fields) == 0:
            self.lock.acquire()
            if len(self.output_fields) == 0:
                for k, v in data[0].items():
                    # FIXME(dkp): for more elegent coding
                    if k not in ['flipped', 'h', 'w']:
                        self.output_fields.append(k)
            self.lock.release()

        data = [[data[i][k] for k in self.output_fields]
                for i in range(len(data))]
        data = list(zip(*data))

        batch_data = [np.stack(d, axis=0) for d in data]
        return batch_data


class BaseDataLoader(object):
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
                 return_list=False,
                 use_prefetch=True):
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

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_sampler=self._batch_sampler,
            collate_fn=self._batch_transforms,
            num_workers=worker_num,
            return_list=return_list,
            use_buffer_reader=use_prefetch,
            use_shared_memory=False)
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
            data = next(self.loader)
            return {
                k: v
                for k, v in zip(self._batch_transforms.output_fields, data)
            }
        except StopIteration:
            self.loader = iter(self.dataloader)
            six.reraise(*sys.exc_info())

    def next(self):
        # python2 compatibility
        return self.__next__()


@register
class TrainReader(BaseDataLoader):
    __shared__ = ['num_classes']

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
    __shared__ = ['num_classes']

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
    __shared__ = ['num_classes']

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
