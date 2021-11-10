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
import six
import sys

from .compose import Compose, BatchCompose

if sys.version_info >= (3, 0):
    pass
else:
    pass
import numpy as np

from paddle.io import DataLoader, DistributedBatchSampler

from ppdet.core.workspace import register
from .shm_utils import _get_shared_memory_size_in_M

from ppdet.utils.logger import setup_logger
logger = setup_logger('reader')

MAIN_PID = os.getpid()


class BaseDataLoader(object):
    """
    Base DataLoader implementation for detection models

    Args:
        sample_transforms (list): a list of transforms to perform
                                  on each sample
        batch_transforms (list): a list of transforms to perform
                                 on batch
        batch_size (int): batch size for batch collating, default 1.
        shuffle (bool): whether to shuffle samples
        drop_last (bool): whether to drop the last incomplete,
                          default False
        num_classes (int): class number of dataset, default 80
        collate_batch (bool): whether to collate batch in dataloader.
            If set to True, the samples will collate into batch according
            to the batch size. Otherwise, the ground-truth will not collate,
            which is used when the number of ground-truch is different in 
            samples.
        use_shared_memory (bool): whether to use shared memory to
                accelerate data loading, enable this only if you
                are sure that the shared memory size of your OS
                is larger than memory cost of input datas of model.
                Note that shared memory will be automatically
                disabled if the shared memory of OS is less than
                1G, which is not enough for detection models.
                Default False.
    """

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=80,
                 collate_batch=True,
                 use_shared_memory=False,
                 **kwargs):
        # sample transform
        self._sample_transforms = Compose(
            sample_transforms, num_classes=num_classes)

        # batch transfrom
        self._batch_transforms = BatchCompose(batch_transforms, num_classes,
                                              collate_batch)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.use_shared_memory = use_shared_memory
        self.kwargs = kwargs

    def __call__(self,
                 dataset,
                 worker_num,
                 batch_sampler=None,
                 return_list=False):
        self.dataset = dataset
        self.dataset.check_or_download_dataset()
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

        # DataLoader do not start sub-process in Windows and Mac
        # system, do not need to use shared memory
        use_shared_memory = self.use_shared_memory and \
                            sys.platform not in ['win32', 'darwin']
        # check whether shared memory size is bigger than 1G(1024M)
        if use_shared_memory:
            shm_size = _get_shared_memory_size_in_M()
            if shm_size is not None and shm_size < 1024.:
                logger.warning("Shared memory size is less than 1G, "
                               "disable shared_memory in DataLoader")
                use_shared_memory = False

        if 'DALI' in self.dataset.__class__.__name__:
            from .dali import COCOPipeline, DALICOCOIterator

            for t in self._sample_transforms.transforms_cls:
                if hasattr(t, 'use_dali'):
                    setattr(t, 'use_dali', True)

            # self._batch_transforms.collate_batch = False
            self._batch_transforms.use_dali = True

            p = COCOPipeline(self.dataset, batch_size=self.batch_size,
                             transforms=self._sample_transforms,
                             batch_transforms=self._batch_transforms)
            p.build()

            self.dataloader = DALICOCOIterator(
                [p],
                ['data', 'size']
            )
            self.loader = self.dataloader
        else:
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
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=True,
                 drop_last=True,
                 num_classes=80,
                 collate_batch=True,
                 **kwargs):
        super(TrainReader, self).__init__(sample_transforms, batch_transforms,
                                          batch_size, shuffle, drop_last,
                                          num_classes, collate_batch, **kwargs)


@register
class EvalReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=True,
                 num_classes=80,
                 **kwargs):
        super(EvalReader, self).__init__(sample_transforms, batch_transforms,
                                         batch_size, shuffle, drop_last,
                                         num_classes, **kwargs)


@register
class TestReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=80,
                 **kwargs):
        super(TestReader, self).__init__(sample_transforms, batch_transforms,
                                         batch_size, shuffle, drop_last,
                                         num_classes, **kwargs)


@register
class EvalMOTReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=1,
                 **kwargs):
        super(EvalMOTReader, self).__init__(sample_transforms, batch_transforms,
                                            batch_size, shuffle, drop_last,
                                            num_classes, **kwargs)


@register
class TestMOTReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=1,
                 **kwargs):
        super(TestMOTReader, self).__init__(sample_transforms, batch_transforms,
                                            batch_size, shuffle, drop_last,
                                            num_classes, **kwargs)
