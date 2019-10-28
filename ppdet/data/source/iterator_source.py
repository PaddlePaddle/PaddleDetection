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
import numpy as np
import copy

import logging
logger = logging.getLogger(__name__)

from ..dataset import Dataset


class IteratorSource(Dataset):
    """
    Load data samples from iterator in stream mode

    Args:
        iter_maker (callable): callable function to generate a iter
        samples (int): number of samples to load, -1 means all
    """

    def __init__(self,
                 iter_maker,
                 samples=-1,
                 **kwargs):
        super(IteratorSource, self).__init__()
        self._epoch = -1

        self._iter_maker = iter_maker
        self._data_iter = None
        self._pos = -1
        self._drained = False
        self._samples = samples
        self._sample_num = -1

    def next(self):
        if self._epoch < 0:
            self.reset()

        if self._data_iter is not None:
            try:
                sample = next(self._data_iter)
                self._pos += 1
                ret = sample
            except StopIteration as e:
                if self._sample_num <= 0:
                    self._sample_num = self._pos
                elif self._sample_num != self._pos:
                    logger.info('num of loaded samples is different '
                        'with previouse setting[prev:%d,now:%d]' % (self._sample_num, self._pos))
                    self._sample_num = self._pos

                self._data_iter = None
                self._drained = True
                raise e
        else:
            raise StopIteration("no more data in " + str(self))

        if self._samples > 0 and self._pos >= self._samples:
            self._data_iter = None
            self._drained = True
            raise StopIteration("no more data in " + str(self))
        else:
            return ret

    def reset(self):
        if self._data_iter is None:
            self._data_iter = self._iter_maker()

        if self._epoch < 0:
            self._epoch = 0
        else:
            self._epoch += 1

        self._pos = 0
        self._drained = False

    def size(self):
        return self._sample_num

    def drained(self):
        assert self._epoch >= 0, "the first epoch has not started yet"
        return self._pos >= self.size()

    def epoch_id(self):
        return self._epoch

