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

# function:
#    interface for accessing data samples in stream

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Dataset(object):
    """interface to access a stream of data samples"""

    def __init__(self):
        self._epoch = -1

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def __str__(self):
        return "{}(fname:{}, epoch:{:d}, size:{:d}, pos:{:d})".format(
            type(self).__name__, self._fname, self._epoch,
            self.size(), self._pos)

    def next(self):
        """get next sample"""
        raise NotImplementedError('%s.next not available' %
                                  (self.__class__.__name__))

    def reset(self):
        """reset to initial status and begins a new epoch"""
        raise NotImplementedError('%s.reset not available' %
                                  (self.__class__.__name__))

    def size(self):
        """get number of samples in this dataset"""
        raise NotImplementedError('%s.size not available' %
                                  (self.__class__.__name__))

    def drained(self):
        """whether all sampled has been readed out for this epoch"""
        raise NotImplementedError('%s.drained not available' %
                                  (self.__class__.__name__))

    def epoch_id(self):
        """return epoch id for latest sample"""
        raise NotImplementedError('%s.epoch_id not available' %
                                  (self.__class__.__name__))
