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

import numpy as np
import functools
import collections
from ..dataset import Dataset


class ProxiedDataset(Dataset):
    """proxy method called to 'self._ds' when if not defined"""

    def __init__(self, ds):
        super(ProxiedDataset, self).__init__()
        self._ds = ds
        methods = filter(lambda k: not k.startswith('_'),
                         Dataset.__dict__.keys())
        for m in methods:
            func = functools.partial(self._proxy_method, getattr(self, m))
            setattr(self, m, func)

    def _proxy_method(self, func, *args, **kwargs):
        """
        proxy call to 'func', if not available then call self._ds.xxx
        whose name is the same with func.__name__
        """
        method = func.__name__
        try:
            return func(*args, **kwargs)
        except NotImplementedError:
            ds_func = getattr(self._ds, method)
            return ds_func(*args, **kwargs)


class MappedDataset(ProxiedDataset):
    def __init__(self, ds, mapper):
        super(MappedDataset, self).__init__(ds)
        self._ds = ds
        self._mapper = mapper

    def next(self):
        sample = self._ds.next()
        return self._mapper(sample)


class BatchedDataset(ProxiedDataset):
    """
    Batching samples

    Args:
        ds (instance of Dataset): dataset to be batched
        batchsize (int): sample number for each batch
        drop_last (bool): drop last samples when not enough for one batch
        drop_empty (bool): drop samples which have empty field
    """

    def __init__(self, ds, batchsize, drop_last=False, drop_empty=True):
        super(BatchedDataset, self).__init__(ds)
        self._batchsz = batchsize
        self._drop_last = drop_last
        self._drop_empty = drop_empty

    def next(self):
        """proxy to self._ds.next"""

        def empty(x):
            if isinstance(x, np.ndarray) and x.size == 0:
                return True
            elif isinstance(x, collections.Sequence) and len(x) == 0:
                return True
            else:
                return False

        def has_empty(items):
            if any(x is None for x in items):
                return True
            if any(empty(x) for x in items):
                return True
            return False

        batch = []
        for _ in range(self._batchsz):
            try:
                out = self._ds.next()
                while self._drop_empty and has_empty(out):
                    out = self._ds.next()
                batch.append(out)
            except StopIteration:
                if not self._drop_last and len(batch) > 0:
                    return batch
                else:
                    raise StopIteration
        return batch
