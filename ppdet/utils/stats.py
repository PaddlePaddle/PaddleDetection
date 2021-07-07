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

import collections
import numpy as np

__all__ = ['SmoothedValue', 'TrainingStats']


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({avg:.4f})"
        self.deque = collections.deque(maxlen=window_size)
        self.fmt = fmt
        self.total = 0.
        self.count = 0

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        return np.median(self.deque)

    @property
    def avg(self):
        return np.mean(self.deque)

    @property
    def max(self):
        return np.max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, max=self.max, value=self.value)


class TrainingStats(object):
    def __init__(self, window_size, delimiter=' '):
        self.meters = None
        self.window_size = window_size
        self.delimiter = delimiter

    def update(self, stats):
        if self.meters is None:
            self.meters = {
                k: SmoothedValue(self.window_size)
                for k in stats.keys()
            }
        for k, v in self.meters.items():
            v.update(stats[k].numpy())

    def get(self, extras=None):
        stats = collections.OrderedDict()
        if extras:
            for k, v in extras.items():
                stats[k] = v
        for k, v in self.meters.items():
            stats[k] = format(v.median, '.6f')

        return stats

    def log(self, extras=None):
        d = self.get(extras)
        strs = []
        for k, v in d.items():
            strs.append("{}: {}".format(k, str(v)))
        return self.delimiter.join(strs)
