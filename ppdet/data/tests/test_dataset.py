#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import time
import unittest
import sys
import logging
import random
import copy
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from ppdet.data.parallel_map import ParallelMap


class MemorySource(object):
    """ memory data source for testing
    """

    def __init__(self, samples):
        self._epoch = -1

        self._pos = -1
        self._drained = False
        self._samples = samples

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._epoch < 0:
            self.reset()

        if self._pos >= self.size():
            self._drained = True
            raise StopIteration("no more data in " + str(self))
        else:
            sample = copy.deepcopy(self._samples[self._pos])
            self._pos += 1
            return sample

    def reset(self):
        if self._epoch < 0:
            self._epoch = 0
        else:
            self._epoch += 1

        self._pos = 0
        self._drained = False
        random.shuffle(self._samples)

    def size(self):
        return len(self._samples)

    def drained(self):
        assert self._epoch >= 0, "the first epoch has not started yet"
        return self._pos >= self.size()

    def epoch_id(self):
        return self._epoch


class TestDataset(unittest.TestCase):
    """Test cases for ppdet.data.dataset
    """

    @classmethod
    def setUpClass(cls):
        """ setup
        """
        pass

    @classmethod
    def tearDownClass(cls):
        """ tearDownClass """
        pass

    def test_next(self):
        """ test next
        """
        samples = list(range(10))
        mem_sc = MemorySource(samples)

        for i, d in enumerate(mem_sc):
            self.assertTrue(d in samples)

    def test_transform_with_abnormal_worker(self):
        """ test dataset transform with abnormally exit process
        """
        samples = list(range(20))
        mem_sc = MemorySource(samples)

        def _worker(sample):
            if sample == 3:
                sys.exit(1)

            return 2 * sample

        test_worker = ParallelMap(
            mem_sc, _worker, worker_num=2, use_process=True, memsize='2M')

        ct = 0
        for i, d in enumerate(test_worker):
            ct += 1
            self.assertTrue(d / 2 in samples)

        self.assertEqual(len(samples) - 1, ct)

    def test_transform_with_delay_worker(self):
        """ test dataset transform with delayed process
        """
        samples = list(range(20))
        mem_sc = MemorySource(samples)

        def _worker(sample):
            if sample == 3:
                time.sleep(30)

            return 2 * sample

        test_worker = ParallelMap(
            mem_sc, _worker, worker_num=2, use_process=True, memsize='2M')

        ct = 0
        for i, d in enumerate(test_worker):
            ct += 1
            self.assertTrue(d / 2 in samples)

        self.assertEqual(len(samples), ct)


if __name__ == '__main__':
    logging.basicConfig()
    unittest.main()
