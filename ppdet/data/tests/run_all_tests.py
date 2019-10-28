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
#!/usr/bin/python
#-*-coding:utf-8-*-
"""Run all tests
"""

import unittest
import test_loader
import test_operator
import test_roidb_source
import test_iterator_source
import test_transformer
import test_reader

if __name__ == '__main__':
    alltests = unittest.TestSuite([
        unittest.TestLoader().loadTestsFromTestCase(t) \
        for t in [
            test_loader.TestLoader,
            test_operator.TestBase,
            test_roidb_source.TestRoiDbSource,
            test_iterator_source.TestIteratorSource,
            test_transformer.TestTransformer,
            test_reader.TestReader,
        ]
    ])

    was_succ = unittest\
                .TextTestRunner(verbosity=2)\
                .run(alltests)\
                .wasSuccessful()

    exit(0 if was_succ else 1)
