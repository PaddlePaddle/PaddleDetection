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
