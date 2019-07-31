import os
import time
import unittest
import sys
import logging

import set_env
from ppdet.data.source import IteratorSource


def _generate_iter_maker(num=10):
    def _reader():
        for i in range(num):
            yield {'image': 'image_' + str(i), 'label': i}

    return _reader

class TestIteratorSource(unittest.TestCase):
    """Test cases for dataset.source.roidb_source
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

    def test_basic(self):
        """ test basic apis 'next/size/drained'
        """
        iter_maker = _generate_iter_maker()
        iter_source = IteratorSource(iter_maker)
        for i, sample in enumerate(iter_source):
            self.assertTrue('image' in sample)
            self.assertGreater(len(sample['image']), 0)
        self.assertTrue(iter_source.drained())
        self.assertEqual(i + 1, iter_source.size())

    def test_reset(self):
        """ test functions 'reset/epoch_id'
        """
        iter_maker = _generate_iter_maker()
        iter_source = IteratorSource(iter_maker)

        self.assertTrue(iter_source.next() is not None)
        self.assertEqual(iter_source.epoch_id(), 0)

        iter_source.reset()

        self.assertEqual(iter_source.epoch_id(), 1)
        self.assertTrue(iter_source.next() is not None)


if __name__ == '__main__':
    unittest.main()
