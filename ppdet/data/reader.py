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
#    Interface to build readers for detection data like COCO or VOC
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from numbers import Integral

import logging
from .source import build_source
from .transform import build_mapper, map, batch, batch_map

logger = logging.getLogger(__name__)


class Reader(object):
    """Interface to make readers for training or evaluation"""

    def __init__(self, data_cf, trans_conf, maxiter=-1):
        self._data_cf = data_cf
        self._trans_conf = trans_conf
        self._maxiter = maxiter
        self._cname2cid = None
        assert isinstance(self._maxiter, Integral), "maxiter should be int"

    def _make_reader(self, mode, my_source=None):
        """Build reader for training or validation"""
        if my_source is None:
            file_conf = self._data_cf[mode]

            # 1, Build data source

            sc_conf = {'data_cf': file_conf, 'cname2cid': self._cname2cid}
            sc = build_source(sc_conf)
        else:
            sc = my_source

        # 2, Buid a transformed dataset
        ops = self._trans_conf[mode]['OPS']
        batchsize = self._trans_conf[mode]['BATCH_SIZE']
        drop_last = False if 'DROP_LAST' not in \
            self._trans_conf[mode] else self._trans_conf[mode]['DROP_LAST']

        mapper = build_mapper(ops, {'is_train': mode == 'TRAIN'})

        worker_args = None
        if 'WORKER_CONF' in self._trans_conf[mode]:
            worker_args = self._trans_conf[mode]['WORKER_CONF']
            worker_args = {k.lower(): v for k, v in worker_args.items()}

        mapped_ds = map(sc, mapper, worker_args)
        # In VAL mode, gt_bbox, gt_label can be empty, and should
        # not be dropped
        batched_ds = batch(mapped_ds, batchsize, drop_last, 
                           drop_empty=(mode!="VAL"))

        trans_conf = {k.lower(): v for k, v in self._trans_conf[mode].items()}
        need_keys = {
            'is_padding',
            'coarsest_stride',
            'random_shapes',
            'multi_scales',
            'use_padded_im_info',
        }
        bm_config = {
            key: value
            for key, value in trans_conf.items() if key in need_keys
        }

        batched_ds = batch_map(batched_ds, bm_config)

        batched_ds.reset()
        if mode.lower() == 'train':
            if self._cname2cid is not None:
                logger.warn('cname2cid already set, it will be overridden')
            self._cname2cid = getattr(sc, 'cname2cid', None)

        # 3, Build a reader
        maxit = -1 if self._maxiter <= 0 else self._maxiter

        def _reader():
            n = 0
            while True:
                for _batch in batched_ds:
                    yield _batch
                    n += 1
                    if maxit > 0 and n == maxit:
                        return
                batched_ds.reset()
                if maxit <= 0:
                    return

        if hasattr(sc, 'get_imid2path'):
            _reader.imid2path = sc.get_imid2path()

        return _reader

    def train(self):
        """Build reader for training"""
        return self._make_reader('TRAIN')

    def val(self):
        """Build reader for validation"""
        return self._make_reader('VAL')

    def test(self):
        """Build reader for inference"""
        return self._make_reader('TEST')

    @classmethod
    def create(cls, mode, data_config,
            transform_config, max_iter=-1,
            my_source=None, ret_iter=True):
        """ create a specific reader """
        reader = Reader({mode: data_config},
            {mode: transform_config}, max_iter)
        if ret_iter:
            return reader._make_reader(mode, my_source)
        else:
            return reader
