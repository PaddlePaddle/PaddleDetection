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
from __future__ import print_function

import copy
import logging
import traceback

from .transformer import MappedDataset, BatchedDataset
from .post_map import build_post_map
from .parallel_map import ParallelMappedDataset
from .operators import BaseOperator, registered_ops

__all__ = ['build_mapper', 'map', 'batch', 'batch_map']

logger = logging.getLogger(__name__)


def build_mapper(ops, context=None):
    """
    Build a mapper for operators in 'ops'

    Args:
        ops (list of operator.BaseOperator or list of op dict):
            configs for oprators, eg:
            [{'name': 'DecodeImage', 'params': {'to_rgb': True}}, {xxx}]
        context (dict): a context object for mapper

    Returns:
        a mapper function which accept one argument 'sample' and
        return the processed result
    """
    new_ops = []
    for _dict in ops:
        new_dict = {}
        for i, j in _dict.items():
            new_dict[i.lower()] = j
        new_ops.append(new_dict)
    ops = new_ops
    op_funcs = []
    op_repr = []
    for op in ops:
        if type(op) is dict and 'op' in op:
            op_func = getattr(BaseOperator, op['op'])
            params = copy.deepcopy(op)
            del params['op']
            o = op_func(**params)
        elif not isinstance(op, BaseOperator):
            op_func = getattr(BaseOperator, op['name'])
            params = {} if 'params' not in op else op['params']
            o = op_func(**params)
        else:
            assert isinstance(op, BaseOperator), \
                "invalid operator when build ops"
            o = op
        op_funcs.append(o)
        op_repr.append('{{}}'.format(str(o)))
    op_repr = '[{}]'.format(','.join(op_repr))

    def _mapper(sample):
        ctx = {} if context is None else copy.deepcopy(context)
        for f in op_funcs:
            try:
                out = f(sample, ctx)
                sample = out
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warn("fail to map op [{}] with error: {} and stack:\n{}".format(f, e, str(stack_info)))
                raise e

        return out

    _mapper.ops = op_repr
    return _mapper


def map(ds, mapper, worker_args=None):
    """
    Apply 'mapper' to 'ds'

    Args:
        ds (instance of Dataset): dataset to be mapped
        mapper (function): action to be executed for every data sample
        worker_args (dict): configs for concurrent mapper
    Returns:
        a mapped dataset
    """

    if worker_args is not None:
        return ParallelMappedDataset(ds, mapper, worker_args)
    else:
        return MappedDataset(ds, mapper)


def batch(ds, batchsize, drop_last=False, drop_empty=True):
    """
    Batch data samples to batches
    Args:
        batchsize (int): number of samples for a batch
        drop_last (bool): drop last few samples if not enough for a batch

    Returns:
        a batched dataset
    """

    return BatchedDataset(ds, 
                          batchsize, 
                          drop_last=drop_last, 
                          drop_empty=drop_empty)


def batch_map(ds, config):
    """
    Post process the batches.

    Args:
        ds (instance of Dataset): dataset to be mapped
        mapper (function): action to be executed for every batch
    Returns:
        a batched dataset which is processed
    """

    mapper = build_post_map(**config)
    return MappedDataset(ds, mapper)


for nm in registered_ops:
    op = getattr(BaseOperator, nm)
    locals()[nm] = op

__all__ += registered_ops
