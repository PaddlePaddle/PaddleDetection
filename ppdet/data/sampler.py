import os
import sys
import six
import time
import math
import socket
import contextlib
import numpy as np

from paddle import fluid
from paddle.io import BatchSampler
from paddle.fluid.layers import collective
from paddle.distributed import ParallelEnv
from paddle.fluid.dygraph.parallel import ParallelStrategy

_parallel_context_initialized = False


class DistributedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, \
                "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
                "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
                "drop_last should be a boolean number"

        self.drop_last = drop_last
        self.nranks = ParallelEnv().nranks
        self.local_rank = ParallelEnv().local_rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.nranks))
        self.total_size = self.num_samples * self.nranks

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
            self.epoch += 1

        # subsample
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            last_batch_size = self.total_size % (self.batch_size * self.nranks)
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(self.local_rank * self.batch_size,
                           len(indices) - last_batch_size,
                           self.batch_size * self.nranks):
                subsampled_indices.extend(indices[i:i + self.batch_size])

            indices = indices[len(indices) - last_batch_size:]
            subsampled_indices.extend(indices[
                self.local_rank * last_local_batch_size:(
                    self.local_rank + 1) * last_local_batch_size])
            return subsampled_indices

        if self.nranks > 1:
            indices = _get_indices_by_batch_size(indices)

        assert len(indices) == self.num_samples
        _sample_iter = iter(indices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        num_samples = self.num_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch


def wait_server_ready(endpoints):
    assert not isinstance(endpoints, six.string_types)
    while True:
        all_ok = True
        not_ready_endpoints = []
        for ep in endpoints:
            ip_port = ep.split(":")
            with contextlib.closing(
                    socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.settimeout(2)
                result = sock.connect_ex((ip_port[0], int(ip_port[1])))
                if result != 0:
                    all_ok = False
                    not_ready_endpoints.append(ep)
        if not all_ok:
            time.sleep(3)
        else:
            break


def init_communicator(program, rank, nranks, wait_port, current_endpoint,
                      endpoints):
    if nranks < 2:
        return
    other_endpoints = endpoints[:]
    other_endpoints.remove(current_endpoint)
    if rank == 0 and wait_port:
        wait_server_ready(other_endpoints)
    block = program.global_block()
    nccl_id_var = block.create_var(
        name=fluid.unique_name.generate('nccl_id'),
        persistable=True,
        type=fluid.core.VarDesc.VarType.RAW)

    block.append_op(
        type='c_gen_nccl_id',
        inputs={},
        outputs={'Out': nccl_id_var},
        attrs={
            'rank': rank,
            'endpoint': current_endpoint,
            'other_endpoints': other_endpoints
        })

    block.append_op(
        type='c_comm_init',
        inputs={'X': nccl_id_var},
        outputs={},
        attrs={
            'nranks': nranks,
            'rank': rank,
            'ring_id': 0,
        })


def prepare_distributed_context(place=None):
    if place is None:
        place = fluid.CUDAPlace(ParallelEnv().dev_id) if ParallelEnv().nranks > 1 \
            else fluid.CUDAPlace(0)

    strategy = ParallelStrategy()
    strategy.nranks = ParallelEnv().nranks
    strategy.local_rank = ParallelEnv().local_rank
    strategy.trainer_endpoints = ParallelEnv().trainer_endpoints
    strategy.current_endpoint = ParallelEnv().current_endpoint

    if strategy.nranks < 2:
        return

    global _parallel_context_initialized

    if not _parallel_context_initialized and isinstance(place, fluid.CUDAPlace):

        def _init_context():
            communicator_prog = fluid.Program()
            init_communicator(communicator_prog, strategy.local_rank,
                              strategy.nranks, True, strategy.current_endpoint,
                              strategy.trainer_endpoints)
            exe = fluid.Executor(place)
            exe.run(communicator_prog)

        fluid.disable_dygraph()
        _init_context()
        fluid.enable_dygraph(place)

    else:
        assert ("Only support CUDAPlace for now.")

    _parallel_context_initialized = True
    return strategy
