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
#   transform samples in 'source' using 'mapper'

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import six
import uuid
import logging
import signal
import threading
from .transformer import ProxiedDataset

logger = logging.getLogger(__name__)


class EndSignal(object):
    def __init__(self, errno=0, errmsg=''):
        self.errno = errno
        self.errmsg = errmsg


class ParallelMappedDataset(ProxiedDataset):
    """
    Transform samples to mapped samples which is similar to 'basic.MappedDataset',
    but multiple workers (threads or processes) will be used

    Notes:
        this class is not thread-safe
    """

    def __init__(self, source, mapper, worker_args):
        super(ParallelMappedDataset, self).__init__(source)
        worker_args = {k.lower(): v for k, v in worker_args.items()}

        args = {'bufsize': 100, 'worker_num': 8,
            'use_process': False, 'memsize': '3G'}
        args.update(worker_args)
        if args['use_process'] and type(args['memsize']) is str:
            assert args['memsize'][-1].lower() == 'g', \
                "invalid param for memsize[%s], should be ended with 'G' or 'g'" % (args['memsize'])
            gb = args['memsize'][:-1]
            args['memsize'] = int(gb) * 1024 ** 3

        self._worker_args = args
        self._started = False
        self._source = source
        self._mapper = mapper
        self._exit = False
        self._setup()

    def _setup(self):
        """setup input/output queues and workers """
        use_process = self._worker_args.get('use_process', False)
        if use_process and sys.platform == "win32":
            logger.info("Use multi-thread reader instead of "
                        "multi-process reader on Windows.")
            use_process = False

        bufsize = self._worker_args['bufsize']
        if use_process:
            from .shared_queue import SharedQueue as Queue
            from multiprocessing import Process as Worker
            from multiprocessing import Event
            memsize = self._worker_args['memsize']
            self._inq = Queue(bufsize, memsize=memsize)
            self._outq = Queue(bufsize, memsize=memsize)
        else:
            if six.PY3:
                from queue import Queue
            else:
                from Queue import Queue
            from threading import Thread as Worker
            from threading import Event
            self._inq = Queue(bufsize)
            self._outq = Queue(bufsize)

        consumer_num = self._worker_args['worker_num']
        id = str(uuid.uuid4())[-3:]
        self._producer = threading.Thread(
            target=self._produce,
            args=('producer-' + id, self._source, self._inq))
        self._producer.daemon = True

        self._consumers = []
        for i in range(consumer_num):
            p = Worker(
                target=self._consume,
                args=('consumer-' + id + '_' + str(i), self._inq, self._outq,
                      self._mapper))
            self._consumers.append(p)
            p.daemon = True

        self._epoch = -1
        self._feeding_ev = Event()
        self._produced = 0  # produced sample in self._produce
        self._consumed = 0  # consumed sample in self.next
        self._stopped_consumers = 0

    def _produce(self, id, source, inq):
        """Fetch data from source and feed it to 'inq' queue"""
        while True:
            self._feeding_ev.wait()
            if self._exit:
                break
            try:
                inq.put(source.next())
                self._produced += 1
            except StopIteration:
                self._feeding_ev.clear()
                self._feeding_ev.wait()  # wait other guy to wake up me
                logger.debug("producer[{}] starts new epoch".format(id))
            except Exception as e:
                msg = "producer[{}] failed with error: {}".format(id, str(e))
                inq.put(EndSignal(-1, msg))
                break

        logger.debug("producer[{}] exits".format(id))

    def _consume(self, id, inq, outq, mapper):
        """Fetch data from 'inq', process it and put result to 'outq'"""
        while True:
            sample = inq.get()
            if isinstance(sample, EndSignal):
                sample.errmsg += "[consumer[{}] exits]".format(id)
                outq.put(sample)
                logger.debug("end signal received, " +
                             "consumer[{}] exits".format(id))
                break

            try:
                result = mapper(sample)
                outq.put(result)
            except Exception as e:
                msg = 'failed to map consumer[%s], error: {}'.format(str(e), id)
                outq.put(EndSignal(-1, msg))
                break

    def drained(self):
        assert self._epoch >= 0, "first epoch has not started yet"
        return self._source.drained() and self._produced == self._consumed

    def stop(self):
        """ notify to exit
        """
        self._exit = True
        self._feeding_ev.set()
        for _ in range(len(self._consumers)):
            self._inq.put(EndSignal(0, "notify consumers to exit"))

    def next(self):
        """ get next transformed sample
        """
        if self._epoch < 0:
            self.reset()

        if self.drained():
            raise StopIteration()

        while True:
            sample = self._outq.get()
            if isinstance(sample, EndSignal):
                self._stopped_consumers += 1
                if sample.errno != 0:
                    logger.warn("consumer failed with error: {}".format(
                        sample.errmsg))

                if self._stopped_consumers < len(self._consumers):
                    self._inq.put(sample)
                else:
                    raise ValueError("all consumers exited, no more samples")
            else:
                self._consumed += 1
                return sample

    def reset(self):
        """ reset for a new epoch of samples
        """
        if self._epoch < 0:
            self._epoch = 0
            for p in self._consumers:
                p.start()
            self._producer.start()
        else:
            if not self.drained():
                logger.warn("do not reset before epoch[%d] finishes".format(
                    self._epoch))
                self._produced = self._produced - self._consumed
            else:
                self._produced = 0

            self._epoch += 1

        assert self._stopped_consumers == 0, "some consumers already exited," \
            + " cannot start another epoch"

        self._source.reset()
        self._consumed = 0
        self._feeding_ev.set()


# FIXME(dengkaipeng): fix me if you have better impliment
# handle terminate reader process, do not print stack frame
def _reader_exit(signum, frame):
    logger.debug("Reader process exit.")
    sys.exit()


signal.signal(signal.SIGTERM, _reader_exit)
