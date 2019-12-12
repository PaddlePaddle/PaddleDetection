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
from __future__ import unicode_literals

import sys
import six
if six.PY3:
    import pickle
    from io import BytesIO as StringIO
    from queue import Empty
else:
    import cPickle as pickle
    from cStringIO import StringIO
    from Queue import Empty

import logging
import traceback
import multiprocessing as mp
from multiprocessing.queues import Queue
from .sharedmemory import SharedMemoryMgr

logger = logging.getLogger(__name__)


class SharedQueueError(ValueError):
    """ SharedQueueError
    """
    pass


class SharedQueue(Queue):
    """ a Queue based on shared memory to communicate data between Process,
        and it's interface is compatible with 'multiprocessing.queues.Queue'
    """

    def __init__(self, maxsize=0, mem_mgr=None, memsize=None, pagesize=None):
        """ init
        """
        if six.PY3:
            super(SharedQueue, self).__init__(maxsize, ctx=mp.get_context())
        else:
            super(SharedQueue, self).__init__(maxsize)

        if mem_mgr is not None:
            self._shared_mem = mem_mgr
        else:
            self._shared_mem = SharedMemoryMgr(
                capacity=memsize, pagesize=pagesize)

    def put(self, obj, **kwargs):
        """ put an object to this queue
        """
        obj = pickle.dumps(obj, -1)
        buff = None
        try:
            buff = self._shared_mem.malloc(len(obj))
            buff.put(obj)
            super(SharedQueue, self).put(buff, **kwargs)
        except Exception as e:
            stack_info = traceback.format_exc()
            err_msg = 'failed to put a element to SharedQueue '\
                'with stack info[%s]' % (stack_info)
            logger.warn(err_msg)

            if buff is not None:
                buff.free()
            raise e

    def get(self, **kwargs):
        """ get an object from this queue
        """
        buff = None
        try:
            buff = super(SharedQueue, self).get(**kwargs)
            data = buff.get()
            return pickle.load(StringIO(data))
        except Empty as e:
            raise e
        except Exception as e:
            stack_info = traceback.format_exc()
            err_msg = 'failed to get element from SharedQueue '\
                        'with stack info[%s]' % (stack_info)
            logger.warn(err_msg)
            raise e
        finally:
            if buff is not None:
                buff.free()

    def release(self):
        self._shared_mem.release()
        self._shared_mem = None
