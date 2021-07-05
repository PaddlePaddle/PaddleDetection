#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import unittest

import contextlib

import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import Program
from paddle.fluid import core


class LayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = 111

    @classmethod
    def tearDownClass(cls):
        pass

    def _get_place(self, force_to_use_cpu=False):
        # this option for ops that only have cpu kernel
        if force_to_use_cpu:
            return core.CPUPlace()
        else:
            if core.is_compiled_with_cuda():
                return core.CUDAPlace(0)
            return core.CPUPlace()

    @contextlib.contextmanager
    def static_graph(self):
        paddle.enable_static()
        scope = fluid.core.Scope()
        program = Program()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program):
                paddle.seed(self.seed)
                paddle.framework.random._manual_program_seed(self.seed)
                yield

    def get_static_graph_result(self,
                                feed,
                                fetch_list,
                                with_lod=False,
                                force_to_use_cpu=False):
        exe = fluid.Executor(self._get_place(force_to_use_cpu))
        exe.run(fluid.default_startup_program())
        return exe.run(fluid.default_main_program(),
                       feed=feed,
                       fetch_list=fetch_list,
                       return_numpy=(not with_lod))

    @contextlib.contextmanager
    def dynamic_graph(self, force_to_use_cpu=False):
        paddle.disable_static()
        with fluid.dygraph.guard(
                self._get_place(force_to_use_cpu=force_to_use_cpu)):
            paddle.seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
            yield
