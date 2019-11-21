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

#from .operators import BaseOperator, registered_ops
from . import operators
from . import batch_operators
from . import loader
from . import coco
from . import voc

from .operators import *
from .batch_operators import *
from .loader import *
from .coco import *
from .voc import *

__all__ = []

#for nm in registered_ops:
#    op = getattr(BaseOperator, nm)
#    locals()[nm] = op

__all__ += registered_ops

print(__all__)
