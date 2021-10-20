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

from . import rpn_head
from . import yolo_head
from . import retina_head
from . import fcos_head
from . import corner_head
from . import efficient_head
from . import ttf_head
from . import solov2_head
from . import eb_head

from .rpn_head import *
from .yolo_head import *
from .retina_head import *
from .fcos_head import *
from .corner_head import *
from .efficient_head import *
from .ttf_head import *
from .solov2_head import *
from .eb_head import *
