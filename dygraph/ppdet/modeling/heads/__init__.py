# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from . import rpn_head
from . import bbox_head
from . import mask_head
from . import yolo_head
from . import roi_extractor
from . import ssd_head
from . import solov2_head

from .rpn_head import *
from .bbox_head import *
from .mask_head import *
from .yolo_head import *
from .roi_extractor import *
from .ssd_head import *
from .solov2_head import *
