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

from . import faster_rcnn
from . import mask_rcnn
from . import cascade_rcnn
from . import cascade_mask_rcnn
from . import cascade_rcnn_cls_aware
from . import yolo
from . import ssd
from . import retinanet
from . import efficientdet
from . import blazeface
from . import faceboxes
from . import fcos
from . import cornernet_squeeze
from . import ttfnet
from . import htc
from . import solov2

from .faster_rcnn import *
from .mask_rcnn import *
from .cascade_rcnn import *
from .cascade_mask_rcnn import *
from .cascade_rcnn_cls_aware import *
from .yolo import *
from .ssd import *
from .retinanet import *
from .efficientdet import *
from .blazeface import *
from .faceboxes import *
from .fcos import *
from .cornernet_squeeze import *
from .ttfnet import *
from .htc import *
from .solov2 import *
