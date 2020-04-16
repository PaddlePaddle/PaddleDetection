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

from . import yolo_loss
from . import smooth_l1_loss
from . import giou_loss
from . import diou_loss
from . import iou_loss
from . import balanced_l1_loss
from . import fcos_loss
from . import diou_loss_yolo
from . import iou_aware_loss
from . import ssd_with_lmk_loss

from .iou_aware_loss import *
from .yolo_loss import *
from .smooth_l1_loss import *
from .giou_loss import *
from .diou_loss import *
from .iou_loss import *
from .balanced_l1_loss import *
from .fcos_loss import *
from .diou_loss_yolo import *
from .ssd_with_lmk_loss import *