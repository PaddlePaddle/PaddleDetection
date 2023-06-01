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

from . import yolo_loss
from . import iou_aware_loss
from . import iou_loss
from . import ssd_loss
from . import fcos_loss
from . import solov2_loss
from . import ctfocal_loss
from . import keypoint_loss
from . import jde_loss
from . import fairmot_loss
from . import gfocal_loss
from . import detr_loss
from . import sparsercnn_loss
from . import focal_loss
from . import smooth_l1_loss
from . import probiou_loss
from . import cot_loss
from . import supcontrast
from . import queryinst_loss
from . import clrnet_loss
from . import clrnet_line_iou_loss

from .yolo_loss import *
from .iou_aware_loss import *
from .iou_loss import *
from .ssd_loss import *
from .fcos_loss import *
from .solov2_loss import *
from .ctfocal_loss import *
from .keypoint_loss import *
from .jde_loss import *
from .fairmot_loss import *
from .gfocal_loss import *
from .detr_loss import *
from .sparsercnn_loss import *
from .focal_loss import *
from .smooth_l1_loss import *
from .pose3d_loss import *
from .probiou_loss import *
from .cot_loss import *
from .supcontrast import *
from .queryinst_loss import *
from .clrnet_loss import *
from .clrnet_line_iou_loss import *