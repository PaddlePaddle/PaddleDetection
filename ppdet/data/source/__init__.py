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

from . import coco
from . import voc
from . import widerface
from . import category
from . import keypoint_coco
from . import mot
from . import sniper_coco

from .coco import *
from .voc import *
from .widerface import *
from .category import *
from .keypoint_coco import *
from .mot import *
from .sniper_coco import SniperCOCODataSet
from .dataset import ImageFolder
from .pose3d_cmb import Pose3DDataset
