# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from . import utils
from . import task_aligned_assigner
from . import atss_assigner
from . import simota_assigner
from . import max_iou_assigner
from . import fcosr_assigner
from . import rotated_task_aligned_assigner
from . import task_aligned_assigner_cr
from . import uniform_assigner

from .utils import *
from .task_aligned_assigner import *
from .atss_assigner import *
from .simota_assigner import *
from .max_iou_assigner import *
from .fcosr_assigner import *
from .rotated_task_aligned_assigner import *
from .task_aligned_assigner_cr import *
from .uniform_assigner import *
from .hungarian_assigner import *
from .pose_utils import *
