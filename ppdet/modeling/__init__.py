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

import warnings
warnings.filterwarnings(
    action='ignore', category=DeprecationWarning, module='ops')

from . import ops
from . import backbones
from . import necks
from . import proposal_generator
from . import heads
from . import losses
from . import architectures
from . import post_process
from . import layers
from . import reid
from . import mot
from . import transformers
from . import assigners
from . import rbox_utils
from . import ssod

from .ops import *
from .backbones import *
from .necks import *
from .proposal_generator import *
from .heads import *
from .losses import *
from .architectures import *
from .post_process import *
from .layers import *
from .reid import *
from .mot import *
from .transformers import *
from .assigners import *
from .rbox_utils import *
from .ssod import *
