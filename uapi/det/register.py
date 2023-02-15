# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
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

import os.path as osp

from ..base.register import register_model_info, register_suite_info
from .model import DetModel
from .config import DetConfig
from .runner import DetRunner

# XXX: Hard-code relative path of repo root dir
REPO_ROOT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
register_suite_info({
    'suite_name': 'Det',
    'model': DetModel,
    'runner': DetRunner,
    'config': DetConfig,
    'runner_root_path': REPO_ROOT_PATH
})

register_model_info({
    'model_name': 'picodet_s_320',
    'suite': 'Det',
    'config_path': osp.join(REPO_ROOT_PATH, 'configs/picodet/legacy_model/picodet_s_320_coco.yml'),
    'auto_compression_config_path': osp.join(REPO_ROOT_PATH, 'configs/picodet_s_qat_dis.yaml'),
    'supported_apis': ['train', 'predict', 'export', 'infer', 'compression']
})
