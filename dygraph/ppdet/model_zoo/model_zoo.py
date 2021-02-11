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

import os
import os.path as osp
import glob
import pkg_resources

try:
    from collections.abc import Sequence
except:
    from collections import Sequence

from ppdet.core.workspace import load_config, create
from ppdet.utils.checkpoint import load_weight
from ppdet.utils.download import get_config_path

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = [
    'list_model', 'get_config_file', 'get_weights_url', 'get_model',
    'MODEL_ZOO_FILENAME'
]

MODEL_ZOO_FILENAME = 'MODEL_ZOO'


def list_model(filters=[]):
    model_zoo_file = pkg_resources.resource_filename('ppdet.model_zoo',
                                                     MODEL_ZOO_FILENAME)
    with open(model_zoo_file) as f:
        model_names = f.read().splitlines()

    # filter model_name
    def filt(name):
        for f in filters:
            if name.find(f) < 0:
                return False
        return True

    if isinstance(filters, str) or not isinstance(filters, Sequence):
        filters = [filters]
    model_names = [name for name in model_names if filt(name)]
    if len(model_names) == 0 and len(filters) > 0:
        raise ValueError("no model found, please check filters seeting, "
                         "filters can be set as following kinds:\n"
                         "\tDataset: coco, voc ...\n"
                         "\tArchitecture: yolo, rcnn, ssd ...\n"
                         "\tBackbone: resnet, vgg, darknet ...\n")

    model_str = "Available Models:\n"
    for model_name in model_names:
        model_str += "\t{}\n".format(model_name)
    logger.info(model_str)


# models and configs save on bcebos under dygraph directory
def get_config_file(model_name):
    return get_config_path("ppdet://dygraph/configs/{}.yml".format(model_name))


def get_weights_url(model_name):
    return "ppdet://dygraph/{}.pdparams".format(model_name)


def get_model(model_name, pretrained=True):
    cfg_file = get_config_file(model_name)
    cfg = load_config(cfg_file)
    model = create(cfg.architecture)

    if pretrained:
        load_weight(model, get_weights_url(model_name))

    return model
