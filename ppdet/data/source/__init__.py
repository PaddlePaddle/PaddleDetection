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
from __future__ import division
from __future__ import print_function

import copy

from .roidb_source import RoiDbSource
from .simple_source import SimpleSource
from .iterator_source import IteratorSource


def build_source(config):
    """
    Build dataset from source data, default source type is 'RoiDbSource'
    Args:
        config (dict): should have following structure:
        {
            data_cf (dict):
                anno_file (str): label file or image list file path
                image_dir (str): root directory for images
                samples (int): number of samples to load, -1 means all
                is_shuffle (bool): should samples be shuffled
                load_img (bool): should images be loaded
                mixup_epoch (int): parse mixup in first n epoch
                with_background (bool): whether load background as a class
            cname2cid (dict): the label name to id dictionary
        }
    """
    if 'data_cf' in config:
        data_cf = config['data_cf']
        data_cf['cname2cid'] = config['cname2cid']
    else:
        data_cf = config

    data_cf = {k.lower(): v for k, v in data_cf.items()}

    args = copy.deepcopy(data_cf)
    # defaut type is 'RoiDbSource'
    source_type = 'RoiDbSource'
    if 'type' in data_cf:
        if data_cf['type'] in ['VOCSource', 'COCOSource', 'RoiDbSource']:
            source_type = 'RoiDbSource'
        else:
            source_type = data_cf['type']
        del args['type']
    if source_type == 'RoiDbSource':
        return RoiDbSource(**args)
    elif source_type == 'SimpleSource':
        return SimpleSource(**args)
    else:
        raise ValueError('source type not supported: ' + source_type)
