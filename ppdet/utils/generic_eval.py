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
from __future__ import unicode_literals

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np

from .coco_eval import bbox2out

import logging
logger = logging.getLogger(__name__)

__all__ = ['bbox2out', 'get_category_info']


def _load_map(map_path="",
              key_idx=0,
              val_idx=1,
              key_type=str,
              val_type=str,
              split_sig="\t"):
    assert key_idx != val_idx, "key value index can not be equal, but they are both {}.".format(
        key_idx)
    maps = dict()
    with open(map_path, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            substr = line.strip("\n").split(split_sig)
            maps[key_type(substr[key_idx])] = val_type(substr[val_idx])
    return maps


def get_category_info(anno_file=None,
                      with_background=True,
                      use_default_label=False):
    clsid2catid = {k: k for k in range(1, 677)}

    catid2name = _load_map(
        map_path="./ppdet/utils/generic_det_label.map",
        key_idx=0,
        val_idx=1,
        key_type=int,
        val_type=str,
        split_sig="\t")

    if not with_background:
        clsid2catid = {k - 1: v for k, v in clsid2catid.items()}
    return clsid2catid, catid2name
