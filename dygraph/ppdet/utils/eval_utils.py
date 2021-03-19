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

import logging
import os

__all__ = ['json_eval_results']

logger = logging.getLogger(__name__)

def json_eval_results(metric: object,
                      json_directory: object=None,
                      dataset: object=None) -> object:
    """
    cocoapi eval with already exists proposal.json, bbox.json or mask.json
    """
    assert metric == 'COCO'
    from ppdet.metrics.coco_utils import cocoapi_eval
    anno_file = dataset.get_anno()
    json_file_list = ['proposal.json', 'bbox.json', 'mask.json']
    if json_directory:
        assert os.path.exists(
            json_directory), "The json directory:{} does not exist".format(
                json_directory)
        for k, v in enumerate(json_file_list):
            json_file_list[k] = os.path.join(str(json_directory), v)

    coco_eval_style = ['proposal', 'bbox', 'segm']
    for i, v_json in enumerate(json_file_list):
        if os.path.exists(v_json):
            cocoapi_eval(v_json, coco_eval_style[i], anno_file=anno_file)
        else:
            logger.info("{} not exists!".format(v_json))
