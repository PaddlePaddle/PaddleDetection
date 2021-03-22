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

import os
import os.path as osp
import re
import random
import shutil

__all__ = ['create_list']


def create_list(year_dirs, output_dir):
    """
    create following list:
        1. trainval.txt
        2. test.txt
    """
    trainval_list = []
    test_list = []
    for year_dir in year_dirs:
        trainval, test = _walk_voc_dir(year_dir, output_dir)
        trainval_list.extend(trainval)
        test_list.extend(test)

    random.shuffle(trainval_list)
    with open(osp.join(output_dir, 'trainval.txt'), 'w') as ftrainval:
        for item in trainval_list:
            ftrainval.write(item[0] + ' ' + item[1] + '\n')

    with open(osp.join(output_dir, 'test.txt'), 'w') as fval:
        ct = 0
        for item in test_list:
            ct += 1
            fval.write(item[0] + ' ' + item[1] + '\n')


def _walk_voc_dir(year_dir, output_dir):
    filelist_dir = osp.join(year_dir, 'ImageSets/Main')
    annotation_dir = osp.join(year_dir, 'Annotations')
    img_dir = osp.join(year_dir, 'JPEGImages')
    trainval_list = []
    test_list = []
    added = set()

    img_dict = {}
    for img_file in os.listdir(img_dir):
        img_dict[img_file.split('.')[0]] = img_file

    for _, _, files in os.walk(filelist_dir):
        for fname in files:
            img_ann_list = []
            if re.match('trainval\.txt', fname):
                img_ann_list = trainval_list
            elif re.match('test\.txt', fname):
                img_ann_list = test_list
            else:
                continue
            fpath = osp.join(filelist_dir, fname)
            for line in open(fpath):
                name_prefix = line.strip().split()[0]
                if name_prefix in added:
                    continue
                added.add(name_prefix)
                ann_path = osp.join(
                    osp.relpath(annotation_dir, output_dir),
                    name_prefix + '.xml')
                img_path = osp.join(
                    osp.relpath(img_dir, output_dir), img_dict[name_prefix])
                img_ann_list.append((img_path, ann_path))

    return trainval_list, test_list
