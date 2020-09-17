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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import json
import os
import os.path as osp
import sys
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re


def main():
    """
    create label_list.txt from dataset, support voc and coco dataset_type.
    
    # create_lable_list from voc dataset
    python tools/create_label_list.py --dataset_type voc --xml_dir path_to_xml_dir
    
    # create_lable_list from coco dataset
    python tools/create_label_list.py --dataset_type coco --json_path path_to_json_path
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_type', help='the type of dataset')
    parser.add_argument('--xml_dir', help='xml directory')
    parser.add_argument('--json_path', help='json_path')
    parser.add_argument(
        '--outfile_path',
        help='output file path',
        default='output/label_list.txt')

    args = parser.parse_args()

    if args.dataset_type == 'coco' or args.dataset_type == 'COCO':
        assert args.dataset_type and args.json_path
        try:
            assert os.path.exists(args.json_path)
        except AssertionError as e:
            print('The json file: {} does not exist!'.format(args.json_path))
            os._exit(0)

        anno = json.load(open(args.json_path))
        categories_list = [cat['name'] for cat in anno['categories']]

        with open(args.outfile_path, 'w') as f:
            for cat in categories_list:
                f.write(cat + '\n')
        print('lable_list file: {} create done!'.format(args.outfile_path))

    if args.dataset_type == 'voc' or args.dataset_type == 'VOC':
        assert args.dataset_type and args.xml_dir
        try:
            assert os.path.exists(args.xml_dir)
        except AssertionError as e:
            print('The xml_dir: {} does not exist!'.format(args.xml_dir))
            os._exit(0)

        categories_list = set()
        for xml_file in tqdm(glob.glob(osp.join(args.xml_dir, '*.xml'))):
            # Read annotation xml
            ann_tree = ET.parse(xml_file)
            ann_root = ann_tree.getroot()

            for obj in ann_root.findall('object'):
                cat_name = obj.findtext('name')
                categories_list.add(cat_name)

        categories_list = sorted(categories_list)

        with open(args.outfile_path, 'w') as f:
            for cat in categories_list:
                f.write(cat + '\n')
        print('lable_list file: {} create done!'.format(args.outfile_path))


if __name__ == '__main__':
    main()
