# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
from convert import load_dota_infos, data_to_coco
from slicebase import SliceBase

wordname_15 = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter'
]

wordname_16 = wordname_15 + ['container-crane']

wordname_18 = wordname_16 + ['airport', 'helipad']

DATA_CLASSES = {
    'dota10': wordname_15,
    'dota15': wordname_16,
    'dota20': wordname_18
}


def parse_args():
    parser = argparse.ArgumentParser('prepare data for training')

    parser.add_argument(
        '--input_dirs',
        nargs='+',
        type=str,
        default=None,
        help='input dirs which contain image and labelTxt dir')

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='output dirs which contain image and labelTxt dir and coco style json file'
    )

    parser.add_argument(
        '--coco_json_file',
        type=str,
        default='',
        help='coco json annotation files')

    parser.add_argument('--subsize', type=int, default=1024, help='patch size')

    parser.add_argument('--gap', type=int, default=200, help='step size')

    parser.add_argument(
        '--data_type', type=str, default='dota10', help='data type')

    parser.add_argument(
        '--rates',
        nargs='+',
        type=float,
        default=[1.],
        help='scales for multi-slice training')

    parser.add_argument(
        '--nproc', type=int, default=8, help='the processor number')

    parser.add_argument(
        '--iof_thr',
        type=float,
        default=0.5,
        help='the minimal iof between a object and a window')

    parser.add_argument(
        '--image_only',
        action='store_true',
        default=False,
        help='only processing image')

    args = parser.parse_args()
    return args


def load_dataset(input_dir, nproc, data_type):
    if 'dota' in data_type.lower():
        infos = load_dota_infos(input_dir, nproc)
    else:
        raise ValueError('only dota dataset is supported now')

    return infos


def main():
    args = parse_args()
    infos = []
    for input_dir in args.input_dirs:
        infos += load_dataset(input_dir, args.nproc, args.data_type)

    slicer = SliceBase(
        args.gap,
        args.subsize,
        args.iof_thr,
        num_process=args.nproc,
        image_only=args.image_only)
    slicer.slice_data(infos, args.rates, args.output_dir)
    if args.coco_json_file:
        infos = load_dota_infos(args.output_dir, args.nproc)
        coco_json_file = os.path.join(args.output_dir, args.coco_json_file)
        class_names = DATA_CLASSES[args.data_type]
        data_to_coco(infos, coco_json_file, class_names, args.nproc)


if __name__ == '__main__':
    main()
