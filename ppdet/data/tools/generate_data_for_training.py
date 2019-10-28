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

# function:
#   tool used convert COCO or VOC data to a pickled file whose
#   schema for each sample is the same.
#
# notes:
#   Original data format of COCO or VOC can also be directly
#   used by 'PPdetection' to train.
#   This tool just convert data to a unified schema,
#   and it's useful when debuging with small dataset.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

import os
import sys
import logging
import pickle as pkl

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
if path not in sys.path:
    sys.path.insert(0, path)

from data.source import loader


def parse_args():
    """ parse arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate Standard Dataset for PPdetection')

    parser.add_argument(
        '--type',
        type=str,
        default='json',
        help='file format of label file, eg: json for COCO and xml for VOC')
    parser.add_argument(
        '--annotation',
        type=str,
        help='label file name for COCO or VOC dataset, '
        'eg: instances_val2017.json or train.txt')
    parser.add_argument(
        '--save-dir',
        type=str,
        default='roidb',
        help='directory to save roidb file which contains pickled samples')
    parser.add_argument(
        '--samples',
        type=int,
        default=-1,
        help='number of samples to dump, default to all')

    args = parser.parse_args()
    return args


def dump_coco_as_pickle(args):
    """ Load COCO data, and then save it as pickled file.

        Notes:
            label file of COCO contains a json which consists
            of label info for each sample
    """
    samples = args.samples
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    anno_path = args.annotation
    roidb, cat2id = loader.load(anno_path, samples, with_cat2id=True)
    samples = len(roidb)
    dsname = os.path.basename(anno_path).rstrip('.json')
    roidb_fname = save_dir + "/%s.roidb" % (dsname)
    with open(roidb_fname, "wb") as fout:
        pkl.dump((roidb, cat2id), fout)

    #for rec in roidb:
    #    sys.stderr.write('%s\n' % (rec['im_file']))
    logging.info('dumped %d samples to file[%s]' % (samples, roidb_fname))


def dump_voc_as_pickle(args):
    """ Load VOC data, and then save it as pickled file.

        Notes:
            we assume label file of VOC contains lines
            each of which corresponds to a xml file
            that contains it's label info
    """
    samples = args.samples
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = args.save_dir
    anno_path = os.path.expanduser(args.annotation)
    roidb, cat2id = loader.load(
        anno_path, samples, with_cat2id=True, use_default_label=None)
    samples = len(roidb)
    part = anno_path.split('/')
    dsname = part[-4]
    roidb_fname = save_dir + "/%s.roidb" % (dsname)
    with open(roidb_fname, "wb") as fout:
        pkl.dump((roidb, cat2id), fout)
    anno_path = os.path.join(anno_path.split('/train.txt')[0], 'label_list.txt')
    with open(anno_path, 'w') as fw:
        for key in cat2id.keys():
            fw.write(key + '\n')
    logging.info('dumped %d samples to file[%s]' % (samples, roidb_fname))


if __name__ == "__main__":
    """ Make sure you have already downloaded original COCO or VOC data,
        then you can convert it using this tool.

    Usage:
        python generate_data_for_training.py --type=json
            --annotation=./annotations/instances_val2017.json
            --save-dir=./roidb --samples=100
    """
    args = parse_args()

    # VOC data are organized in xml files
    if args.type == 'xml':
        dump_voc_as_pickle(args)
    # COCO data are organized in json file
    elif args.type == 'json':
        dump_coco_as_pickle(args)
    else:
        TypeError('Can\'t deal with {} type. '\
            'Only xml or json file format supported'.format(args.type))
