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
#   load data records from local files(maybe in COCO or VOC data formats)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
import logging
import pickle as pkl

logger = logging.getLogger(__name__)


def check_records(records):
    """ check the fields of 'records' must contains some keys
    """
    needed_fields = [
        'im_file', 'im_id', 'h', 'w', 'is_crowd', 'gt_class', 'gt_bbox',
        'gt_poly'
    ]

    for i, rec in enumerate(records):
        for k in needed_fields:
            assert k in rec, 'not found field[%s] in record[%d]' % (k, i)


def load_roidb(anno_file, sample_num=-1):
    """ load normalized data records from file 
        'anno_file' which is a pickled file.
        And the records should has a structure:
        {
            'im_file': str, # image file name
            'im_id': int, # image id
            'h': int, # height of image
            'w': int, # width of image
            'is_crowd': bool,
            'gt_class': list of np.ndarray, # classids info
            'gt_bbox': list of np.ndarray, # bounding box info
            'gt_poly': list of int, # poly info
        }

    Args:
        anno_file (str): file name for picked records
        sample_num (int): number of samples to load

    Returns:
        list of records for detection model training
    """

    assert anno_file.endswith('.roidb'), 'invalid roidb file[%s]' % (anno_file)
    with open(anno_file, 'rb') as f:
        roidb = f.read()
        # for support python3 and python2
        try:
            records, cname2cid = pkl.loads(roidb, encoding='bytes')
        except:
            records, cname2cid = pkl.loads(roidb)

        assert type(records) is list, 'invalid data type from roidb'

    if sample_num > 0 and sample_num < len(records):
        records = records[:sample_num]

    return records, cname2cid


def load(fname,
         samples=-1,
         with_background=True,
         with_cat2id=False,
         use_default_label=None,
         cname2cid=None):
    """ Load data records from 'fnames'

    Args:
        fnames (str): file name for data record, eg:
            instances_val2017.json or COCO17_val2017.roidb
        samples (int): number of samples to load, default to all
        with_background (bool): whether load background as a class.
                                default True.
        with_cat2id (bool): whether return cname2cid info out
        use_default_label (bool): whether use the default mapping of label to id
        cname2cid (dict): the mapping of category name to id

    Returns:
        list of loaded records whose structure is:
        {
            'im_file': str, # image file name
            'im_id': int, # image id
            'h': int, # height of image
            'w': int, # width of image
            'is_crowd': bool,
            'gt_class': list of np.ndarray, # classids info
            'gt_bbox': list of np.ndarray, # bounding box info
            'gt_poly': list of int, # poly info
        }

    """

    if fname.endswith('.roidb'):
        records, cname2cid = load_roidb(fname, samples)
    elif fname.endswith('.json'):
        from . import coco_loader
        records, cname2cid = coco_loader.load(fname, samples, with_background)
    elif os.path.isfile(fname):
        from . import voc_loader
        if use_default_label is None or cname2cid is not None:
            records, cname2cid = voc_loader.get_roidb(fname, samples, cname2cid,
                                                with_background=with_background)
        else:
            records, cname2cid = voc_loader.load(fname, samples,
                                                 use_default_label,
                                                 with_background=with_background)
    else:
        raise ValueError('invalid file type when load data from file[%s]' %
                         (fname))
    check_records(records)
    if with_cat2id:
        return records, cname2cid
    else:
        return records
