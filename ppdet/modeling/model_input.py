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
from __future__ import print_function
from __future__ import division

from collections import OrderedDict
from ppdet.data.transform.operators import *

from paddle import fluid

__all__ = ['create_feed']

# yapf: disable
feed_var_def = [
    {'name': 'im_info',       'shape': [3],  'dtype': 'float32', 'lod_level': 0},
    {'name': 'im_id',         'shape': [1],  'dtype': 'int32',   'lod_level': 0},
    {'name': 'gt_box',        'shape': [4],  'dtype': 'float32', 'lod_level': 1},
    {'name': 'gt_label',      'shape': [1],  'dtype': 'int32',   'lod_level': 1},
    {'name': 'is_crowd',      'shape': [1],  'dtype': 'int32',   'lod_level': 1},
    {'name': 'gt_mask',       'shape': [2],  'dtype': 'float32', 'lod_level': 3},
    {'name': 'is_difficult',  'shape': [1],  'dtype': 'int32',   'lod_level': 1},
    {'name': 'gt_score',      'shape': [1],  'dtype': 'float32', 'lod_level': 0},
    {'name': 'im_shape',      'shape': [3],  'dtype': 'float32', 'lod_level': 0},
    {'name': 'im_size',       'shape': [2],  'dtype': 'int32',   'lod_level': 0},
]
# yapf: enable


def create_feed(feed, use_pyreader=True, sub_prog_feed=False):
    image_shape = feed.image_shape
    feed_var_map = {var['name']: var for var in feed_var_def}
    feed_var_map['image'] = {
        'name': 'image',
        'shape': image_shape,
        'dtype': 'float32',
        'lod_level': 0
    }

    # tensor padding with 0 is used instead of LoD tensor when 
    # num_max_boxes is set
    if getattr(feed, 'num_max_boxes', None) is not None:
        feed_var_map['gt_label']['shape'] = [feed.num_max_boxes]
        feed_var_map['gt_score']['shape'] = [feed.num_max_boxes]
        feed_var_map['gt_box']['shape'] = [feed.num_max_boxes, 4]
        feed_var_map['is_difficult']['shape'] = [feed.num_max_boxes]
        feed_var_map['gt_label']['lod_level'] = 0
        feed_var_map['gt_score']['lod_level'] = 0
        feed_var_map['gt_box']['lod_level'] = 0
        feed_var_map['is_difficult']['lod_level'] = 0

    base_name_list = ['image']
    num_scale = getattr(feed, 'num_scale', 1)
    sample_transform = feed.sample_transforms
    multiscale_test = False
    aug_flip = False
    for t in sample_transform:
        if isinstance(t, MultiscaleTestResize):
            multiscale_test = True
            aug_flip = t.use_flip
            assert (len(t.target_size)+1)*(aug_flip+1) == num_scale, \
                "num_scale: {} is not equal to the actual number of scale: {}."\
                .format(num_scale, (len(t.target_size)+1)*(aug_flip+1))
            break

    if aug_flip:
        num_scale //= 2
        base_name_list.insert(0, 'flip_image')
        feed_var_map['flip_image'] = {
            'name': 'flip_image',
            'shape': image_shape,
            'dtype': 'float32',
            'lod_level': 0
        }

    image_name_list = []
    if multiscale_test:
        for base_name in base_name_list:
            for i in range(0, num_scale):
                name = base_name if i == 0 else base_name + '_scale_' + str(i -
                                                                            1)
                feed_var_map[name] = {
                    'name': name,
                    'shape': image_shape,
                    'dtype': 'float32',
                    'lod_level': 0
                }
                image_name_list.append(name)
        feed_var_map['im_info']['shape'] = [feed.num_scale * 3]
        feed.fields = image_name_list + feed.fields[1:]
    if sub_prog_feed:
        box_names = ['bbox', 'bbox_flip']
        for box_name in box_names:
            sub_prog_feed = {
                'name': box_name,
                'shape': [6],
                'dtype': 'float32',
                'lod_level': 1
            }

            feed.fields = feed.fields + [box_name]
            feed_var_map[box_name] = sub_prog_feed

    feed_vars = OrderedDict([(key, fluid.layers.data(
        name=feed_var_map[key]['name'],
        shape=feed_var_map[key]['shape'],
        dtype=feed_var_map[key]['dtype'],
        lod_level=feed_var_map[key]['lod_level'])) for key in feed.fields])

    pyreader = None
    if use_pyreader:
        pyreader = fluid.io.PyReader(
            feed_list=list(feed_vars.values()),
            capacity=64,
            use_double_buffer=True,
            iterable=False)
    return pyreader, feed_vars
