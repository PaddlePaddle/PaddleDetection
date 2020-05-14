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


def multiscale_def(image_shape, num_scale, use_flip=True):
    base_name_list = ['image']
    multiscale_def = {}
    ms_def_names = []
    if use_flip:
        num_scale //= 2
        base_name_list.append('image_flip')
        multiscale_def['image_flip'] = {
            'shape': [None] + image_shape,
            'dtype': 'float32',
            'lod_level': 0
        }
        multiscale_def['im_info_image_flip'] = {
            'shape': [None, 3],
            'dtype': 'float32',
            'lod_level': 0
        }
        ms_def_names.append('image_flip')
        ms_def_names.append('im_info_image_flip')
    for base_name in base_name_list:
        for i in range(0, num_scale - 1):
            name = base_name + '_scale_' + str(i)
            multiscale_def[name] = {
                'shape': [None] + image_shape,
                'dtype': 'float32',
                'lod_level': 0
            }
            im_info_name = 'im_info_' + name
            multiscale_def[im_info_name] = {
                'shape': [None, 3],
                'dtype': 'float32',
                'lod_level': 0
            }
            ms_def_names.append(name)
            ms_def_names.append(im_info_name)
    return multiscale_def, ms_def_names
