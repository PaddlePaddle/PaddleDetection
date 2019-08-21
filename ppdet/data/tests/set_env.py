#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import os
import six
import logging

import matplotlib
matplotlib.use('Agg', force=False)

prefix = os.path.dirname(os.path.abspath(__file__))

#coco data for testing
if six.PY3:
    version = 'python3'
else:
    version = 'python2'

data_root = os.path.join(prefix, 'data/coco.test.%s' % (version))

# coco data for testing
coco_data = {
    'TRAIN': {
        'ANNO_FILE': os.path.join(data_root, 'train2017.roidb'),
        'IMAGE_DIR': os.path.join(data_root, 'train2017')
    },
    'VAL': {
        'ANNO_FILE': os.path.join(data_root, 'val2017.roidb'),
        'IMAGE_DIR': os.path.join(data_root, 'val2017')
    }
}

script = os.path.join(os.path.dirname(__file__), 'data/prepare_data.sh')

if not os.path.exists(data_root):
    ret = os.system('bash %s %s' % (script, version))
    if ret != 0:
        logging.error('not found file[%s], you should manually prepare '
                      'your data using "data/prepare_data.sh"' % (data_root))
        sys.exit(1)
