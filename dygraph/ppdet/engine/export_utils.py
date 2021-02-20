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

import os
import yaml
from collections import OrderedDict

from ppdet.metrics import get_categories

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

# Global dictionary
TRT_MIN_SUBGRAPH = {
    'YOLO': 3,
    'SSD': 40,
    'RCNN': 40,
    'RetinaNet': 40,
    'EfficientDet': 40,
    'Face': 3,
    'TTFNet': 3,
    'FCOS': 16,
    'SOLOv2': 60,
}


def _parse_reader(reader_cfg, dataset_cfg, metric, arch, image_shape):
    preprocess_list = []

    anno_file = dataset_cfg.get_anno()

    clsid2catid, catid2name = get_categories(metric, anno_file)

    label_list = [str(cat) for cat in catid2name.values()]

    sample_transforms = reader_cfg['sample_transforms']
    for st in sample_transforms[1:]:
        for key, value in st.items():
            p = {'type': key}
            p.update(value)
            preprocess_list.append(p)
    batch_transforms = reader_cfg.get('batch_transforms', None)
    if batch_transforms:
        methods = [list(bt.keys())[0] for bt in batch_transforms]
        for bt in batch_transforms:
            for key, value in bt.items():
                # for deploy/infer, use PadStride(stride) instead PadBatch(pad_to_stride, pad_gt)
                if key == 'PadBatch':
                    preprocess_list.append({
                        'type': 'PadStride',
                        'stride': value['pad_to_stride']
                    })
                    break

    return preprocess_list, label_list, image_shape


def _dump_infer_config(config, path, image_shape, model):
    arch_state = False
    from ppdet.core.config.yaml_helpers import setup_orderdict
    setup_orderdict()
    infer_cfg = OrderedDict({
        'mode': 'fluid',
        'draw_threshold': 0.5,
        'metric': config['metric'],
        'image_shape': image_shape
    })
    infer_arch = config['architecture']

    for arch, min_subgraph_size in TRT_MIN_SUBGRAPH.items():
        if arch in infer_arch:
            infer_cfg['arch'] = arch
            infer_cfg['min_subgraph_size'] = min_subgraph_size
            arch_state = True
            break
    if not arch_state:
        logger.error(
            'Architecture: {} is not supported for exporting model now'.format(
                infer_arch))
        os._exit(0)
    if 'Mask' in infer_arch:
        infer_cfg['mask'] = True
    infer_cfg['Preprocess'], infer_cfg[
        'label_list'], image_shape = _parse_reader(
            config['TestReader'], config['TestDataset'], config['metric'],
            infer_cfg['arch'], image_shape)

    yaml.dump(infer_cfg, open(path, 'w'))
    logger.info("Export inference config file to {}".format(os.path.join(path)))
    return image_shape
