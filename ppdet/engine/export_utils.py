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

import paddle
from ppdet.data.source.category import get_categories

from ppdet.utils.logger import setup_logger
logger = setup_logger('ppdet.engine')

# Global dictionary
TRT_MIN_SUBGRAPH = {
    'YOLO': 3,
    'SSD': 60,
    'RCNN': 40,
    'RetinaNet': 40,
    'S2ANet': 80,
    'EfficientDet': 40,
    'Face': 3,
    'TTFNet': 60,
    'FCOS': 16,
    'SOLOv2': 60,
    'HigherHRNet': 3,
    'HRNet': 3,
    'DeepSORT': 3,
    'ByteTrack': 10,
    'JDE': 10,
    'FairMOT': 5,
    'GFL': 16,
    'PicoDet': 3,
    'CenterNet': 5,
    'TOOD': 5,
    'YOLOX': 8,
}

KEYPOINT_ARCH = ['HigherHRNet', 'TopDownHRNet']
MOT_ARCH = ['DeepSORT', 'JDE', 'FairMOT', 'ByteTrack']


def _prune_input_spec(input_spec, program, targets):
    # try to prune static program to figure out pruned input spec
    # so we perform following operations in static mode
    device = paddle.get_device()
    paddle.enable_static()
    paddle.set_device(device)
    pruned_input_spec = [{}]
    program = program.clone()
    program = program._prune(targets=targets)
    global_block = program.global_block()
    for name, spec in input_spec[0].items():
        try:
            v = global_block.var(name)
            pruned_input_spec[0][name] = spec
        except Exception:
            pass
    paddle.disable_static(place=device)
    return pruned_input_spec


def _parse_reader(reader_cfg, dataset_cfg, metric, arch, image_shape):
    preprocess_list = []

    anno_file = dataset_cfg.get_anno()

    clsid2catid, catid2name = get_categories(metric, anno_file, arch)

    label_list = [str(cat) for cat in catid2name.values()]

    fuse_normalize = reader_cfg.get('fuse_normalize', False)
    sample_transforms = reader_cfg['sample_transforms']
    for st in sample_transforms[1:]:
        for key, value in st.items():
            p = {'type': key}
            if key == 'Resize':
                if int(image_shape[1]) != -1:
                    value['target_size'] = image_shape[1:]
            if fuse_normalize and key == 'NormalizeImage':
                continue
            p.update(value)
            preprocess_list.append(p)
    batch_transforms = reader_cfg.get('batch_transforms', None)
    if batch_transforms:
        for bt in batch_transforms:
            for key, value in bt.items():
                # for deploy/infer, use PadStride(stride) instead PadBatch(pad_to_stride)
                if key == 'PadBatch':
                    preprocess_list.append({
                        'type': 'PadStride',
                        'stride': value['pad_to_stride']
                    })
                    break

    return preprocess_list, label_list


def _parse_tracker(tracker_cfg):
    tracker_params = {}
    for k, v in tracker_cfg.items():
        tracker_params.update({k: v})
    return tracker_params


def _dump_infer_config(config, path, image_shape, model):
    arch_state = False
    from ppdet.core.config.yaml_helpers import setup_orderdict
    setup_orderdict()
    use_dynamic_shape = True if image_shape[2] == -1 else False
    infer_cfg = OrderedDict({
        'mode': 'paddle',
        'draw_threshold': 0.5,
        'metric': config['metric'],
        'use_dynamic_shape': use_dynamic_shape
    })
    export_onnx = config.get('export_onnx', False)

    infer_arch = config['architecture']
    if 'RCNN' in infer_arch and export_onnx:
        logger.warning(
            "Exporting RCNN model to ONNX only support batch_size = 1")
        infer_cfg['export_onnx'] = True

    if infer_arch in MOT_ARCH:
        if infer_arch == 'DeepSORT':
            tracker_cfg = config['DeepSORTTracker']
        else:
            tracker_cfg = config['JDETracker']
        infer_cfg['tracker'] = _parse_tracker(tracker_cfg)

    for arch, min_subgraph_size in TRT_MIN_SUBGRAPH.items():
        if arch in infer_arch:
            infer_cfg['arch'] = arch
            infer_cfg['min_subgraph_size'] = min_subgraph_size
            arch_state = True
            break

    if infer_arch == 'YOLOX':
        infer_cfg['arch'] = infer_arch
        infer_cfg['min_subgraph_size'] = TRT_MIN_SUBGRAPH[infer_arch]
        arch_state = True

    if not arch_state:
        logger.error(
            'Architecture: {} is not supported for exporting model now.\n'.
            format(infer_arch) +
            'Please set TRT_MIN_SUBGRAPH in ppdet/engine/export_utils.py')
        os._exit(0)
    if 'mask_head' in config[config['architecture']] and config[config[
            'architecture']]['mask_head']:
        infer_cfg['mask'] = True
    label_arch = 'detection_arch'
    if infer_arch in KEYPOINT_ARCH:
        label_arch = 'keypoint_arch'

    if infer_arch in MOT_ARCH:
        label_arch = 'mot_arch'
        reader_cfg = config['TestMOTReader']
        dataset_cfg = config['TestMOTDataset']
    else:
        reader_cfg = config['TestReader']
        dataset_cfg = config['TestDataset']

    infer_cfg['Preprocess'], infer_cfg['label_list'] = _parse_reader(
        reader_cfg, dataset_cfg, config['metric'], label_arch, image_shape[1:])

    if infer_arch == 'PicoDet':
        if hasattr(config, 'export') and config['export'].get(
                'post_process',
                False) and not config['export'].get('benchmark', False):
            infer_cfg['arch'] = 'GFL'
        head_name = 'PicoHeadV2' if config['PicoHeadV2'] else 'PicoHead'
        infer_cfg['NMS'] = config[head_name]['nms']
        # In order to speed up the prediction, the threshold of nms 
        # is adjusted here, which can be changed in infer_cfg.yml
        config[head_name]['nms']["score_threshold"] = 0.3
        config[head_name]['nms']["nms_threshold"] = 0.5
        infer_cfg['fpn_stride'] = config[head_name]['fpn_stride']

    yaml.dump(infer_cfg, open(path, 'w'))
    logger.info("Export inference config file to {}".format(os.path.join(path)))
