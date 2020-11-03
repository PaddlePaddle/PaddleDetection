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
import numpy as np
from collections import OrderedDict

import logging
logger = logging.getLogger(__name__)

import paddle.fluid as fluid

__all__ = ['dump_infer_config', 'save_infer_model']

# Global dictionary
TRT_MIN_SUBGRAPH = {
    'YOLO': 3,
    'SSD': 3,
    'RCNN': 40,
    'RetinaNet': 40,
    'EfficientDet': 40,
    'Face': 3,
    'TTFNet': 3,
    'FCOS': 3,
    'SOLOv2': 60,
}
RESIZE_SCALE_SET = {
    'RCNN',
    'RetinaNet',
    'FCOS',
    'SOLOv2',
}


def parse_reader(reader_cfg, metric, arch):
    preprocess_list = []

    image_shape = reader_cfg['inputs_def'].get('image_shape', [3, None, None])
    has_shape_def = not None in image_shape

    dataset = reader_cfg['dataset']
    anno_file = dataset.get_anno()
    with_background = dataset.with_background
    use_default_label = dataset.use_default_label

    if metric == 'COCO':
        from ppdet.utils.coco_eval import get_category_info
    elif metric == "VOC":
        from ppdet.utils.voc_eval import get_category_info
    elif metric == "WIDERFACE":
        from ppdet.utils.widerface_eval_utils import get_category_info
    else:
        raise ValueError(
            "metric only supports COCO, VOC, WIDERFACE, but received {}".format(
                metric))
    clsid2catid, catid2name = get_category_info(anno_file, with_background,
                                                use_default_label)

    label_list = [str(cat) for cat in catid2name.values()]

    sample_transforms = reader_cfg['sample_transforms']
    for st in sample_transforms[1:]:
        method = st.__class__.__name__
        p = {'type': method.replace('Image', '')}
        params = st.__dict__
        params.pop('_id')
        if p['type'] == 'Resize' and has_shape_def:
            params['target_size'] = min(image_shape[
                1:]) if arch in RESIZE_SCALE_SET else image_shape[1]
            params['max_size'] = max(image_shape[
                1:]) if arch in RESIZE_SCALE_SET else 0
            params['image_shape'] = image_shape[1:]
            if 'target_dim' in params:
                params.pop('target_dim')
        if p['type'] == 'ResizeAndPad':
            assert has_shape_def, "missing input shape"
            p['type'] = 'Resize'
            p['target_size'] = params['target_dim']
            p['max_size'] = params['target_dim']
            p['interp'] = params['interp']
            p['image_shape'] = image_shape[1:]
            preprocess_list.append(p)
            continue
        p.update(params)
        preprocess_list.append(p)
    batch_transforms = reader_cfg.get('batch_transforms', None)
    if batch_transforms:
        methods = [bt.__class__.__name__ for bt in batch_transforms]
        for bt in batch_transforms:
            method = bt.__class__.__name__
            if method == 'PadBatch':
                preprocess_list.append({'type': 'PadStride'})
                params = bt.__dict__
                preprocess_list[-1].update({'stride': params['pad_to_stride']})
                break

    return with_background, preprocess_list, label_list


def dump_infer_config(FLAGS, config):
    arch_state = 0
    cfg_name = os.path.basename(FLAGS.config).split('.')[0]
    save_dir = os.path.join(FLAGS.output_dir, cfg_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    from ppdet.core.config.yaml_helpers import setup_orderdict
    setup_orderdict()
    infer_cfg = OrderedDict({
        'use_python_inference': False,
        'mode': 'fluid',
        'draw_threshold': 0.5,
        'metric': config['metric']
    })
    infer_arch = config['architecture']

    for arch, min_subgraph_size in TRT_MIN_SUBGRAPH.items():
        if arch in infer_arch:
            infer_cfg['arch'] = arch
            infer_cfg['min_subgraph_size'] = min_subgraph_size
            arch_state = 1
            break
    if not arch_state:
        logger.error(
            'Architecture: {} is not supported for exporting model now'.format(
                infer_arch))
        os._exit(0)

    if 'Mask' in config['architecture']:
        infer_cfg['mask_resolution'] = config['MaskHead']['resolution']
    infer_cfg['with_background'], infer_cfg['Preprocess'], infer_cfg[
        'label_list'] = parse_reader(config['TestReader'], config['metric'],
                                     infer_cfg['arch'])

    yaml.dump(infer_cfg, open(os.path.join(save_dir, 'infer_cfg.yml'), 'w'))
    logger.info("Export inference config file to {}".format(
        os.path.join(save_dir, 'infer_cfg.yml')))


def prune_feed_vars(feeded_var_names, target_vars, prog):
    """
    Filter out feed variables which are not in program,
    pruned feed variables are only used in post processing
    on model output, which are not used in program, such
    as im_id to identify image order, im_shape to clip bbox
    in image.
    """
    exist_var_names = []
    prog = prog.clone()
    prog = prog._prune(targets=target_vars)
    global_block = prog.global_block()
    for name in feeded_var_names:
        try:
            v = global_block.var(name)
            exist_var_names.append(str(v.name))
        except Exception:
            logger.info('save_inference_model pruned unused feed '
                        'variables {}'.format(name))
            pass
    return exist_var_names


def save_infer_model(FLAGS, exe, feed_vars, test_fetches, infer_prog):
    cfg_name = os.path.basename(FLAGS.config).split('.')[0]
    save_dir = os.path.join(FLAGS.output_dir, cfg_name)
    feed_var_names = [var.name for var in feed_vars.values()]
    fetch_list = sorted(test_fetches.items(), key=lambda i: i[0])
    target_vars = [var[1] for var in fetch_list]
    feed_var_names = prune_feed_vars(feed_var_names, target_vars, infer_prog)
    logger.info("Export inference model to {}, input: {}, output: "
                "{}...".format(save_dir, feed_var_names,
                               [str(var.name) for var in target_vars]))
    fluid.io.save_inference_model(
        save_dir,
        feeded_var_names=feed_var_names,
        target_vars=target_vars,
        executor=exe,
        main_program=infer_prog,
        params_filename="__params__")
