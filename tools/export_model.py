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
from __future__ import division
from __future__ import print_function

import os

from paddle import fluid

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.cli import ArgsParser
import ppdet.utils.checkpoint as checkpoint

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_cpp_config(config):
    cfg_name = os.path.basename(FLAGS.config).split('.')[0]
    save_dir = os.path.join(FLAGS.output_dir, cfg_name)
    cpp_cfg = {
        'use_python_inference': False,
        'mode': 'fluid',
        'draw_threshold': 0.5,
        'metric': config['metric']
    }
    cpp_arch_list = {'YOLO': 3, 'SSD': 40, 'RCNN': 40, 'RetinaNet': 40}
    arch = config['architecture']

    for cpp_arch, min_subgraph_size in cpp_arch_list.items():
        if cpp_arch in arch:
            cpp_cfg['arch'] = cpp_arch
            cpp_cfg['min_subgraph_size'] = min_subgraph_size
            break
    preprocess_list = [{
        'type': 'Resize'
    }, {
        'type': 'Normalize'
    }, {
        'type': 'Permute'
    }]

    image_shape = config['TestReader']['inputs_def']['image_shape']
    scale_set = {'RCNN', 'RetinaNet'}
    preprocess_list[0]['target_size'] = image_shape[1]
    if cpp_cfg['arch'] in scale_set:
        preprocess_list[0]['max_size'] = image_shape[1]

    sample_transforms = config['TestReader']['sample_transforms']
    for p in preprocess_list[1:]:
        for st in sample_transforms:
            method = st.__class__.__name__
            if p['type'] in method:
                params = st.__dict__
                params.pop('_id')
                p.update(params)
                break
    batch_transforms = config['TestReader'].get('batch_transforms', None)
    if batch_transforms:
        methods = [bt.__class__.__name__ for bt in batch_transforms]
        for bt in batch_transforms:
            method = bt.__class__.__name__
            if method == 'PadBatch':
                preprocess_list.append({'type': 'PadStride'})
                params = bt.__dict__
                preprocess_list[-1].update({'stride': params['pad_to_stride']})
                break

    cpp_cfg['Preprocess'] = preprocess_list
    import yaml
    yaml.dump(cpp_cfg, open(os.path.join(save_dir, 'cpp_infer.yml'), 'w'))


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
    target_vars = list(test_fetches.values())
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


def main():
    cfg = load_config(FLAGS.config)

    if 'architecture' in cfg:
        main_arch = cfg.architecture
    else:
        raise ValueError("'architecture' not specified in config file.")

    merge_config(FLAGS.opt)

    # Use CPU for exporting inference model instead of GPU
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    model = create(main_arch)

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs_def = cfg['TestReader']['inputs_def']
            inputs_def['use_dataloader'] = False
            feed_vars, _ = model.build_inputs(**inputs_def)
            test_fetches = model.test(feed_vars)
    infer_prog = infer_prog.clone(True)

    exe.run(startup_prog)
    checkpoint.load_params(exe, infer_prog, cfg.weights)

    save_infer_model(FLAGS, exe, feed_vars, test_fetches, infer_prog)
    parse_cpp_config(cfg)


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output model files.")
    FLAGS = parser.parse_args()
    main()
