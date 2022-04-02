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
import sys

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import paddle.fluid as fluid

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

try:
    from ppdet.utils.eval_utils import parse_fetches, eval_run, eval_results, json_eval_results
    import ppdet.utils.checkpoint as checkpoint
    from ppdet.utils.check import check_gpu, check_xpu, check_npu, check_version, check_config, enable_static_mode

    from ppdet.data.reader import create_reader

    from ppdet.core.workspace import load_config, merge_config, create
    from ppdet.utils.cli import ArgsParser
except ImportError as e:
    if sys.argv[0].find('static') >= 0:
        logger.error("Importing ppdet failed when running static model "
                     "with error: {}\n"
                     "please try:\n"
                     "\t1. run static model under PaddleDetection/static "
                     "directory\n"
                     "\t2. run 'pip uninstall ppdet' to uninstall ppdet "
                     "dynamic version firstly.".format(e))
        sys.exit(-1)
    else:
        raise e


def main():
    """
    Main evaluate function
    """
    env = os.environ
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)
    # disable npu in config by default and check use_npu
    if 'use_npu' not in cfg:
        cfg.use_npu = False
    check_npu(cfg.use_npu)
    use_xpu = False
    if hasattr(cfg, 'use_xpu'):
        check_xpu(cfg.use_xpu)
        use_xpu = cfg.use_xpu
    # check if paddlepaddle version is satisfied
    check_version()

    assert not (use_xpu and cfg.use_gpu), \
            'Can not run on both XPU and GPU'

    assert not (cfg.use_npu and cfg.use_gpu), \
            'Can not run on both NPU and GPU'

    main_arch = cfg.architecture

    multi_scale_test = getattr(cfg, 'MultiScaleTEST', None)

    if cfg.use_gpu and 'FLAGS_selected_gpus' in env:
        device_id = int(env['FLAGS_selected_gpus'])
    elif cfg.use_npu and 'FLAGS_selected_npus' in env:
        device_id = int(env['FLAGS_selected_npus'])
    elif use_xpu and 'FLAGS_selected_xpus' in env:
        device_id = int(env['FLAGS_selected_xpus'])
    else:
        device_id = 0

    # define executor
    if cfg.use_gpu:
        place = fluid.CUDAPlace(device_id)
    elif cfg.use_npu:
        place = fluid.NPUPlace(device_id)
    elif use_xpu:
        place = fluid.XPUPlace(device_id)
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    # build program
    model = create(main_arch)
    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs_def = cfg['EvalReader']['inputs_def']
            feed_vars, loader = model.build_inputs(**inputs_def)
            if multi_scale_test is None:
                fetches = model.eval(feed_vars)
            else:
                fetches = model.eval(feed_vars, multi_scale_test)
    eval_prog = eval_prog.clone(True)

    reader = create_reader(cfg.EvalReader, devices_num=1)
    # When iterable mode, set set_sample_list_generator(reader, place)
    loader.set_sample_list_generator(reader)

    dataset = cfg['EvalReader']['dataset']

    # eval already exists json file
    if FLAGS.json_eval:
        logger.info(
            "In json_eval mode, PaddleDetection will evaluate json files in "
            "output_eval directly. And proposal.json, bbox.json and mask.json "
            "will be detected by default.")
        json_eval_results(
            cfg.metric, json_directory=FLAGS.output_eval, dataset=dataset)
        return

    compile_program = fluid.CompiledProgram(eval_prog).with_data_parallel()
    if use_xpu or cfg.use_npu:
        compile_program = eval_prog

    assert cfg.metric != 'OID', "eval process of OID dataset \
                          is not supported."

    if cfg.metric == "WIDERFACE":
        raise ValueError("metric type {} does not support in tools/eval.py, "
                         "please use tools/face_eval.py".format(cfg.metric))
    assert cfg.metric in ['COCO', 'VOC'], \
            "unknown metric type {}".format(cfg.metric)
    extra_keys = []

    if cfg.metric == 'COCO':
        extra_keys = ['im_info', 'im_id', 'im_shape']
    if cfg.metric == 'VOC':
        extra_keys = ['gt_bbox', 'gt_class', 'is_difficult']

    keys, values, cls = parse_fetches(fetches, eval_prog, extra_keys)

    # whether output bbox is normalized in model output layer
    is_bbox_normalized = False
    if hasattr(model, 'is_bbox_normalized') and \
            callable(model.is_bbox_normalized):
        is_bbox_normalized = model.is_bbox_normalized()

    sub_eval_prog = None
    sub_keys = None
    sub_values = None
    # build sub-program
    if 'Mask' in main_arch and multi_scale_test:
        sub_eval_prog = fluid.Program()
        with fluid.program_guard(sub_eval_prog, startup_prog):
            with fluid.unique_name.guard():
                inputs_def = cfg['EvalReader']['inputs_def']
                inputs_def['mask_branch'] = True
                feed_vars, eval_loader = model.build_inputs(**inputs_def)
                sub_fetches = model.eval(
                    feed_vars, multi_scale_test, mask_branch=True)
                assert cfg.metric == 'COCO'
                extra_keys = ['im_id', 'im_shape']
        sub_keys, sub_values, _ = parse_fetches(sub_fetches, sub_eval_prog,
                                                extra_keys)
        sub_eval_prog = sub_eval_prog.clone(True)

    # load model
    exe.run(startup_prog)
    if 'weights' in cfg:
        checkpoint.load_params(exe, startup_prog, cfg.weights)

    resolution = None
    if 'Mask' in cfg.architecture or cfg.architecture == 'HybridTaskCascade':
        resolution = model.mask_head.resolution
    results = eval_run(exe, compile_program, loader, keys, values, cls, cfg,
                       sub_eval_prog, sub_keys, sub_values, resolution)

    # evaluation
    # if map_type not set, use default 11point, only use in VOC eval
    map_type = cfg.map_type if 'map_type' in cfg else '11point'
    save_only = getattr(cfg, 'save_prediction_only', False)
    eval_results(
        results,
        cfg.metric,
        cfg.num_classes,
        resolution,
        is_bbox_normalized,
        FLAGS.output_eval,
        map_type,
        dataset=dataset,
        save_only=save_only)


if __name__ == '__main__':
    enable_static_mode()
    parser = ArgsParser()
    parser.add_argument(
        "--json_eval",
        action='store_true',
        default=False,
        help="Whether to re eval with already exists bbox.json or mask.json")
    parser.add_argument(
        "-f",
        "--output_eval",
        default=None,
        type=str,
        help="Evaluation file directory, default is current directory.")
    FLAGS = parser.parse_args()
    main()
