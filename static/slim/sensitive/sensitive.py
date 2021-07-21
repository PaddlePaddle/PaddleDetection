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
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from paddle import fluid
from ppdet.core.workspace import load_config, merge_config, create

from ppdet.data.reader import create_reader

from ppdet.utils.eval_utils import parse_fetches, eval_run, eval_results
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_version, check_config, enable_static_mode
import ppdet.utils.checkpoint as checkpoint
from paddleslim.prune import sensitivity
import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def main():
    env = os.environ

    print("FLAGS.config: {}".format(FLAGS.config))
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    check_version()

    main_arch = cfg.architecture

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    # build program
    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            model = create(main_arch)
            inputs_def = cfg['EvalReader']['inputs_def']
            feed_vars, eval_loader = model.build_inputs(**inputs_def)
            fetches = model.eval(feed_vars)
    eval_prog = eval_prog.clone(True)
    if FLAGS.print_params:
        print(
            "-------------------------All parameters in current graph----------------------"
        )
        for block in eval_prog.blocks:
            for param in block.all_parameters():
                print("parameter name: {}\tshape: {}".format(param.name,
                                                             param.shape))
        print(
            "------------------------------------------------------------------------------"
        )
        return

    eval_reader = create_reader(cfg.EvalReader)
    # When iterable mode, set set_sample_list_generator(eval_reader, place)
    eval_loader.set_sample_list_generator(eval_reader)

    # parse eval fetches
    extra_keys = []
    if cfg.metric == 'COCO':
        extra_keys = ['im_info', 'im_id', 'im_shape']
    if cfg.metric == 'VOC':
        extra_keys = ['gt_bbox', 'gt_class', 'is_difficult']
    if cfg.metric == 'WIDERFACE':
        extra_keys = ['im_id', 'im_shape', 'gt_box']
    eval_keys, eval_values, eval_cls = parse_fetches(fetches, eval_prog,
                                                     extra_keys)

    exe.run(startup_prog)

    fuse_bn = getattr(model.backbone, 'norm_type', None) == 'affine_channel'

    ignore_params = cfg.finetune_exclude_pretrained_params \
                 if 'finetune_exclude_pretrained_params' in cfg else []

    start_iter = 0

    if cfg.weights:
        checkpoint.load_params(exe, eval_prog, cfg.weights)
    else:
        logger.warning("Please set cfg.weights to load trained model.")

    # whether output bbox is normalized in model output layer
    is_bbox_normalized = False
    if hasattr(model, 'is_bbox_normalized') and \
            callable(model.is_bbox_normalized):
        is_bbox_normalized = model.is_bbox_normalized()

    # if map_type not set, use default 11point, only use in VOC eval
    map_type = cfg.map_type if 'map_type' in cfg else '11point'

    def test(program):

        compiled_eval_prog = fluid.CompiledProgram(program)

        results = eval_run(
            exe,
            compiled_eval_prog,
            eval_loader,
            eval_keys,
            eval_values,
            eval_cls,
            cfg=cfg)
        resolution = None
        if 'mask' in results[0]:
            resolution = model.mask_head.resolution
        dataset = cfg['EvalReader']['dataset']
        box_ap_stats = eval_results(
            results,
            cfg.metric,
            cfg.num_classes,
            resolution,
            is_bbox_normalized,
            FLAGS.output_eval,
            map_type,
            dataset=dataset)
        return box_ap_stats[0]

    pruned_params = FLAGS.pruned_params

    assert (
        FLAGS.pruned_params is not None
    ), "FLAGS.pruned_params is empty!!! Please set it by '--pruned_params' option."
    pruned_params = FLAGS.pruned_params.strip().split(",")
    logger.info("pruned params: {}".format(pruned_params))
    pruned_ratios = [float(n) for n in FLAGS.pruned_ratios.strip().split(" ")]
    logger.info("pruned ratios: {}".format(pruned_ratios))
    sensitivity(
        eval_prog,
        place,
        pruned_params,
        test,
        sensitivities_file=FLAGS.sensitivities_file,
        pruned_ratios=pruned_ratios)


if __name__ == '__main__':
    enable_static_mode()
    parser = ArgsParser()
    parser.add_argument(
        "--output_eval",
        default=None,
        type=str,
        help="Evaluation directory, default is current directory.")
    parser.add_argument(
        "-d",
        "--dataset_dir",
        default=None,
        type=str,
        help="Dataset path, same as DataFeed.dataset.dataset_dir")
    parser.add_argument(
        "-s",
        "--sensitivities_file",
        default="sensitivities.data",
        type=str,
        help="The file used to save sensitivities.")
    parser.add_argument(
        "-p",
        "--pruned_params",
        default=None,
        type=str,
        help="The parameters to be pruned when calculating sensitivities.")
    parser.add_argument(
        "-r",
        "--pruned_ratios",
        default="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9",
        type=str,
        help="The ratios pruned iteratively for each parameter when calculating sensitivities."
    )
    parser.add_argument(
        "-P",
        "--print_params",
        default=False,
        action='store_true',
        help="Whether to only print the parameters' names and shapes.")
    FLAGS = parser.parse_args()
    main()
