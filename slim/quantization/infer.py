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

import os, sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import glob
import numpy as np
from PIL import Image

import paddle
from paddle import fluid

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.eval_utils import parse_fetches
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu, check_version, check_config, enable_static_mode
from ppdet.utils.visualizer import visualize_results
import ppdet.utils.checkpoint as checkpoint

from ppdet.data.reader import create_reader
from tools.infer import get_test_images, get_save_image_name
import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
from paddleslim.quant import quant_aware, convert


def main():
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()

    main_arch = cfg.architecture

    dataset = cfg.TestReader['dataset']

    test_images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
    dataset.set_images(test_images)

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    model = create(main_arch)

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs_def = cfg['TestReader']['inputs_def']
            feed_vars, loader = model.build_inputs(**inputs_def)
            test_fetches = model.test(feed_vars)
    infer_prog = infer_prog.clone(True)

    reader = create_reader(cfg.TestReader)
    # When iterable mode, set set_sample_list_generator(reader, place)
    loader.set_sample_list_generator(reader)
    not_quant_pattern = []
    if FLAGS.not_quant_pattern:
        not_quant_pattern = FLAGS.not_quant_pattern
    config = {
        'weight_quantize_type': 'channel_wise_abs_max',
        'activation_quantize_type': 'moving_average_abs_max',
        'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d'],
        'not_quant_pattern': not_quant_pattern
    }

    infer_prog = quant_aware(infer_prog, place, config, for_test=True)

    exe.run(startup_prog)

    if cfg.weights:
        checkpoint.load_params(exe, infer_prog, cfg.weights)
    infer_prog = convert(infer_prog, place, config, save_int8=False)

    # parse infer fetches
    assert cfg.metric in ['COCO', 'VOC', 'OID', 'WIDERFACE'], \
            "unknown metric type {}".format(cfg.metric)
    extra_keys = []
    if cfg['metric'] in ['COCO', 'OID']:
        extra_keys = ['im_info', 'im_id', 'im_shape']
    if cfg['metric'] == 'VOC' or cfg['metric'] == 'WIDERFACE':
        extra_keys = ['im_id', 'im_shape']
    keys, values, _ = parse_fetches(test_fetches, infer_prog, extra_keys)

    # parse dataset category
    if cfg.metric == 'COCO':
        from ppdet.utils.coco_eval import bbox2out, mask2out, get_category_info
    if cfg.metric == 'OID':
        from ppdet.utils.oid_eval import bbox2out, get_category_info
    if cfg.metric == "VOC":
        from ppdet.utils.voc_eval import bbox2out, get_category_info
    if cfg.metric == "WIDERFACE":
        from ppdet.utils.widerface_eval_utils import bbox2out, get_category_info

    anno_file = dataset.get_anno()
    with_background = dataset.with_background
    use_default_label = dataset.use_default_label

    clsid2catid, catid2name = get_category_info(anno_file, with_background,
                                                use_default_label)

    # whether output bbox is normalized in model output layer
    is_bbox_normalized = False
    if hasattr(model, 'is_bbox_normalized') and \
            callable(model.is_bbox_normalized):
        is_bbox_normalized = model.is_bbox_normalized()

    imid2path = dataset.get_imid2path()
    iter_id = 0
    try:
        loader.start()
        while True:
            outs = exe.run(infer_prog, fetch_list=values, return_numpy=False)
            res = {
                k: (np.array(v), v.recursive_sequence_lengths())
                for k, v in zip(keys, outs)
            }
            logger.info('Infer iter {}'.format(iter_id))
            iter_id += 1
            bbox_results = None
            mask_results = None
            if 'bbox' in res:
                bbox_results = bbox2out([res], clsid2catid, is_bbox_normalized)
            if 'mask' in res:
                mask_results = mask2out([res], clsid2catid,
                                        model.mask_head.resolution)

            # visualize result
            im_ids = res['im_id'][0]
            for im_id in im_ids:
                image_path = imid2path[int(im_id)]
                image = Image.open(image_path).convert('RGB')

                image = visualize_results(image,
                                          int(im_id), catid2name,
                                          FLAGS.draw_threshold, bbox_results,
                                          mask_results)

                save_name = get_save_image_name(FLAGS.output_dir, image_path)
                logger.info("Detection bbox results save in {}".format(
                    save_name))
                image.save(save_name, quality=95)
    except (StopIteration, fluid.core.EOFException):
        loader.reset()


if __name__ == '__main__':
    enable_static_mode()
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--not_quant_pattern",
        nargs='+',
        type=str,
        help="Layers which name_scope contains string in not_quant_pattern will not be quantized"
    )

    FLAGS = parser.parse_args()
    main()
