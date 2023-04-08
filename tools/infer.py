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
import sys

# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import glob
import ast

import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_mlu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.slim import build_slim_model

from ppdet.utils.logger import setup_logger
logger = setup_logger('train')


def parse_args():
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
        "--slim_config",
        default=None,
        type=str,
        help="Configuration file of slim method.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="Whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/image",
        help='VisualDL logging directory for image.')
    parser.add_argument(
        "--save_results",
        type=bool,
        default=False,
        help="Whether to save inference results to output_dir.")
    parser.add_argument(
        "--slice_infer",
        action='store_true',
        help="Whether to slice the image and merge the inference results for small object detection."
    )
    parser.add_argument(
        '--slice_size',
        nargs='+',
        type=int,
        default=[640, 640],
        help="Height of the sliced image.")
    parser.add_argument(
        "--overlap_ratio",
        nargs='+',
        type=float,
        default=[0.25, 0.25],
        help="Overlap height ratio of the sliced image.")
    parser.add_argument(
        "--combine_method",
        type=str,
        default='nms',
        help="Combine method of the sliced images' detection results, choose in ['nms', 'nmm', 'concat']."
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.6,
        help="Combine method matching threshold.")
    parser.add_argument(
        "--match_metric",
        type=str,
        default='ios',
        help="Combine method matching metric, choose in ['iou', 'ios'].")
    parser.add_argument(
        "--visualize",
        type=ast.literal_eval,
        default=True,
        help="Whether to save visualize results to output_dir.")
    args = parser.parse_args()
    return args


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def run(FLAGS, cfg):
    # build trainer
    trainer = Trainer(cfg, mode='test')

    # load weights
    trainer.load_weights(cfg.weights)

    # get inference images
    images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)

    # inference
    if FLAGS.slice_infer:
        trainer.slice_predict(
            images,
            slice_size=FLAGS.slice_size,
            overlap_ratio=FLAGS.overlap_ratio,
            combine_method=FLAGS.combine_method,
            match_threshold=FLAGS.match_threshold,
            match_metric=FLAGS.match_metric,
            draw_threshold=FLAGS.draw_threshold,
            output_dir=FLAGS.output_dir,
            save_results=FLAGS.save_results,
            visualize=FLAGS.visualize)
    else:
        trainer.predict(
            images,
            draw_threshold=FLAGS.draw_threshold,
            output_dir=FLAGS.output_dir,
            save_results=FLAGS.save_results,
            visualize=FLAGS.visualize)


def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)

    # disable npu in config by default
    if 'use_npu' not in cfg:
        cfg.use_npu = False

    # disable xpu in config by default
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False

    if 'use_gpu' not in cfg:
        cfg.use_gpu = False

    # disable mlu in config by default
    if 'use_mlu' not in cfg:
        cfg.use_mlu = False

    if cfg.use_gpu:
        place = paddle.set_device('gpu')
    elif cfg.use_npu:
        place = paddle.set_device('npu')
    elif cfg.use_xpu:
        place = paddle.set_device('xpu')
    elif cfg.use_mlu:
        place = paddle.set_device('mlu')
    else:
        place = paddle.set_device('cpu')

    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_npu(cfg.use_npu)
    check_xpu(cfg.use_xpu)
    check_mlu(cfg.use_mlu)
    check_version()

    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
