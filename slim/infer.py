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
import glob
import time

import numpy as np
from PIL import Image
sys.path.append("../../")


def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be set before
# `import paddle`. Otherwise, it would not take any effect.
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

from paddle import fluid
from ppdet.utils.cli import print_total_cfg
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.modeling.model_input import create_feed
from ppdet.data.data_feed import create_reader

from ppdet.utils.eval_utils import parse_fetches
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu
from ppdet.utils.visualizer import visualize_results
import ppdet.utils.checkpoint as checkpoint

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def get_save_image_name(output_dir, image_path):
    """
    Get save image name from source image path.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_name = os.path.split(image_path)[-1]
    name, ext = os.path.splitext(image_name)
    return os.path.join(output_dir, "{}".format(name)) + ext


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
    images = []

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        images.append(infer_img)
        return images

    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.extend(glob.glob('{}/*.{}'.format(infer_dir, ext)))

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def main():
    cfg = load_config(FLAGS.config)

    if 'architecture' in cfg:
        main_arch = cfg.architecture
    else:
        raise ValueError("'architecture' not specified in config file.")

    merge_config(FLAGS.opt)

    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)
    # print_total_cfg(cfg)

    if 'test_feed' not in cfg:
        test_feed = create(main_arch + 'TestFeed')
    else:
        test_feed = create(cfg.test_feed)

    test_images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
    test_feed.dataset.add_images(test_images)

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    infer_prog, feed_var_names, fetch_list = fluid.io.load_inference_model(
        dirname=FLAGS.model_path,
        model_filename=FLAGS.model_name,
        params_filename=FLAGS.params_name,
        executor=exe)

    reader = create_reader(test_feed)
    feeder = fluid.DataFeeder(
        place=place, feed_list=feed_var_names, program=infer_prog)

    # parse infer fetches
    assert cfg.metric in ['COCO', 'VOC'], \
            "unknown metric type {}".format(cfg.metric)
    extra_keys = []
    if cfg['metric'] == 'COCO':
        extra_keys = ['im_info', 'im_id', 'im_shape']
    if cfg['metric'] == 'VOC':
        extra_keys = ['im_id', 'im_shape']
    keys, values, _ = parse_fetches({
        'bbox': fetch_list
    }, infer_prog, extra_keys)

    # parse dataset category
    if cfg.metric == 'COCO':
        from ppdet.utils.coco_eval import bbox2out, mask2out, get_category_info
    if cfg.metric == "VOC":
        from ppdet.utils.voc_eval import bbox2out, get_category_info

    anno_file = getattr(test_feed.dataset, 'annotation', None)
    with_background = getattr(test_feed, 'with_background', True)
    use_default_label = getattr(test_feed, 'use_default_label', False)
    clsid2catid, catid2name = get_category_info(anno_file, with_background,
                                                use_default_label)

    # whether output bbox is normalized in model output layer
    is_bbox_normalized = False

    # use tb-paddle to log image
    if FLAGS.use_tb:
        from tb_paddle import SummaryWriter
        tb_writer = SummaryWriter(FLAGS.tb_log_dir)
        tb_image_step = 0
        tb_image_frame = 0  # each frame can display ten pictures at most. 

    imid2path = reader.imid2path
    keys = ['bbox']
    infer_time = True
    compile_prog = fluid.compiler.CompiledProgram(infer_prog)

    for iter_id, data in enumerate(reader()):
        feed_data = [[d[0], d[1]] for d in data]
        # for infer time
        if infer_time:
            warmup_times = 10
            repeats_time = 100
            feed_data_dict = feeder.feed(feed_data)
            for i in range(warmup_times):
                exe.run(compile_prog,
                        feed=feed_data_dict,
                        fetch_list=fetch_list,
                        return_numpy=False)
            start_time = time.time()
            for i in range(repeats_time):
                exe.run(compile_prog,
                        feed=feed_data_dict,
                        fetch_list=fetch_list,
                        return_numpy=False)

            print("infer time: {} ms/sample".format((time.time() - start_time) *
                                                    1000 / repeats_time))
            infer_time = False

        outs = exe.run(compile_prog,
                       feed=feeder.feed(feed_data),
                       fetch_list=fetch_list,
                       return_numpy=False)
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(keys, outs)
        }
        res['im_id'] = [[d[2] for d in data]]
        logger.info('Infer iter {}'.format(iter_id))

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

            # use tb-paddle to log original image           
            if FLAGS.use_tb:
                original_image_np = np.array(image)
                tb_writer.add_image(
                    "original/frame_{}".format(tb_image_frame),
                    original_image_np,
                    tb_image_step,
                    dataformats='HWC')

            image = visualize_results(image,
                                      int(im_id), catid2name,
                                      FLAGS.draw_threshold, bbox_results,
                                      mask_results)

            # use tb-paddle to log image with bbox
            if FLAGS.use_tb:
                infer_image_np = np.array(image)
                tb_writer.add_image(
                    "bbox/frame_{}".format(tb_image_frame),
                    infer_image_np,
                    tb_image_step,
                    dataformats='HWC')
                tb_image_step += 1
                if tb_image_step % 10 == 0:
                    tb_image_step = 0
                    tb_image_frame += 1

            save_name = get_save_image_name(FLAGS.output_dir, image_path)
            logger.info("Detection bbox results save in {}".format(save_name))
            image.save(save_name, quality=95)


if __name__ == '__main__':
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
        "--use_tb",
        type=bool,
        default=False,
        help="whether to record the data to Tensorboard.")
    parser.add_argument(
        '--tb_log_dir',
        type=str,
        default="tb_log_dir/image",
        help='Tensorboard logging directory for image.')
    parser.add_argument(
        '--model_path', type=str, default=None, help="inference model path")
    parser.add_argument(
        '--model_name',
        type=str,
        default='__model__.infer',
        help="model filename for inference model")
    parser.add_argument(
        '--params_name',
        type=str,
        default='__params__',
        help="params filename for inference model")
    FLAGS = parser.parse_args()
    main()
