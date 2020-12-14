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
import os, sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

# ignore numba warning
import warnings
warnings.filterwarnings('ignore')
import glob
import numpy as np
from PIL import Image
import paddle
from paddle.distributed import ParallelEnv
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.visualizer import visualize_results
from ppdet.utils.cli import ArgsParser
from ppdet.utils.checkpoint import load_weight
from ppdet.utils.eval_utils import get_infer_results
import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


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
        "--use_vdl",
        type=bool,
        default=False,
        help="whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/image",
        help='VisualDL logging directory for image.')
    args = parser.parse_args()
    return args


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


def run(FLAGS, cfg, place):

    # Model
    main_arch = cfg.architecture
    model = create(cfg.architecture)

    # data
    dataset = cfg.TestDataset
    test_images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
    dataset.set_images(test_images)
    test_loader, _ = create('TestReader')(dataset, cfg['worker_num'], place)

    # TODO: support other metrics
    imid2path = dataset.get_imid2path()

    from ppdet.utils.coco_eval import get_category_info
    anno_file = dataset.get_anno()
    with_background = cfg.with_background
    use_default_label = dataset.use_default_label
    clsid2catid, catid2name = get_category_info(anno_file, with_background,
                                                use_default_label)

    # Init Model
    load_weight(model, cfg.weights)

    # Run Infer 
    for iter_id, data in enumerate(test_loader):
        # forward
        fields = cfg.TestReader['inputs_def']['fields']
        model.eval()
        outs = model(
            data=data,
            input_def=cfg.TestReader['inputs_def']['fields'],
            mode='infer')
        for key, value in outs.items():
            outs[key] = value.numpy()
        im_shape = data[fields.index('im_shape')].numpy()
        scale_factor = data[fields.index('scale_factor')].numpy()
        im_ids = data[fields.index('im_id')].numpy()
        im_info = [im_shape, scale_factor, im_ids]

        if 'mask' in outs and 'bbox' in outs:
            mask_resolution = model.mask_post_process.mask_resolution
            from ppdet.py_op.post_process import mask_post_process
            outs['mask'] = mask_post_process(outs, im_shape, scale_factor,
                                             mask_resolution)

        eval_type = []
        if 'bbox' in outs:
            eval_type.append('bbox')
        if 'mask' in outs:
            eval_type.append('mask')

        batch_res = get_infer_results([outs], eval_type, clsid2catid, [im_info])
        logger.info('Infer iter {}'.format(iter_id))
        bbox_res = None
        mask_res = None

        bbox_num = outs['bbox_num']
        start = 0
        for i, im_id in enumerate(im_ids):
            image_path = imid2path[int(im_id)]
            image = Image.open(image_path).convert('RGB')
            end = start + bbox_num[i]

            # use VisualDL to log original image
            if FLAGS.use_vdl:
                original_image_np = np.array(image)
                vdl_writer.add_image(
                    "original/frame_{}".format(vdl_image_frame),
                    original_image_np, vdl_image_step)

            if 'bbox' in batch_res:
                bbox_res = batch_res['bbox'][start:end]
            if 'mask' in batch_res:
                mask_res = batch_res['mask'][start:end]

            image = visualize_results(image, bbox_res, mask_res,
                                      int(im_id), catid2name,
                                      FLAGS.draw_threshold)

            # use VisualDL to log image with bbox
            if FLAGS.use_vdl:
                infer_image_np = np.array(image)
                vdl_writer.add_image("bbox/frame_{}".format(vdl_image_frame),
                                     infer_image_np, vdl_image_step)
                vdl_image_step += 1
                if vdl_image_step % 10 == 0:
                    vdl_image_step = 0
                    vdl_image_frame += 1

            # save image with detection
            save_name = get_save_image_name(FLAGS.output_dir, image_path)
            logger.info("Detection bbox results save in {}".format(save_name))
            image.save(save_name, quality=95)
            start = end


def main():
    FLAGS = parse_args()

    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    place = 'gpu:{}'.format(ParallelEnv().dev_id) if cfg.use_gpu else 'cpu'
    place = paddle.set_device(place)
    run(FLAGS, cfg, place)


if __name__ == '__main__':
    main()
