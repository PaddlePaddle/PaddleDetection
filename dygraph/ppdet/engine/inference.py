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
import numpy as np
from PIL import Image

import paddle

from ppdet.core.workspace import create
from ppdet.utils.visualizer import visualize_results
from ppdet.utils.eval_utils import get_infer_results

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['infer_detector']


def _get_save_image_name(output_dir, image_path):
    """
    Get save image name from source image path.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_name = os.path.split(image_path)[-1]
    name, ext = os.path.splitext(image_name)
    return os.path.join(output_dir, "{}".format(name)) + ext


def infer_detector(model,
                   loader,
                   cfg,
                   draw_threshold=0.5,
                   output_dir='output',
                   use_visual_dl=False):
    extra_key = ['im_shape', 'scale_factor', 'im_id']

    # TODO: support other metrics
    imid2path = loader.dataset.get_imid2path()

    anno_file = loader.dataset.get_anno()
    with_background = cfg.with_background
    use_default_label = loader.dataset.use_default_label

    if cfg.metric == 'COCO':
        from ppdet.utils.coco_eval import get_category_info
    elif cfg.metric == 'VOC':
        from ppdet.utils.voc_eval import get_category_info
    else:
        raise ValueError("unrecongnized metric type: {}".format(cfg.metric))
    clsid2catid, catid2name = get_category_info(anno_file, with_background,
                                                use_default_label)

    # Run Infer 
    for iter_id, data in enumerate(loader):
        # forward
        model.eval()
        outs = model(data, mode='infer')
        for key in extra_key:
            outs[key] = data[key]
        for key, value in outs.items():
            outs[key] = value.numpy()

        if 'mask' in outs and 'bbox' in outs:
            mask_resolution = model.mask_post_process.mask_resolution
            from ppdet.py_op.post_process import mask_post_process
            outs['mask'] = mask_post_process(
                outs, outs['im_shape'], outs['scale_factor'], mask_resolution)

        eval_type = []
        if 'bbox' in outs:
            eval_type.append('bbox')
        if 'mask' in outs:
            eval_type.append('mask')

        batch_res = get_infer_results([outs], eval_type, clsid2catid)
        logger.info('Infer iter {}'.format(iter_id))
        bbox_res = None
        mask_res = None

        bbox_num = outs['bbox_num']
        start = 0
        for i, im_id in enumerate(outs['im_id']):
            image_path = imid2path[int(im_id)]
            image = Image.open(image_path).convert('RGB')
            end = start + bbox_num[i]

            # use VisualDL to log original image
            if use_visual_dl:
                original_image_np = np.array(image)
                vdl_writer.add_image(
                    "original/frame_{}".format(vdl_image_frame),
                    original_image_np, vdl_image_step)

            if 'bbox' in batch_res:
                bbox_res = batch_res['bbox'][start:end]
            if 'mask' in batch_res:
                mask_res = batch_res['mask'][start:end]

            image = visualize_results(image, bbox_res, mask_res,
                                      int(outs['im_id']), catid2name,
                                      draw_threshold)

            # use VisualDL to log image with bbox
            if use_visual_dl:
                infer_image_np = np.array(image)
                vdl_writer.add_image("bbox/frame_{}".format(vdl_image_frame),
                                     infer_image_np, vdl_image_step)
                vdl_image_step += 1
                if vdl_image_step % 10 == 0:
                    vdl_image_step = 0
                    vdl_image_frame += 1

            # save image with detection
            save_name = _get_save_image_name(output_dir, image_path)
            logger.info("Detection bbox results save in {}".format(save_name))
            image.save(save_name, quality=95)
            start = end
