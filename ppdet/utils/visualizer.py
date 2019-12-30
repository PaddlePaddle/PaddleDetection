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
from __future__ import unicode_literals

import numpy as np
from PIL import Image, ImageDraw

from .colormap import colormap

__all__ = ['visualize_results']


def visualize_results(image,
                      im_id,
                      catid2name,
                      threshold=0.5,
                      bbox_results=None,
                      mask_results=None):
    """
    Visualize bbox and mask results
    """
    if mask_results:
        image = draw_mask(image, im_id, mask_results, threshold)
    if bbox_results:
        image = draw_bbox(image, im_id, catid2name, bbox_results, threshold)
    return image


def draw_mask(image, im_id, segms, threshold, alpha=0.7):
    """
    Draw mask on image
    """
    mask_color_id = 0
    w_ratio = .4
    color_list = colormap(rgb=True)
    img_array = np.array(image).astype('float32')
    for dt in np.array(segms):
        if im_id != dt['image_id']:
            continue
        segm, score = dt['segmentation'], dt['score']
        if score < threshold:
            continue
        import pycocotools.mask as mask_util
        mask = mask_util.decode(segm) * 255
        color_mask = color_list[mask_color_id % len(color_list), 0:3]
        mask_color_id += 1
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        img_array[idx[0], idx[1], :] *= 1.0 - alpha
        img_array[idx[0], idx[1], :] += alpha * color_mask
    return Image.fromarray(img_array.astype('uint8'))


def draw_bbox(image, im_id, catid2name, bboxes, threshold):
    """
    Draw bbox on image
    """
    draw = ImageDraw.Draw(image)

    catid2color = {}
    color_list = colormap(rgb=True)[:40]
    for dt in np.array(bboxes):
        if im_id != dt['image_id']:
            continue
        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        if score < threshold:
            continue

        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h

        if catid not in catid2color:
            idx = np.random.randint(len(color_list))
            catid2color[catid] = color_list[idx]
        color = tuple(catid2color[catid])

        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=2,
            fill=color)

        # draw label
        text = "{} {:.2f}".format(catid2name[catid], score)
        tw, th = draw.textsize(text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

    return image
