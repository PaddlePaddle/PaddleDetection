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
import cv2

from .colormap import colormap

__all__ = ['visualize_results']


def visualize_results(image,
                      bbox_res,
                      mask_res,
                      segm_res,
                      im_id,
                      catid2name,
                      threshold=0.5):
    """
    Visualize bbox and mask results
    """
    if bbox_res is not None:
        image = draw_bbox(image, im_id, catid2name, bbox_res, threshold)
    if mask_res is not None:
        image = draw_mask(image, im_id, mask_res, threshold)
    if segm_res is not None:
        image = draw_segm(image, im_id, catid2name, segm_res, threshold)
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


def draw_segm(image,
              im_id,
              catid2name,
              segms,
              threshold,
              alpha=0.7,
              draw_box=True):
    """
    Draw segmentation on image
    """
    mask_color_id = 0
    w_ratio = .4
    color_list = colormap(rgb=True)
    img_array = np.array(image).astype('float32')
    for dt in np.array(segms):
        if im_id != dt['image_id']:
            continue
        segm, score, catid = dt['segmentation'], dt['score'], dt['category_id']
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

        if not draw_box:
            center_y, center_x = ndimage.measurements.center_of_mass(mask)
            label_text = "{}".format(catid2name[catid])
            vis_pos = (max(int(center_x) - 10, 0), int(center_y))
            cv2.putText(img_array, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))
        else:
            mask = mask_util.decode(segm) * 255
            sum_x = np.sum(mask, axis=0)
            x = np.where(sum_x > 0.5)[0]
            sum_y = np.sum(mask, axis=1)
            y = np.where(sum_y > 0.5)[0]
            x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
            cv2.rectangle(img_array, (x0, y0), (x1, y1),
                          tuple(color_mask.astype('int32').tolist()), 1)
            bbox_text = '%s %.2f' % (catid2name[catid], score)
            t_size = cv2.getTextSize(bbox_text, 0, 0.3, thickness=1)[0]
            cv2.rectangle(img_array, (x0, y0), (x0 + t_size[0],
                                                y0 - t_size[1] - 3),
                          tuple(color_mask.astype('int32').tolist()), -1)
            cv2.putText(
                img_array,
                bbox_text, (x0, y0 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3, (0, 0, 0),
                1,
                lineType=cv2.LINE_AA)

    return Image.fromarray(img_array.astype('uint8'))
