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

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import math

from .colormap import colormap
from ppdet.utils.logger import setup_logger
from ppdet.utils.compact import imagedraw_textsize_c
from ppdet.utils.download import get_path
logger = setup_logger(__name__)

__all__ = ['visualize_results']


def visualize_results(image,
                      bbox_res,
                      mask_res,
                      segm_res,
                      keypoint_res,
                      pose3d_res,
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
    if keypoint_res is not None:
        image = draw_pose(image, keypoint_res, threshold)
    if pose3d_res is not None:
        pose3d = np.array(pose3d_res[0]['pose3d']) * 1000
        image = draw_pose3d(image, pose3d, visual_thread=threshold)
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
    font_url = "https://paddledet.bj.bcebos.com/simfang.ttf"
    font_path, _ = get_path(font_url, "~/.cache/paddle/")
    font_size = 18
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    draw = ImageDraw.Draw(image)

    catid2color = {}
    color_list = colormap(rgb=True)[:40]
    for dt in np.array(bboxes):
        if im_id != dt['image_id']:
            continue
        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        if score < threshold:
            continue

        if catid not in catid2color:
            idx = np.random.randint(len(color_list))
            catid2color[catid] = color_list[idx]
        color = tuple(catid2color[catid])

        # draw bbox
        if len(bbox) == 4:
            # draw bbox
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                 (xmin, ymin)],
                width=2,
                fill=color)
        elif len(bbox) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            draw.line(
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                width=2,
                fill=color)
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
        else:
            logger.error('the shape of bbox must be [M, 4] or [M, 8]!')

        # draw label
        text = "{} {:.2f}".format(catid2name[catid], score)
        tw, th = imagedraw_textsize_c(draw, text, font=font)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255), font=font)

    return image


def save_result(save_path, results, catid2name, threshold):
    """
    save result as txt
    """
    img_id = int(results["im_id"])
    with open(save_path, 'w') as f:
        if "bbox_res" in results:
            for dt in results["bbox_res"]:
                catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
                if score < threshold:
                    continue
                # each bbox result as a line
                # for rbox: classname score x1 y1 x2 y2 x3 y3 x4 y4
                # for bbox: classname score x1 y1 w h
                bbox_pred = '{} {} '.format(catid2name[catid],
                                            score) + ' '.join(
                                                [str(e) for e in bbox])
                f.write(bbox_pred + '\n')
        elif "keypoint_res" in results:
            for dt in results["keypoint_res"]:
                kpts = dt['keypoints']
                scores = dt['score']
                keypoint_pred = [img_id, scores, kpts]
                print(keypoint_pred, file=f)
        else:
            print("No valid results found, skip txt save")


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


def draw_pose(image,
              results,
              visual_thread=0.6,
              save_name='pose.jpg',
              save_dir='output',
              returnimg=False,
              ids=None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        plt.switch_backend('agg')
    except Exception as e:
        logger.error('Matplotlib not found, please install matplotlib.'
                     'for example: `pip install matplotlib`.')
        raise e

    skeletons = np.array([item['keypoints'] for item in results])
    kpt_nums = 17
    if len(skeletons) > 0:
        kpt_nums = int(skeletons.shape[1] / 3)
    skeletons = skeletons.reshape(-1, kpt_nums, 3)
    if kpt_nums == 17:  #plot coco keypoint
        EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8),
                 (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14),
                 (13, 15), (14, 16), (11, 12)]
    else:  #plot mpii keypoint
        EDGES = [(0, 1), (1, 2), (3, 4), (4, 5), (2, 6), (3, 6), (6, 7), (7, 8),
                 (8, 9), (10, 11), (11, 12), (13, 14), (14, 15), (8, 12),
                 (8, 13)]
    NUM_EDGES = len(EDGES)

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')
    plt.figure()

    img = np.array(image).astype('float32')

    color_set = results['colors'] if 'colors' in results else None

    if 'bbox' in results and ids is None:
        bboxs = results['bbox']
        for j, rect in enumerate(bboxs):
            xmin, ymin, xmax, ymax = rect
            color = colors[0] if color_set is None else colors[color_set[j] %
                                                               len(colors)]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)

    canvas = img.copy()
    for i in range(kpt_nums):
        for j in range(len(skeletons)):
            if skeletons[j][i, 2] < visual_thread:
                continue
            if ids is None:
                color = colors[i] if color_set is None else colors[color_set[j]
                                                                   %
                                                                   len(colors)]
            else:
                color = get_color(ids[j])

            cv2.circle(
                canvas,
                tuple(skeletons[j][i, 0:2].astype('int32')),
                2,
                color,
                thickness=-1)

    to_plot = cv2.addWeighted(img, 0.3, canvas, 0.7, 0)
    fig = matplotlib.pyplot.gcf()

    stickwidth = 2

    for i in range(NUM_EDGES):
        for j in range(len(skeletons)):
            edge = EDGES[i]
            if skeletons[j][edge[0], 2] < visual_thread or skeletons[j][edge[
                    1], 2] < visual_thread:
                continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                       (int(length / 2), stickwidth),
                                       int(angle), 0, 360, 1)
            if ids is None:
                color = colors[i] if color_set is None else colors[color_set[j]
                                                                   %
                                                                   len(colors)]
            else:
                color = get_color(ids[j])
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    image = Image.fromarray(canvas.astype('uint8'))
    plt.close()
    return image


def draw_pose3d(image,
                pose3d,
                pose2d=None,
                visual_thread=0.6,
                save_name='pose3d.jpg',
                returnimg=True):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        plt.switch_backend('agg')
    except Exception as e:
        logger.error('Matplotlib not found, please install matplotlib.'
                     'for example: `pip install matplotlib`.')
        raise e

    if pose3d.shape[0] == 24:
        joints_connectivity_dict = [
            [0, 1, 0], [1, 2, 0], [5, 4, 1], [4, 3, 1], [2, 3, 0], [2, 14, 1],
            [3, 14, 1], [14, 16, 1], [15, 16, 1], [15, 12, 1], [6, 7, 0],
            [7, 8, 0], [11, 10, 1], [10, 9, 1], [8, 12, 0], [9, 12, 1],
            [12, 19, 1], [19, 18, 1], [19, 20, 0], [19, 21, 1], [22, 20, 0],
            [23, 21, 1]
        ]
    elif pose3d.shape[0] == 14:
        joints_connectivity_dict = [
            [0, 1, 0], [1, 2, 0], [5, 4, 1], [4, 3, 1], [2, 3, 0], [2, 12, 0],
            [3, 12, 1], [6, 7, 0], [7, 8, 0], [11, 10, 1], [10, 9, 1],
            [8, 12, 0], [9, 12, 1], [12, 13, 1]
        ]
    else:
        print(
            "not defined joints number :{}, cannot visualize because unknown of joint connectivity".
            format(pose.shape[0]))
        return

    def draw3Dpose(pose3d,
                   ax,
                   lcolor="#3498db",
                   rcolor="#e74c3c",
                   add_labels=False):
        #    pose3d = orthographic_projection(pose3d, cam)
        for i in joints_connectivity_dict:
            x, y, z = [
                np.array([pose3d[i[0], j], pose3d[i[1], j]]) for j in range(3)
            ]
            ax.plot(-x, -z, -y, lw=2, c=lcolor if i[2] else rcolor)

        RADIUS = 1000
        center_xy = 2 if pose3d.shape[0] == 14 else 14
        x, y, z = pose3d[center_xy, 0], pose3d[center_xy, 1], pose3d[center_xy,
                                                                     2]
        ax.set_xlim3d([-RADIUS + x, RADIUS + x])
        ax.set_ylim3d([-RADIUS + y, RADIUS + y])
        ax.set_zlim3d([-RADIUS + z, RADIUS + z])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    def draw2Dpose(pose2d,
                   ax,
                   lcolor="#3498db",
                   rcolor="#e74c3c",
                   add_labels=False):
        for i in joints_connectivity_dict:
            if pose2d[i[0], 2] and pose2d[i[1], 2]:
                x, y = [
                    np.array([pose2d[i[0], j], pose2d[i[1], j]])
                    for j in range(2)
                ]
                ax.plot(x, y, 0, lw=2, c=lcolor if i[2] else rcolor)

    def draw_img_pose(pose3d,
                      pose2d=None,
                      frame=None,
                      figsize=(12, 12),
                      savepath=None):
        fig = plt.figure(figsize=figsize, dpi=80)
        # fig.clear()
        fig.tight_layout()

        ax = fig.add_subplot(221)
        if frame is not None:
            ax.imshow(frame, interpolation='nearest')
        if pose2d is not None:
            draw2Dpose(pose2d, ax)

        ax = fig.add_subplot(222, projection='3d')
        ax.view_init(45, 45)
        draw3Dpose(pose3d, ax)
        ax = fig.add_subplot(223, projection='3d')
        ax.view_init(0, 0)
        draw3Dpose(pose3d, ax)
        ax = fig.add_subplot(224, projection='3d')
        ax.view_init(0, 90)
        draw3Dpose(pose3d, ax)

        if savepath is not None:
            plt.savefig(savepath)
            plt.close()
        else:
            return fig

    def fig2data(fig):
        """
        fig = plt.figure()
        image = fig2data(fig)
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        return image.convert("RGB")

    fig = draw_img_pose(pose3d, pose2d, frame=image)
    data = fig2data(fig)
    if returnimg is False:
        data.save(save_name)
    else:
        return data
