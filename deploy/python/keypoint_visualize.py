# coding: utf-8
# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import cv2
import os
import numpy as np
import math


def map_coco_to_personlab(keypoints):
    permute = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    return keypoints[:, permute, :]


def draw_pose(imgfile,
              results,
              visual_thread=0.6,
              save_name='pose.jpg',
              save_dir='output',
              returnimg=False):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        plt.switch_backend('agg')
    except Exception as e:
        logger.error('Matplotlib not found, please install matplotlib.'
                     'for example: `pip install matplotlib`.')
        raise e

    EDGES = [(0, 14), (0, 13), (0, 4), (0, 1), (14, 16), (13, 15), (4, 10),
             (1, 7), (10, 11), (7, 8), (11, 12), (8, 9), (4, 5), (1, 2), (5, 6),
             (2, 3)]
    NUM_EDGES = len(EDGES)

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')
    plt.figure()

    img = cv2.imread(imgfile) if type(imgfile) == str else imgfile
    skeletons, scores = results['keypoint']

    if 'bbox' in results:
        bboxs = results['bbox']
        for idx, rect in enumerate(bboxs):
            xmin, ymin, xmax, ymax = rect
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[0], 1)

    canvas = img.copy()
    for i in range(17):
        rgba = np.array(cmap(1 - i / 17. - 1. / 34))
        rgba[0:3] *= 255
        for j in range(len(skeletons)):
            if skeletons[j][i, 2] < visual_thread:
                continue
            cv2.circle(
                canvas,
                tuple(skeletons[j][i, 0:2].astype('int32')),
                2,
                colors[i],
                thickness=-1)

    to_plot = cv2.addWeighted(img, 0.3, canvas, 0.7, 0)
    fig = matplotlib.pyplot.gcf()

    stickwidth = 2

    skeletons = map_coco_to_personlab(skeletons)
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
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    if returnimg:
        return canvas
    save_name = os.path.join(
        save_dir, os.path.splitext(os.path.basename(imgfile))[0] + '_vis.jpg')
    plt.imsave(save_name, canvas[:, :, ::-1])
    print("keypoint visualize image saved to: " + save_name)
    plt.close()
