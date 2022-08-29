# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#
# Reference: https://github.com/CAPTAIN-WHU/DOTA_devkit

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import copy
from numbers import Number
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm
import shapely.geometry as shgeo


def choose_best_pointorder_fit_another(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = poly1
    combinate = [
        np.array([x1, y1, x2, y2, x3, y3, x4, y4]),
        np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
        np.array([x3, y3, x4, y4, x1, y1, x2, y2]),
        np.array([x4, y4, x1, y1, x2, y2, x3, y3])
    ]
    dst_coordinate = np.array(poly2)
    distances = np.array(
        [np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]


def cal_line_length(point1, point2):
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


class SliceBase(object):
    def __init__(self,
                 gap=512,
                 subsize=1024,
                 thresh=0.7,
                 choosebestpoint=True,
                 ext='.png',
                 padding=True,
                 num_process=8,
                 image_only=False):
        self.gap = gap
        self.subsize = subsize
        self.slide = subsize - gap
        self.thresh = thresh
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        self.padding = padding
        self.num_process = num_process
        self.image_only = image_only

    def get_windows(self, height, width):
        windows = []
        left, up = 0, 0
        while (left < width):
            if (left + self.subsize >= width):
                left = max(width - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, width - 1)
                down = min(up + self.subsize, height - 1)
                windows.append((left, up, right, down))
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= width):
                break
            else:
                left = left + self.slide

        return windows

    def slice_image_single(self, image, windows, output_dir, output_name):
        image_dir = os.path.join(output_dir, 'images')
        for (left, up, right, down) in windows:
            image_name = output_name + str(left) + '___' + str(up) + self.ext
            subimg = copy.deepcopy(image[up:up + self.subsize, left:left +
                                         self.subsize])
            h, w, c = subimg.shape
            if (self.padding):
                outimg = np.zeros((self.subsize, self.subsize, 3))
                outimg[0:h, 0:w, :] = subimg
                cv2.imwrite(os.path.join(image_dir, image_name), outimg)
            else:
                cv2.imwrite(os.path.join(image_dir, image_name), subimg)

    def iof(self, poly1, poly2):
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou

    def translate(self, poly, left, up):
        n = len(poly)
        out_poly = np.zeros(n)
        for i in range(n // 2):
            out_poly[i * 2] = int(poly[i * 2] - left)
            out_poly[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return out_poly

    def get_poly4_from_poly5(self, poly):
        distances = [
            cal_line_length((poly[i * 2], poly[i * 2 + 1]),
                            (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1]))
            for i in range(int(len(poly) / 2 - 1))
        ]
        distances.append(
            cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        out_poly = []
        while count < 5:
            if (count == pos):
                out_poly.append(
                    (poly[count * 2] + poly[(count * 2 + 2) % 10]) / 2)
                out_poly.append(
                    (poly[(count * 2 + 1) % 10] + poly[(count * 2 + 3) % 10]) /
                    2)
                count = count + 1
            elif (count == (pos + 1) % 5):
                count = count + 1
                continue

            else:
                out_poly.append(poly[count * 2])
                out_poly.append(poly[count * 2 + 1])
                count = count + 1
        return out_poly

    def slice_anno_single(self, annos, windows, output_dir, output_name):
        anno_dir = os.path.join(output_dir, 'labelTxt')
        for (left, up, right, down) in windows:
            image_poly = shgeo.Polygon(
                [(left, up), (right, up), (right, down), (left, down)])
            anno_file = output_name + str(left) + '___' + str(up) + '.txt'
            with open(os.path.join(anno_dir, anno_file), 'w') as f:
                for anno in annos:
                    gt_poly = shgeo.Polygon(
                        [(anno['poly'][0], anno['poly'][1]),
                         (anno['poly'][2], anno['poly'][3]),
                         (anno['poly'][4], anno['poly'][5]),
                         (anno['poly'][6], anno['poly'][7])])
                    if gt_poly.area <= 0:
                        continue
                    inter_poly, iof = self.iof(gt_poly, image_poly)
                    if iof == 1:
                        final_poly = self.translate(anno['poly'], left, up)
                    elif iof > 0:
                        inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                        out_poly = list(inter_poly.exterior.coords)[0:-1]
                        if len(out_poly) < 4 or len(out_poly) > 5:
                            continue

                        final_poly = []
                        for p in out_poly:
                            final_poly.append(p[0])
                            final_poly.append(p[1])

                        if len(out_poly) == 5:
                            final_poly = self.get_poly4_from_poly5(final_poly)

                        if self.choosebestpoint:
                            final_poly = choose_best_pointorder_fit_another(
                                final_poly, anno['poly'])

                        final_poly = self.translate(final_poly, left, up)
                        final_poly = np.clip(final_poly, 1, self.subsize)
                    else:
                        continue
                    outline = ' '.join(list(map(str, final_poly)))
                    if iof >= self.thresh:
                        outline = outline + ' ' + anno['name'] + ' ' + str(anno[
                            'difficult'])
                    else:
                        outline = outline + ' ' + anno['name'] + ' ' + '2'

                    f.write(outline + '\n')

    def slice_data_single(self, info, rate, output_dir):
        file_name = info['image_file']
        base_name = os.path.splitext(os.path.split(file_name)[-1])[0]
        base_name = base_name + '__' + str(rate) + '__'
        img = cv2.imread(file_name)
        if img.shape == ():
            return

        if (rate != 1):
            resize_img = cv2.resize(
                img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            resize_img = img

        height, width, _ = resize_img.shape
        windows = self.get_windows(height, width)
        self.slice_image_single(resize_img, windows, output_dir, base_name)
        if not self.image_only:
            annos = info['annotation']
            for anno in annos:
                anno['poly'] = list(map(lambda x: rate * x, anno['poly']))
            self.slice_anno_single(annos, windows, output_dir, base_name)

    def check_or_mkdirs(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def slice_data(self, infos, rates, output_dir):
        """
        Args:
            infos (list[dict]): data_infos
            rates (float, list): scale rates
            output_dir (str): output directory
        """
        if isinstance(rates, Number):
            rates = [rates, ]

        self.check_or_mkdirs(output_dir)
        self.check_or_mkdirs(os.path.join(output_dir, 'images'))
        if not self.image_only:
            self.check_or_mkdirs(os.path.join(output_dir, 'labelTxt'))

        pbar = tqdm(total=len(rates) * len(infos), desc='slicing data')

        if self.num_process <= 1:
            for rate in rates:
                for info in infos:
                    self.slice_data_single(info, rate, output_dir)
                    pbar.update()
        else:
            pool = Pool(self.num_process)
            for rate in rates:
                for info in infos:
                    pool.apply_async(
                        self.slice_data_single, (info, rate, output_dir),
                        callback=lambda x: pbar.update())

            pool.close()
            pool.join()

        pbar.close()
