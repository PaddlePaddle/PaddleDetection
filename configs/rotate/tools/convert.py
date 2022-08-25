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
import json
import cv2
from tqdm import tqdm
from multiprocessing import Pool


def load_dota_info(image_dir, anno_dir, file_name, ext=None):
    base_name, extension = os.path.splitext(file_name)
    if ext and (extension != ext and extension not in ext):
        return None
    info = {'image_file': os.path.join(image_dir, file_name), 'annotation': []}
    anno_file = os.path.join(anno_dir, base_name + '.txt')
    if not os.path.exists(anno_file):
        return info
    with open(anno_file, 'r') as f:
        for line in f:
            items = line.strip().split()
            if (len(items) < 9):
                continue

            anno = {
                'poly': list(map(float, items[:8])),
                'name': items[8],
                'difficult': '0' if len(items) == 9 else items[9],
            }
            info['annotation'].append(anno)

    return info


def load_dota_infos(root_dir, num_process=8, ext=None):
    image_dir = os.path.join(root_dir, 'images')
    anno_dir = os.path.join(root_dir, 'labelTxt')
    data_infos = []
    if num_process > 1:
        pool = Pool(num_process)
        results = []
        for file_name in os.listdir(image_dir):
            results.append(
                pool.apply_async(load_dota_info, (image_dir, anno_dir,
                                                  file_name, ext)))

        pool.close()
        pool.join()

        for result in results:
            info = result.get()
            if info:
                data_infos.append(info)

    else:
        for file_name in os.listdir(image_dir):
            info = load_dota_info(image_dir, anno_dir, file_name, ext)
            if info:
                data_infos.append(info)

    return data_infos


def process_single_sample(info, image_id, class_names):
    image_file = info['image_file']
    single_image = dict()
    single_image['file_name'] = os.path.split(image_file)[-1]
    single_image['id'] = image_id
    image = cv2.imread(image_file)
    height, width, _ = image.shape
    single_image['width'] = width
    single_image['height'] = height

    # process annotation field
    single_objs = []
    objects = info['annotation']
    for obj in objects:
        poly, name, difficult = obj['poly'], obj['name'], obj['difficult']
        if difficult == '2':
            continue

        single_obj = dict()
        single_obj['category_id'] = class_names.index(name) + 1
        single_obj['segmentation'] = [poly]
        single_obj['iscrowd'] = 0
        xmin, ymin, xmax, ymax = min(poly[0::2]), min(poly[1::2]), max(poly[
            0::2]), max(poly[1::2])
        width, height = xmax - xmin, ymax - ymin
        single_obj['bbox'] = [xmin, ymin, width, height]
        single_obj['area'] = height * width
        single_obj['image_id'] = image_id
        single_objs.append(single_obj)

    return (single_image, single_objs)


def data_to_coco(infos, output_path, class_names, num_process):
    data_dict = dict()
    data_dict['categories'] = []

    for i, name in enumerate(class_names):
        data_dict['categories'].append({
            'id': i + 1,
            'name': name,
            'supercategory': name
        })

    pbar = tqdm(total=len(infos), desc='data to coco')
    images, annotations = [], []
    if num_process > 1:
        pool = Pool(num_process)
        results = []
        for i, info in enumerate(infos):
            image_id = i + 1
            results.append(
                pool.apply_async(
                    process_single_sample, (info, image_id, class_names),
                    callback=lambda x: pbar.update()))

        pool.close()
        pool.join()

        for result in results:
            single_image, single_anno = result.get()
            images.append(single_image)
            annotations += single_anno

    else:
        for i, info in enumerate(infos):
            image_id = i + 1
            single_image, single_anno = process_single_sample(info, image_id,
                                                              class_names)
            images.append(single_image)
            annotations += single_anno
            pbar.update()

    pbar.close()

    for i, anno in enumerate(annotations):
        anno['id'] = i + 1

    data_dict['images'] = images
    data_dict['annotations'] = annotations

    with open(output_path, 'w') as f:
        json.dump(data_dict, f)
