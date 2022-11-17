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

import os
import json
import argparse
import numpy as np


def save_json(path, images, annotations, categories):
    new_json = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }
    with open(path, 'w') as f:
        json.dump(new_json, f)
    print('{} saved, with {} images and {} annotations.'.format(
        path, len(images), len(annotations)))


def gen_semi_data(data_dir,
                  json_file,
                  percent=10.0,
                  seed=1,
                  seed_offset=0,
                  txt_file=None):
    json_name = json_file.split('/')[-1].split('.')[0]
    json_file = os.path.join(data_dir, json_file)
    anno = json.load(open(json_file, 'r'))
    categories = anno['categories']
    all_images = anno['images']
    all_anns = anno['annotations']
    print(
        'Totally {} images and {} annotations, about {} gts per image.'.format(
            len(all_images), len(all_anns), len(all_anns) / len(all_images)))

    if txt_file:
        print('Using percent {} and seed {}.'.format(percent, seed))
        txt_file = os.path.join(data_dir, txt_file)
        sup_idx = json.load(open(txt_file, 'r'))[str(percent)][str(seed)]
        # max(sup_idx) = 117262 # 10%, sup_idx is not image_id
    else:
        np.random.seed(seed + seed_offset)
        sup_len = int(percent / 100.0 * len(all_images))
        sup_idx = np.random.choice(
            range(len(all_images)), size=sup_len, replace=False)
    labeled_images, labeled_anns = [], []
    labeled_im_ids = []
    unlabeled_images, unlabeled_anns = [], []

    for i in range(len(all_images)):
        if i in sup_idx:
            labeled_im_ids.append(all_images[i]['id'])
            labeled_images.append(all_images[i])
        else:
            unlabeled_images.append(all_images[i])

    for an in all_anns:
        im_id = an['image_id']
        if im_id in labeled_im_ids:
            labeled_anns.append(an)
        else:
            continue

    save_path = '{}/{}'.format(data_dir, 'semi_annotations')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    sup_name = '{}.{}@{}.json'.format(json_name, seed, int(percent))
    sup_path = os.path.join(save_path, sup_name)
    save_json(sup_path, labeled_images, labeled_anns, categories)

    unsup_name = '{}.{}@{}-unlabeled.json'.format(json_name, seed, int(percent))
    unsup_path = os.path.join(save_path, unsup_name)
    save_json(unsup_path, unlabeled_images, unlabeled_anns, categories)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset/coco')
    parser.add_argument(
        '--json_file', type=str, default='annotations/instances_train2017.json')
    parser.add_argument('--percent', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--seed_offset', type=int, default=0)
    parser.add_argument('--txt_file', type=str, default='COCO_supervision.txt')
    args = parser.parse_args()
    print(args)
    gen_semi_data(args.data_dir, args.json_file, args.percent, args.seed,
                  args.seed_offset, args.txt_file)
