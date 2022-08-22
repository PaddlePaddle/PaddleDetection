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

import matplotlib.pyplot as plt
import json
import numpy as np
import argparse


def median(data):
    data.sort()
    mid = len(data) // 2
    median = (data[mid] + data[~mid]) / 2
    return median


def draw_distribution(width, height, out_path):
    w_bins = int((max(width) - min(width)) // 10)
    h_bins = int((max(height) - min(height)) // 10)
    plt.figure()
    plt.subplot(221)
    plt.hist(width, bins=w_bins, color='green')
    plt.xlabel('Width rate *1000')
    plt.ylabel('number')
    plt.title('Distribution of Width')
    plt.subplot(222)
    plt.hist(height, bins=h_bins, color='blue')
    plt.xlabel('Height rate *1000')
    plt.title('Distribution of Height')
    plt.savefig(out_path)
    print(f'Distribution saved as {out_path}')
    plt.show()


def get_ratio_infos(jsonfile, out_img):
    allannjson = json.load(open(jsonfile, 'r'))
    be_im_id = 1
    be_im_w = []
    be_im_h = []
    ratio_w = []
    ratio_h = []
    images = allannjson['images']
    for i, ann in enumerate(allannjson['annotations']):
        if ann['iscrowd']:
            continue
        x0, y0, w, h = ann['bbox'][:]
        if be_im_id == ann['image_id']:
            be_im_w.append(w)
            be_im_h.append(h)
        else:
            im_w = images[be_im_id - 1]['width']
            im_h = images[be_im_id - 1]['height']
            im_m_w = np.mean(be_im_w)
            im_m_h = np.mean(be_im_h)
            dis_w = im_m_w / im_w
            dis_h = im_m_h / im_h
            ratio_w.append(dis_w)
            ratio_h.append(dis_h)
            be_im_id = ann['image_id']
            be_im_w = [w]
            be_im_h = [h]

    im_w = images[be_im_id - 1]['width']
    im_h = images[be_im_id - 1]['height']
    im_m_w = np.mean(be_im_w)
    im_m_h = np.mean(be_im_h)
    dis_w = im_m_w / im_w
    dis_h = im_m_h / im_h
    ratio_w.append(dis_w)
    ratio_h.append(dis_h)
    mid_w = median(ratio_w)
    mid_h = median(ratio_h)
    ratio_w = [i * 1000 for i in ratio_w]
    ratio_h = [i * 1000 for i in ratio_h]
    print(f'Median of ratio_w is {mid_w}')
    print(f'Median of ratio_h is {mid_h}')
    print('all_img with box: ', len(ratio_h))
    print('all_ann: ', len(allannjson['annotations']))
    draw_distribution(ratio_w, ratio_h, out_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json_path', type=str, default=None, help="Dataset json path.")
    parser.add_argument(
        '--out_img',
        type=str,
        default='box_distribution.jpg',
        help="Name of distibution img.")
    args = parser.parse_args()

    get_ratio_infos(args.json_path, args.out_img)


if __name__ == "__main__":
    main()
