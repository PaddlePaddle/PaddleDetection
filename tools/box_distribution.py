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
from pycocotools.coco import COCO
from tqdm import tqdm


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


def get_ratio_infos(jsonfile, out_img, eval_size, small_stride):
    coco = COCO(annotation_file=jsonfile)
    allannjson = json.load(open(jsonfile, 'r'))
    be_im_id = allannjson['annotations'][0]['image_id'] 
    be_im_w = []
    be_im_h = []
    ratio_w = []
    ratio_h = []
    im_wid,im_hei=[],[]
    for ann in tqdm(allannjson['annotations']):
        if ann['iscrowd']:
            continue
        x0, y0, w, h = ann['bbox'][:]
        if be_im_id == ann['image_id']:
            be_im_w.append(w)
            be_im_h.append(h)
        else:
            im_w = coco.imgs[be_im_id]['width']
            im_h = coco.imgs[be_im_id]['height']
            im_wid.append(im_w)
            im_hei.append(im_h)
            im_m_w = np.mean(be_im_w)
            im_m_h = np.mean(be_im_h)
            dis_w = im_m_w / im_w
            dis_h = im_m_h / im_h
            ratio_w.append(dis_w)
            ratio_h.append(dis_h)
            be_im_id = ann['image_id']
            be_im_w = [w]
            be_im_h = [h]
        

    im_w = coco.imgs[be_im_id]['width']
    im_h = coco.imgs[be_im_id]['height']
    im_wid.append(im_w)
    im_hei.append(im_h)
    all_im_m_w = np.mean(im_wid)
    all_im_m_h = np.mean(im_hei)


    im_m_w = np.mean(be_im_w)
    im_m_h = np.mean(be_im_h)
    dis_w = im_m_w / im_w
    dis_h = im_m_h / im_h
    ratio_w.append(dis_w)
    ratio_h.append(dis_h)
    mid_w = median(ratio_w)
    mid_h = median(ratio_h)

    reg_ratio = []
    ratio_all = ratio_h + ratio_w
    for r in ratio_all:
        if r < 0.2:
            reg_ratio.append(r)
        elif r < 0.4:
            reg_ratio.append(r/2)
        else:
            reg_ratio.append(r/4)
    reg_ratio = sorted(reg_ratio)
    max_ratio = reg_ratio[int(0.95*len(reg_ratio))]
    reg_max = round(max_ratio*eval_size/small_stride)
    
    ratio_w = [i * 1000 for i in ratio_w]
    ratio_h = [i * 1000 for i in ratio_h]
    print(f'Suggested reg_range[1] is {reg_max+1}' )
    print(f'Mean of all img_w is {all_im_m_w}')
    print(f'Mean of all img_h is {all_im_m_h}') 
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
        '--eval_size', type=int, default=640, help="eval size.")
    parser.add_argument(
        '--small_stride', type=int, default=8, help="smallest stride.")
    parser.add_argument(
        '--out_img',
        type=str,
        default='box_distribution.jpg',
        help="Name of distibution img.")
    args = parser.parse_args()

    get_ratio_infos(args.json_path, args.out_img, args.eval_size, args.small_stride)


if __name__ == "__main__":
    main()