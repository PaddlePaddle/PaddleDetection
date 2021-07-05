# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os.path as osp
import json
import glob
import cv2
import argparse

# add python path of PadleDetection to sys.path
parent_path = osp.abspath(osp.join(__file__, *(['..'] * 3)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from ppdet.modeling.bbox_utils import poly2rbox
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

class_name_15 = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter'
]

class_name_16 = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter', 'container-crane'
]


def dota_2_coco(image_dir,
                txt_dir,
                json_path='dota_coco.json',
                is_obb=True,
                dota_version='v1.0'):
    """
    image_dir: image dir
    txt_dir: txt label dir
    json_path: json save path
    is_obb: is obb or not
    dota_version: dota_version v1.0 or v1.5 or v2.0
    """

    img_lists = glob.glob("{}/*.png".format(image_dir))
    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    inst_count = 0

    # categories
    class_name2id = {}
    if dota_version == 'v1.0':
        for class_id, class_name in enumerate(class_name_15):
            class_name2id[class_name] = class_id + 1
            single_cat = {
                'id': class_id + 1,
                'name': class_name,
                'supercategory': class_name
            }
            data_dict['categories'].append(single_cat)

    for image_id, img_path in enumerate(img_lists):
        single_image = {}
        basename = osp.basename(img_path)
        single_image['file_name'] = basename
        single_image['id'] = image_id
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        single_image['width'] = width
        single_image['height'] = height
        # add image
        data_dict['images'].append(single_image)

        # annotations
        anno_txt_path = osp.join(txt_dir, osp.splitext(basename)[0] + '.txt')
        if not osp.exists(anno_txt_path):
            logger.warning('path of {} not exists'.format(anno_txt_path))

        for line in open(anno_txt_path):
            line = line.strip()
            # skip
            if line.find('imagesource') >= 0 or line.find('gsd') >= 0:
                continue

            # x1,y1,x2,y2,x3,y3,x4,y4 class_name, is_different
            single_obj_anno = line.split(' ')
            assert len(single_obj_anno) == 10
            single_obj_poly = [float(e) for e in single_obj_anno[0:8]]
            single_obj_classname = single_obj_anno[8]
            single_obj_different = int(single_obj_anno[9])

            single_obj = {}

            single_obj['category_id'] = class_name2id[single_obj_classname]
            single_obj['segmentation'] = []
            single_obj['segmentation'].append(single_obj_poly)
            single_obj['iscrowd'] = 0

            # rbox or bbox
            if is_obb:
                polys = [single_obj_poly]
                rboxs = poly2rbox(polys)
                rbox = rboxs[0].tolist()
                single_obj['bbox'] = rbox
                single_obj['area'] = rbox[2] * rbox[3]
            else:
                xmin, ymin, xmax, ymax = min(single_obj_poly[0::2]), min(single_obj_poly[1::2]), \
                                     max(single_obj_poly[0::2]), max(single_obj_poly[1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                single_obj['area'] = width * height

            single_obj['image_id'] = image_id
            data_dict['annotations'].append(single_obj)
            single_obj['id'] = inst_count
            inst_count = inst_count + 1
            # add annotation
            data_dict['annotations'].append(single_obj)

    with open(json_path, 'w') as f:
        json.dump(data_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dota anno to coco')
    parser.add_argument('--images_dir', help='path_to_images')
    parser.add_argument('--label_dir', help='path_to_labelTxt', type=str)
    parser.add_argument(
        '--json_path',
        help='save json path',
        type=str,
        default='dota_coco.json')
    parser.add_argument(
        '--is_obb', help='is_obb or not', type=bool, default=True)
    parser.add_argument(
        '--dota_version',
        help='dota_version, v1.0 or v1.5 or v2.0',
        type=str,
        default='v1.0')

    args = parser.parse_args()

    # process
    dota_2_coco(args.images_dir, args.label_dir, args.json_path, args.is_obb,
                args.dota_version)
    print('done!')
