import json
import glob
import os
import numpy as np
import argparse
import logging
logger = logging.getLogger(__name__)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def poly2bbox(points):
    xs = []
    ys = []
    for p in points:
        xs.append(float(p['x']))
        ys.append(float(p['y']))
    return [min(xs), min(ys), max(xs), max(ys)]


def get_images(data, num, image_file):
    image = {}
    image['height'] = data['size']['height']
    image['width'] = data['size']['width']
    image['id'] = num + 1
    image['file_name'] = image_file
    return image


def get_anno(points, bbox, image_num, object_num):
    anno = {}
    xyxy = []
    for p in points:
        point_x = p['x']
        point_y = p['y']
        xyxy.extend([point_x, point_y])
    anno['segmentation'] = [xyxy]
    anno['iscrowd'] = 0
    anno['image_id'] = image_num + 1
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    anno['bbox'] = [bbox[0], bbox[1], bbox_w, bbox_h]
    anno['area'] = bbox_w * bbox_h
    anno['category_id'] = 1
    anno['id'] = object_num + 1
    return anno


def deal_json(anno_path):
    data_coco = {}
    images_list = []
    annotations_list = []
    category = {}
    category['supercategory'] = '0'
    category['id'] = 1
    category['name'] = '1'
    data_coco['categories'] = [category]

    image_num = -1
    object_num = -1
    ct = 0
    anno_file = os.path.join(anno_path, 'all.txt')
    with open(anno_file, 'r') as f:
        files = f.readlines()
    flag = True

    for f in files:
        if flag:
            flag = False
            continue
        image_num = image_num + 1
        image = f.split()[1]
        anno = eval(f.split()[-1])['result'][0]
        images_list.append(get_images(anno, image_num, image))

        if len(anno['elements']) == 0:
            continue

        for obj in anno['elements']:
            object_num += 1
            points = obj['points']
            bbox = poly2bbox(points)
            annotations_list.append(
                get_anno(points, bbox, image_num, object_num))

    data_coco['images'] = images_list
    data_coco['annotations'] = annotations_list
    return data_coco


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--anno_path', help='the path of annotation')
    args = parser.parse_args()
    save_path = os.path.join(args.anno_path, 'coco_anno.json')
    data_coco = deal_json(args.anno_path)
    json.dump(data_coco, open(save_path, 'w'), indent=4, cls=MyEncoder)


if __name__ == '__main__':
    main()
