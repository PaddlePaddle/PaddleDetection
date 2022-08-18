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

import os
import re
import glob

import numpy as np
from multiprocessing import Pool
from functools import partial
from shapely.geometry import Polygon
import argparse

nms_thresh = 0.1

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


def rbox_iou(g, p):
    """
    iou of rbox
    """
    g = np.array(g)
    p = np.array(p)
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    g = g.buffer(0)
    p = p.buffer(0)
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def py_cpu_nms_poly_fast(dets, thresh):
    """
    Args:
        dets: pred results
        thresh: nms threshold

    Returns: index of keep
    """
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = [
            dets[i][0], dets[i][1], dets[i][2], dets[i][3], dets[i][4],
            dets[i][5], dets[i][6], dets[i][7]
        ]
        polys.append(tm_polygon)
    polys = np.array(polys)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        # h_keep_inds = np.where(hbb_ovr == 0)[0]
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = rbox_iou(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
            # ovr.append(iou)
            # ovr_index.append(tmp_order[j])

        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]

        order = order[inds + 1]
    return keep


def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly) / 2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly


def nmsbynamedict(nameboxdict, nms, thresh):
    """
    Args:
        nameboxdict: nameboxdict
        nms:   nms
        thresh: nms threshold

    Returns: nms result as dict
    """
    nameboxnmsdict = {x: [] for x in nameboxdict}
    for imgname in nameboxdict:
        keep = nms(np.array(nameboxdict[imgname]), thresh)
        outdets = []
        for index in keep:
            outdets.append(nameboxdict[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict


def merge_single(output_dir, nms, pred_class_lst):
    """
    Args:
        output_dir: output_dir
        nms:  nms
        pred_class_lst: pred_class_lst
        class_name: class_name

    Returns:

    """
    class_name, pred_bbox_list = pred_class_lst
    nameboxdict = {}
    for line in pred_bbox_list:
        splitline = line.split(' ')
        subname = splitline[0]
        splitname = subname.split('__')
        oriname = splitname[0]
        pattern1 = re.compile(r'__\d+___\d+')
        x_y = re.findall(pattern1, subname)
        x_y_2 = re.findall(r'\d+', x_y[0])
        x, y = int(x_y_2[0]), int(x_y_2[1])

        pattern2 = re.compile(r'__([\d+\.]+)__\d+___')

        rate = re.findall(pattern2, subname)[0]

        confidence = splitline[1]
        poly = list(map(float, splitline[2:]))
        origpoly = poly2origpoly(poly, x, y, rate)
        det = origpoly
        det.append(confidence)
        det = list(map(float, det))
        if (oriname not in nameboxdict):
            nameboxdict[oriname] = []
        nameboxdict[oriname].append(det)
    nameboxnmsdict = nmsbynamedict(nameboxdict, nms, nms_thresh)

    # write result
    dstname = os.path.join(output_dir, class_name + '.txt')
    with open(dstname, 'w') as f_out:
        for imgname in nameboxnmsdict:
            for det in nameboxnmsdict[imgname]:
                confidence = det[-1]
                bbox = det[0:-1]
                outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(
                    map(str, bbox))
                f_out.write(outline + '\n')


def dota_generate_test_result(pred_txt_dir,
                              output_dir='output',
                              dota_version='v1.0'):
    """
    pred_txt_dir: dir of pred txt
    output_dir: dir of output
    dota_version: dota_version v1.0 or v1.5 or v2.0
    """
    pred_txt_list = glob.glob("{}/*.txt".format(pred_txt_dir))

    # step1: summary pred bbox
    pred_classes = {}
    class_lst = class_name_15 if dota_version == 'v1.0' else class_name_16
    for class_name in class_lst:
        pred_classes[class_name] = []

    for current_txt in pred_txt_list:
        img_id = os.path.split(current_txt)[1]
        img_id = img_id.split('.txt')[0]
        with open(current_txt) as f:
            res = f.readlines()
            for item in res:
                item = item.split(' ')
                pred_class = item[0]
                item[0] = img_id
                pred_bbox = ' '.join(item)
                pred_classes[pred_class].append(pred_bbox)

    pred_classes_lst = []
    for class_name in pred_classes.keys():
        print('class_name: {}, count: {}'.format(class_name,
                                                 len(pred_classes[class_name])))
        pred_classes_lst.append((class_name, pred_classes[class_name]))

    # step2: merge
    pool = Pool(len(class_lst))
    nms = py_cpu_nms_poly_fast
    mergesingle_fn = partial(merge_single, output_dir, nms)
    pool.map(mergesingle_fn, pred_classes_lst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dota anno to coco')
    parser.add_argument('--pred_txt_dir', help='path of pred txt dir')
    parser.add_argument(
        '--output_dir', help='path of output dir', default='output')
    parser.add_argument(
        '--dota_version',
        help='dota_version, v1.0 or v1.5 or v2.0',
        type=str,
        default='v1.0')

    args = parser.parse_args()

    # process
    dota_generate_test_result(args.pred_txt_dir, args.output_dir,
                              args.dota_version)
    print('done!')
