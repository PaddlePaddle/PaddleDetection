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

import os
import numpy as np

from ppdet.data.source.widerface import widerface_label
from ppdet.utils.coco_eval import bbox2out

import logging
logger = logging.getLogger(__name__)

__all__ = [
    'get_shrink', 'bbox_vote', 'save_widerface_bboxes', 'save_fddb_bboxes',
    'to_chw_bgr', 'bbox2out', 'get_category_info', 'lmk2out'
]


def to_chw_bgr(image):
    """
    Transpose image from HWC to CHW and from RBG to BGR.
    Args:
        image (np.array): an image with HWC and RBG layout.
    """
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    image = image[[2, 1, 0], :, :]
    return image


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    if det.shape[0] == 0:
        dets = np.array([[10, 10, 20, 20, 0.002]])
        det = np.empty(shape=[0, 5])
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # nms
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            if det.shape[0] == 0:
                try:
                    dets = np.row_stack((dets, det_accu))
                except:
                    dets = det_accu
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4],
                                      axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    dets = dets[0:750, :]
    # Only keep 0.3 or more
    keep_index = np.where(dets[:, 4] >= 0.01)[0]
    dets = dets[keep_index, :]
    return dets


def get_shrink(height, width):
    """
    Args:
        height (int): image height.
        width (int): image width.
    """
    # avoid out of memory
    max_shrink_v1 = (0x7fffffff / 577.0 / (height * width))**0.5
    max_shrink_v2 = ((678 * 1024 * 2.0 * 2.0) / (height * width))**0.5

    def get_round(x, loc):
        str_x = str(x)
        if '.' in str_x:
            str_before, str_after = str_x.split('.')
            len_after = len(str_after)
            if len_after >= 3:
                str_final = str_before + '.' + str_after[0:loc]
                return float(str_final)
            else:
                return x

    max_shrink = get_round(min(max_shrink_v1, max_shrink_v2), 2) - 0.3
    if max_shrink >= 1.5 and max_shrink < 2:
        max_shrink = max_shrink - 0.1
    elif max_shrink >= 2 and max_shrink < 3:
        max_shrink = max_shrink - 0.2
    elif max_shrink >= 3 and max_shrink < 4:
        max_shrink = max_shrink - 0.3
    elif max_shrink >= 4 and max_shrink < 5:
        max_shrink = max_shrink - 0.4
    elif max_shrink >= 5:
        max_shrink = max_shrink - 0.5
    elif max_shrink <= 0.1:
        max_shrink = 0.1

    shrink = max_shrink if max_shrink < 1 else 1
    return shrink, max_shrink


def save_widerface_bboxes(image_path, bboxes_scores, output_dir):
    image_name = image_path.split('/')[-1]
    image_class = image_path.split('/')[-2]
    odir = os.path.join(output_dir, image_class)
    if not os.path.exists(odir):
        os.makedirs(odir)

    ofname = os.path.join(odir, '%s.txt' % (image_name[:-4]))
    f = open(ofname, 'w')
    f.write('{:s}\n'.format(image_class + '/' + image_name))
    f.write('{:d}\n'.format(bboxes_scores.shape[0]))
    for box_score in bboxes_scores:
        xmin, ymin, xmax, ymax, score = box_score
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(xmin, ymin, (
            xmax - xmin + 1), (ymax - ymin + 1), score))
    f.close()
    logger.info("The predicted result is saved as {}".format(ofname))


def save_fddb_bboxes(bboxes_scores,
                     output_dir,
                     output_fname='pred_fddb_res.txt'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    predict_file = os.path.join(output_dir, output_fname)
    f = open(predict_file, 'w')
    for image_path, dets in bboxes_scores.iteritems():
        f.write('{:s}\n'.format(image_path))
        f.write('{:d}\n'.format(dets.shape[0]))
        for box_score in dets:
            xmin, ymin, xmax, ymax, score = box_score
            width, height = xmax - xmin, ymax - ymin
            f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'
                    .format(xmin, ymin, width, height, score))
    logger.info("The predicted result is saved as {}".format(predict_file))
    return predict_file


def get_category_info(anno_file=None,
                      with_background=True,
                      use_default_label=False):
    if use_default_label or anno_file is None \
            or not os.path.exists(anno_file):
        logger.info("Not found annotation file {}, load "
                    "wider-face categories.".format(anno_file))
        return widerfaceall_category_info(with_background)
    else:
        logger.info("Load categories from {}".format(anno_file))
        return get_category_info_from_anno(anno_file, with_background)


def get_category_info_from_anno(anno_file, with_background=True):
    """
    Get class id to category id map and category id
    to category name map from annotation file.
    Args:
        anno_file (str): annotation file path
        with_background (bool, default True):
            whether load background as class 0.
    """
    cats = []
    with open(anno_file) as f:
        for line in f.readlines():
            cats.append(line.strip())

    if cats[0] != 'background' and with_background:
        cats.insert(0, 'background')
    if cats[0] == 'background' and not with_background:
        cats = cats[1:]

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name


def widerfaceall_category_info(with_background=True):
    """
    Get class id to category id map and category id
    to category name map of mixup wider_face dataset

    Args:
        with_background (bool, default True):
            whether load background as class 0.
    """
    label_map = widerface_label(with_background)
    label_map = sorted(label_map.items(), key=lambda x: x[1])
    cats = [l[0] for l in label_map]

    if with_background:
        cats.insert(0, 'background')

    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}

    return clsid2catid, catid2name


def lmk2out(results, is_bbox_normalized=False):
    """
    Args:
        results: request a dict, should include: `landmark`, `im_id`,
                 if is_bbox_normalized=True, also need `im_shape`.
        is_bbox_normalized: whether or not landmark is normalized.
    """
    xywh_res = []
    for t in results:
        bboxes = t['bbox'][0]
        lengths = t['bbox'][1][0]
        im_ids = np.array(t['im_id'][0]).flatten()
        if bboxes.shape == (1, 1) or bboxes is None:
            continue
        face_index = t['face_index'][0]
        prior_box = t['prior_boxes'][0]
        predict_lmk = t['landmark'][0]
        prior = np.reshape(prior_box, (-1, 4))
        predictlmk = np.reshape(predict_lmk, (-1, 10))

        k = 0
        for a in range(len(lengths)):
            num = lengths[a]
            im_id = int(im_ids[a])
            for i in range(num):
                score = bboxes[k][1]
                theindex = face_index[i][0]
                me_prior = prior[theindex, :]
                lmk_pred = predictlmk[theindex, :]
                prior_w = me_prior[2] - me_prior[0]
                prior_h = me_prior[3] - me_prior[1]
                prior_w_center = (me_prior[2] + me_prior[0]) / 2
                prior_h_center = (me_prior[3] + me_prior[1]) / 2
                lmk_decode = np.zeros((10))
                for j in [0, 2, 4, 6, 8]:
                    lmk_decode[j] = lmk_pred[j] * 0.1 * prior_w + prior_w_center
                for j in [1, 3, 5, 7, 9]:
                    lmk_decode[j] = lmk_pred[j] * 0.1 * prior_h + prior_h_center
                im_shape = t['im_shape'][0][a].tolist()
                image_h, image_w = int(im_shape[0]), int(im_shape[1])
                if is_bbox_normalized:
                    lmk_decode = lmk_decode * np.array([
                        image_w, image_h, image_w, image_h, image_w, image_h,
                        image_w, image_h, image_w, image_h
                    ])
                lmk_res = {
                    'image_id': im_id,
                    'landmark': lmk_decode,
                    'score': score,
                }
                xywh_res.append(lmk_res)
                k += 1
    return xywh_res
