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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from collections import OrderedDict

import paddle

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['face_eval_run', 'lmk2out']


def face_eval_run(model,
                  image_dir,
                  gt_file,
                  pred_dir='output/pred',
                  eval_mode='widerface',
                  multi_scale=False):
    # load ground truth files
    with open(gt_file, 'r') as f:
        gt_lines = f.readlines()
    imid2path = []
    pos_gt = 0
    while pos_gt < len(gt_lines):
        name_gt = gt_lines[pos_gt].strip('\n\t').split()[0]
        imid2path.append(name_gt)
        pos_gt += 1
        n_gt = int(gt_lines[pos_gt].strip('\n\t').split()[0])
        pos_gt += 1 + n_gt
    logger.info('The ground truth file load {} images'.format(len(imid2path)))

    dets_dist = OrderedDict()
    for iter_id, im_path in enumerate(imid2path):
        image_path = os.path.join(image_dir, im_path)
        if eval_mode == 'fddb':
            image_path += '.jpg'
        assert os.path.exists(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if multi_scale:
            shrink, max_shrink = get_shrink(image.shape[0], image.shape[1])
            det0 = detect_face(model, image, shrink)
            det1 = flip_test(model, image, shrink)
            [det2, det3] = multi_scale_test(model, image, max_shrink)
            det4 = multi_scale_test_pyramid(model, image, max_shrink)
            det = np.row_stack((det0, det1, det2, det3, det4))
            dets = bbox_vote(det)
        else:
            dets = detect_face(model, image, 1)
        if eval_mode == 'widerface':
            save_widerface_bboxes(image_path, dets, pred_dir)
        else:
            dets_dist[im_path] = dets
        if iter_id % 100 == 0:
            logger.info('Test iter {}'.format(iter_id))
    if eval_mode == 'fddb':
        save_fddb_bboxes(dets_dist, pred_dir)
    logger.info("Finish evaluation.")


def detect_face(model, image, shrink):
    image_shape = [image.shape[0], image.shape[1]]
    if shrink != 1:
        h, w = int(image_shape[0] * shrink), int(image_shape[1] * shrink)
        image = cv2.resize(image, (w, h))
        image_shape = [h, w]

    img = face_img_process(image)
    image_shape = np.asarray([image_shape])
    scale_factor = np.asarray([[shrink, shrink]])
    data = {
        "image": paddle.to_tensor(
            img, dtype='float32'),
        "im_shape": paddle.to_tensor(
            image_shape, dtype='float32'),
        "scale_factor": paddle.to_tensor(
            scale_factor, dtype='float32')
    }
    model.eval()
    detection = model(data)
    detection = detection['bbox'].numpy()
    # layout: xmin, ymin, xmax. ymax, score
    if np.prod(detection.shape) == 1:
        logger.info("No face detected")
        return np.array([[0, 0, 0, 0, 0]])
    det_conf = detection[:, 1]
    det_xmin = detection[:, 2]
    det_ymin = detection[:, 3]
    det_xmax = detection[:, 4]
    det_ymax = detection[:, 5]

    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))
    return det


def flip_test(model, image, shrink):
    img = cv2.flip(image, 1)
    det_f = detect_face(model, img, shrink)
    det_t = np.zeros(det_f.shape)
    img_width = image.shape[1]
    det_t[:, 0] = img_width - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = img_width - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def multi_scale_test(model, image, max_shrink):
    # Shrink detecting is only used to detect big faces
    st = 0.5 if max_shrink >= 0.75 else 0.5 * max_shrink
    det_s = detect_face(model, image, st)
    index = np.where(
        np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1)
        > 30)[0]
    det_s = det_s[index, :]
    # Enlarge one times
    bt = min(2, max_shrink) if max_shrink > 1 else (st + max_shrink) / 2
    det_b = detect_face(model, image, bt)

    # Enlarge small image x times for small faces
    if max_shrink > 2:
        bt *= 2
        while bt < max_shrink:
            det_b = np.row_stack((det_b, detect_face(model, image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(model, image, max_shrink)))

    # Enlarged images are only used to detect small faces.
    if bt > 1:
        index = np.where(
            np.minimum(det_b[:, 2] - det_b[:, 0] + 1,
                       det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    # Shrinked images are only used to detect big faces.
    else:
        index = np.where(
            np.maximum(det_b[:, 2] - det_b[:, 0] + 1,
                       det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]
    return det_s, det_b


def multi_scale_test_pyramid(model, image, max_shrink):
    # Use image pyramids to detect faces
    det_b = detect_face(model, image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [0.75, 1.25, 1.5, 1.75]
    for i in range(len(st)):
        if st[i] <= max_shrink:
            det_temp = detect_face(model, image, st[i])
            # Enlarged images are only used to detect small faces.
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            # Shrinked images are only used to detect big faces.
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b


def to_chw(image):
    """
    Transpose image from HWC to CHW.
    Args:
        image (np.array): an image with HWC layout.
    """
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    return image


def face_img_process(image,
                     mean=[104., 117., 123.],
                     std=[127.502231, 127.502231, 127.502231]):
    img = np.array(image)
    img = to_chw(img)
    img = img.astype('float32')
    img -= np.array(mean)[:, np.newaxis, np.newaxis].astype('float32')
    img /= np.array(std)[:, np.newaxis, np.newaxis].astype('float32')
    img = [img]
    img = np.array(img)
    return img


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
    keep_index = np.where(dets[:, 4] >= 0.01)[0]
    dets = dets[keep_index, :]
    return dets


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
