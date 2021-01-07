# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import division

import cv2
import numpy as np
from PIL import Image, ImageDraw
import solov2_blazeface.face_makeup_main as face_makeup_main


def visualize_box_mask(im,
                       results,
                       threshold=0.5,
                       beard_file=None,
                       glasses_file=None,
                       hat_file=None):
    if isinstance(im, str):
        im = Image.open(im).convert('RGB')
    else:
        im = Image.fromarray(im)

    if 'segm' in results:
        im, x0, x1, y0, y1, flag_seg = draw_segm(
            im,
            results['segm'],
            results['label'],
            results['score'],
            threshold=threshold)
        return im, x0, x1, y0, y1, 0, 0, 0, 0, flag_seg
    if 'landmark' in results:
        im, left, right, up, bottom, h_xmin, h_ymin, h_xmax, h_ymax, flag_face = trans_lmk(
            im, results['landmark'], beard_file, glasses_file, hat_file)
        return im, left, right, up, bottom, h_xmin, h_ymin, h_xmax, h_ymax, flag_face
    else:
        return im, 0, 0, 0, 0, 0, 0, 0, 0, 0


def draw_segm(im, np_segms, np_label, np_score, threshold=0.5, alpha=0.7):
    """
    Draw segmentation on image
    """
    im = np.array(im).astype('float32')
    np_segms = np_segms.astype(np.uint8)
    index_label = np.where(np_label == 0)[0]
    index = np.where(np_score[index_label] > threshold)[0]
    index = index_label[index]
    if index.size == 0:
        im = Image.fromarray(im.astype('uint8'))
        return im, 0, 0, 0, 0, 0
    person_segms = np_segms[index]
    person_mask_single_channel = np.sum(person_segms, axis=0)
    person_mask_single_channel[person_mask_single_channel > 1] = 1
    person_mask = np.expand_dims(person_mask_single_channel, axis=2)
    person_mask = np.repeat(person_mask, 3, axis=2)
    im = im * person_mask

    sum_x = np.sum(person_mask_single_channel, axis=0)
    x = np.where(sum_x > 0.5)[0]
    sum_y = np.sum(person_mask_single_channel, axis=1)
    y = np.where(sum_y > 0.5)[0]
    x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]

    return Image.fromarray(im.astype('uint8')), x0, x1, y0, y1, 1


def lmk2out(bboxes, np_lmk, im_info, threshold=0.5, is_bbox_normalized=True):
    image_w, image_h = im_info['origin_shape']
    scale = im_info['scale']
    face_index, landmark, prior_box = np_lmk[:]
    xywh_res = []
    if bboxes.shape == (1, 1) or bboxes is None:
        return np.array([])
    prior = np.reshape(prior_box, (-1, 4))
    predict_lmk = np.reshape(landmark, (-1, 10))
    k = 0
    for i in range(bboxes.shape[0]):
        score = bboxes[i][1]
        if score < threshold:
            continue
        theindex = face_index[i][0]
        me_prior = prior[theindex, :]
        lmk_pred = predict_lmk[theindex, :]
        prior_h = me_prior[2] - me_prior[0]
        prior_w = me_prior[3] - me_prior[1]
        prior_h_center = (me_prior[2] + me_prior[0]) / 2
        prior_w_center = (me_prior[3] + me_prior[1]) / 2
        lmk_decode = np.zeros((10))
        for j in [0, 2, 4, 6, 8]:
            lmk_decode[j] = lmk_pred[j] * 0.1 * prior_w + prior_h_center
        for j in [1, 3, 5, 7, 9]:
            lmk_decode[j] = lmk_pred[j] * 0.1 * prior_h + prior_w_center
        if is_bbox_normalized:
            lmk_decode = lmk_decode * np.array([
                image_h, image_w, image_h, image_w, image_h, image_w, image_h,
                image_w, image_h, image_w
            ])
        xywh_res.append(lmk_decode)
    return np.asarray(xywh_res)


def post_processing(image, lmk_decode, hat_path, beard_path, glasses_path):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    p_left, p_right, p_up, p_bottom, h_xmax, h_ymax = [0] * 6
    h_xmin, h_ymin = 10000, 10000
    # Add beard on the face
    if beard_path is not None:
        image = face_makeup_main.add_beard(image, lmk_decode, beard_path)
    # Add glasses on the face
    if glasses_path is not None:
        image = face_makeup_main.add_glasses(image, lmk_decode, glasses_path)
    # Add hat on the face
    if hat_path is not None:
        image, p_left, p_right, p_up, p_bottom, h_xmin, h_ymin, h_xmax, h_ymax = face_makeup_main.add_hat(
            image, lmk_decode, hat_path)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print('-----------  Post Processing Success -----------')
    return image, p_left, p_right, p_up, p_bottom, h_xmin, h_ymin, h_xmax, h_ymax


def trans_lmk(image, lmk_results, beard_file, glasses_file, hat_file):
    p_left, p_right, p_up, p_bottom, h_xmax, h_ymax = [0] * 6
    h_xmin, h_ymin = 10000, 10000
    if lmk_results.shape[0] == 0:
        return image, p_left, p_right, p_up, p_bottom, h_xmin, h_ymin, h_xmax, h_ymax, 0
    for lmk_decode in lmk_results:

        x1, y1, x2, y2 = lmk_decode[0], lmk_decode[1], lmk_decode[
            2], lmk_decode[3]
        x4, y4, x5, y5 = lmk_decode[6], lmk_decode[7], lmk_decode[
            8], lmk_decode[9]
        # Refine the order of keypoint 
        if x1 > x2:
            lmk_decode[0], lmk_decode[1], lmk_decode[2], lmk_decode[
                3] = lmk_decode[2], lmk_decode[3], lmk_decode[0], lmk_decode[1]
        if x4 < x5:
            lmk_decode[6], lmk_decode[7], lmk_decode[8], lmk_decode[
                9] = lmk_decode[8], lmk_decode[9], lmk_decode[6], lmk_decode[7]
        # Add decoration to the face
        image, p_left_temp, p_right_temp, p_up_temp, p_bottom_temp, h_xmin_temp, h_ymin_temp, h_xmax_temp, h_ymax_temp = post_processing(
            image, lmk_decode, hat_file, beard_file, glasses_file)

        p_left = max(p_left, p_left_temp)
        p_right = max(p_right, p_right_temp)
        p_up = max(p_up, p_up_temp)
        p_bottom = max(p_bottom, p_bottom_temp)
        h_xmin = min(h_xmin, h_xmin_temp)
        h_ymin = min(h_ymin, h_ymin_temp)
        h_xmax = max(h_xmax, h_xmax_temp)
        h_ymax = max(h_ymax, h_ymax_temp)

    return image, p_left, p_right, p_up, p_bottom, h_xmin, h_ymin, h_xmax, h_ymax, 1
