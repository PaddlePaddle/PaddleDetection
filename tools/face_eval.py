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
import sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import paddle.fluid as fluid
import numpy as np
import cv2
from collections import OrderedDict

import ppdet.utils.checkpoint as checkpoint
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.widerface_eval_utils import get_shrink, bbox_vote, \
    save_widerface_bboxes, save_fddb_bboxes, to_chw_bgr
from ppdet.core.workspace import load_config, merge_config, create

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def face_img_process(image,
                     mean=[104., 117., 123.],
                     std=[127.502231, 127.502231, 127.502231]):
    img = np.array(image)
    img = to_chw_bgr(img)
    img = img.astype('float32')
    img -= np.array(mean)[:, np.newaxis, np.newaxis].astype('float32')
    img /= np.array(std)[:, np.newaxis, np.newaxis].astype('float32')
    img = [img]
    img = np.array(img)
    return img


def face_eval_run(exe,
                  compile_program,
                  fetches,
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
            det0 = detect_face(exe, compile_program, fetches, image, shrink)
            det1 = flip_test(exe, compile_program, fetches, image, shrink)
            [det2, det3] = multi_scale_test(exe, compile_program, fetches,
                                            image, max_shrink)
            det4 = multi_scale_test_pyramid(exe, compile_program, fetches,
                                            image, max_shrink)
            det = np.row_stack((det0, det1, det2, det3, det4))
            dets = bbox_vote(det)
        else:
            dets = detect_face(exe, compile_program, fetches, image, 1)
        if eval_mode == 'widerface':
            save_widerface_bboxes(image_path, dets, pred_dir)
        else:
            dets_dist[im_path] = dets
        if iter_id % 100 == 0:
            logger.info('Test iter {}'.format(iter_id))
    if eval_mode == 'fddb':
        save_fddb_bboxes(dets_dist, pred_dir)
    logger.info("Finish evaluation.")


def detect_face(exe, compile_program, fetches, image, shrink):
    image_shape = [3, image.shape[0], image.shape[1]]
    if shrink != 1:
        h, w = int(image_shape[1] * shrink), int(image_shape[2] * shrink)
        image = cv2.resize(image, (w, h))
        image_shape = [3, h, w]

    img = face_img_process(image)
    detection, = exe.run(compile_program,
                         feed={'image': img},
                         fetch_list=[fetches['bbox']],
                         return_numpy=False)
    detection = np.array(detection)
    # layout: xmin, ymin, xmax. ymax, score
    if np.prod(detection.shape) == 1:
        logger.info("No face detected")
        return np.array([[0, 0, 0, 0, 0]])
    det_conf = detection[:, 1]
    det_xmin = image_shape[2] * detection[:, 2] / shrink
    det_ymin = image_shape[1] * detection[:, 3] / shrink
    det_xmax = image_shape[2] * detection[:, 4] / shrink
    det_ymax = image_shape[1] * detection[:, 5] / shrink

    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))
    return det


def flip_test(exe, compile_program, fetches, image, shrink):
    img = cv2.flip(image, 1)
    det_f = detect_face(exe, compile_program, fetches, img, shrink)
    det_t = np.zeros(det_f.shape)
    img_width = image.shape[1]
    det_t[:, 0] = img_width - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = img_width - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def multi_scale_test(exe, compile_program, fetches, image, max_shrink):
    # Shrink detecting is only used to detect big faces
    st = 0.5 if max_shrink >= 0.75 else 0.5 * max_shrink
    det_s = detect_face(exe, compile_program, fetches, image, st)
    index = np.where(
        np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1)
        > 30)[0]
    det_s = det_s[index, :]
    # Enlarge one times
    bt = min(2, max_shrink) if max_shrink > 1 else (st + max_shrink) / 2
    det_b = detect_face(exe, compile_program, fetches, image, bt)

    # Enlarge small image x times for small faces
    if max_shrink > 2:
        bt *= 2
        while bt < max_shrink:
            det_b = np.row_stack((det_b, detect_face(exe, compile_program,
                                                     fetches, image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(exe, compile_program, fetches,
                                                 image, max_shrink)))

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


def multi_scale_test_pyramid(exe, compile_program, fetches, image, max_shrink):
    # Use image pyramids to detect faces
    det_b = detect_face(exe, compile_program, fetches, image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [0.75, 1.25, 1.5, 1.75]
    for i in range(len(st)):
        if st[i] <= max_shrink:
            det_temp = detect_face(exe, compile_program, fetches, image, st[i])
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


def main():
    """
    Main evaluate function
    """
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)
    check_version()

    main_arch = cfg.architecture

    # define executor
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # build program
    model = create(main_arch)
    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs_def = cfg['EvalReader']['inputs_def']
            inputs_def['use_dataloader'] = False
            feed_vars, _ = model.build_inputs(**inputs_def)
            fetches = model.eval(feed_vars)

    eval_prog = eval_prog.clone(True)

    # load model
    exe.run(startup_prog)
    if 'weights' in cfg:
        checkpoint.load_params(exe, eval_prog, cfg.weights)

    assert cfg.metric in ['WIDERFACE'], \
            "unknown metric type {}".format(cfg.metric)

    dataset = cfg['EvalReader']['dataset']

    annotation_file = dataset.get_anno()
    dataset_dir = dataset.dataset_dir
    image_dir = os.path.join(
        dataset_dir,
        dataset.image_dir) if FLAGS.eval_mode == 'widerface' else dataset_dir

    pred_dir = FLAGS.output_eval if FLAGS.output_eval else 'output/pred'
    face_eval_run(
        exe,
        eval_prog,
        fetches,
        image_dir,
        annotation_file,
        pred_dir=pred_dir,
        eval_mode=FLAGS.eval_mode,
        multi_scale=FLAGS.multi_scale)


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "-f",
        "--output_eval",
        default=None,
        type=str,
        help="Evaluation file directory, default is current directory.")
    parser.add_argument(
        "-e",
        "--eval_mode",
        default="widerface",
        type=str,
        help="Evaluation mode, include `widerface` and `fddb`, default is `widerface`."
    )
    parser.add_argument(
        "--multi_scale",
        action='store_true',
        default=False,
        help="If True it will select `multi_scale` evaluation. Default is `False`, it will select `single-scale` evaluation."
    )
    FLAGS = parser.parse_args()
    main()
