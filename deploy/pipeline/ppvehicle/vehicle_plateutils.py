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

import argparse
import os
import sys
import platform
import cv2
import numpy as np
import paddle
from PIL import Image, ImageDraw, ImageFont
import math
from paddle import inference
import time
import ast


def create_predictor(args, cfg, mode):
    if mode == "det":
        model_dir = cfg['det_model_dir']
    else:
        model_dir = cfg['rec_model_dir']

    if model_dir is None:
        print("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)

    model_file_path = model_dir + "/inference.pdmodel"
    params_file_path = model_dir + "/inference.pdiparams"
    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(model_file_path))
    if not os.path.exists(params_file_path):
        raise ValueError("not find params file path {}".format(
            params_file_path))

    config = inference.Config(model_file_path, params_file_path)

    batch_size = 1

    if args.device == "GPU":
        gpu_id = get_infer_gpuid()
        if gpu_id is None:
            print(
                "GPU is not found in current device by nvidia-smi. Please check your device or ignore it if run on jetson."
            )
        config.enable_use_gpu(500, 0)

        precision_map = {
            'trt_int8': inference.PrecisionType.Int8,
            'trt_fp32': inference.PrecisionType.Float32,
            'trt_fp16': inference.PrecisionType.Half
        }
        min_subgraph_size = 15
        if args.run_mode in precision_map.keys():
            config.enable_tensorrt_engine(
                workspace_size=(1 << 25) * batch_size,
                max_batch_size=batch_size,
                min_subgraph_size=min_subgraph_size,
                precision_mode=precision_map[args.run_mode])
            use_dynamic_shape = True

            if mode == "det":
                min_input_shape = {
                    "x": [1, 3, 50, 50],
                    "conv2d_92.tmp_0": [1, 120, 20, 20],
                    "conv2d_91.tmp_0": [1, 24, 10, 10],
                    "conv2d_59.tmp_0": [1, 96, 20, 20],
                    "nearest_interp_v2_1.tmp_0": [1, 256, 10, 10],
                    "nearest_interp_v2_2.tmp_0": [1, 256, 20, 20],
                    "conv2d_124.tmp_0": [1, 256, 20, 20],
                    "nearest_interp_v2_3.tmp_0": [1, 64, 20, 20],
                    "nearest_interp_v2_4.tmp_0": [1, 64, 20, 20],
                    "nearest_interp_v2_5.tmp_0": [1, 64, 20, 20],
                    "elementwise_add_7": [1, 56, 2, 2],
                    "nearest_interp_v2_0.tmp_0": [1, 256, 2, 2]
                }
                max_input_shape = {
                    "x": [1, 3, 1536, 1536],
                    "conv2d_92.tmp_0": [1, 120, 400, 400],
                    "conv2d_91.tmp_0": [1, 24, 200, 200],
                    "conv2d_59.tmp_0": [1, 96, 400, 400],
                    "nearest_interp_v2_1.tmp_0": [1, 256, 200, 200],
                    "conv2d_124.tmp_0": [1, 256, 400, 400],
                    "nearest_interp_v2_2.tmp_0": [1, 256, 400, 400],
                    "nearest_interp_v2_3.tmp_0": [1, 64, 400, 400],
                    "nearest_interp_v2_4.tmp_0": [1, 64, 400, 400],
                    "nearest_interp_v2_5.tmp_0": [1, 64, 400, 400],
                    "elementwise_add_7": [1, 56, 400, 400],
                    "nearest_interp_v2_0.tmp_0": [1, 256, 400, 400]
                }
                opt_input_shape = {
                    "x": [1, 3, 640, 640],
                    "conv2d_92.tmp_0": [1, 120, 160, 160],
                    "conv2d_91.tmp_0": [1, 24, 80, 80],
                    "conv2d_59.tmp_0": [1, 96, 160, 160],
                    "nearest_interp_v2_1.tmp_0": [1, 256, 80, 80],
                    "nearest_interp_v2_2.tmp_0": [1, 256, 160, 160],
                    "conv2d_124.tmp_0": [1, 256, 160, 160],
                    "nearest_interp_v2_3.tmp_0": [1, 64, 160, 160],
                    "nearest_interp_v2_4.tmp_0": [1, 64, 160, 160],
                    "nearest_interp_v2_5.tmp_0": [1, 64, 160, 160],
                    "elementwise_add_7": [1, 56, 40, 40],
                    "nearest_interp_v2_0.tmp_0": [1, 256, 40, 40]
                }
                min_pact_shape = {
                    "nearest_interp_v2_26.tmp_0": [1, 256, 20, 20],
                    "nearest_interp_v2_27.tmp_0": [1, 64, 20, 20],
                    "nearest_interp_v2_28.tmp_0": [1, 64, 20, 20],
                    "nearest_interp_v2_29.tmp_0": [1, 64, 20, 20]
                }
                max_pact_shape = {
                    "nearest_interp_v2_26.tmp_0": [1, 256, 400, 400],
                    "nearest_interp_v2_27.tmp_0": [1, 64, 400, 400],
                    "nearest_interp_v2_28.tmp_0": [1, 64, 400, 400],
                    "nearest_interp_v2_29.tmp_0": [1, 64, 400, 400]
                }
                opt_pact_shape = {
                    "nearest_interp_v2_26.tmp_0": [1, 256, 160, 160],
                    "nearest_interp_v2_27.tmp_0": [1, 64, 160, 160],
                    "nearest_interp_v2_28.tmp_0": [1, 64, 160, 160],
                    "nearest_interp_v2_29.tmp_0": [1, 64, 160, 160]
                }
                min_input_shape.update(min_pact_shape)
                max_input_shape.update(max_pact_shape)
                opt_input_shape.update(opt_pact_shape)
            elif mode == "rec":
                imgH = int(cfg['rec_image_shape'][-2])
                min_input_shape = {"x": [1, 3, imgH, 10]}
                max_input_shape = {"x": [batch_size, 3, imgH, 2304]}
                opt_input_shape = {"x": [batch_size, 3, imgH, 320]}
                config.exp_disable_tensorrt_ops(["transpose2"])
            elif mode == "cls":
                min_input_shape = {"x": [1, 3, 48, 10]}
                max_input_shape = {"x": [batch_size, 3, 48, 1024]}
                opt_input_shape = {"x": [batch_size, 3, 48, 320]}
            else:
                use_dynamic_shape = False
            if use_dynamic_shape:
                config.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape)

    else:
        config.disable_gpu()
        if hasattr(args, "cpu_threads"):
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        else:
            # default cpu threads as 10
            config.set_cpu_math_library_num_threads(10)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.run_mode == "fp16":
                config.enable_mkldnn_bfloat16()
    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()
    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    config.delete_pass("matmul_transpose_reshape_fuse_pass")
    if mode == 'table':
        config.delete_pass("fc_fuse_pass")  # not supported for table
    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)

    # create predictor
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    output_tensors = get_output_tensors(cfg, mode, predictor)
    return predictor, input_tensor, output_tensors, config


def get_output_tensors(cfg, mode, predictor):
    output_names = predictor.get_output_names()
    output_tensors = []
    output_name = 'softmax_0.tmp_0'
    if output_name in output_names:
        return [predictor.get_output_handle(output_name)]
    else:
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
    return output_tensors


def get_infer_gpuid():
    sysstr = platform.system()
    if sysstr == "Windows":
        return 0

    if not paddle.device.is_compiled_with_rocm():
        cmd = "env | grep CUDA_VISIBLE_DEVICES"
    else:
        cmd = "env | grep HIP_VISIBLE_DEVICES"
    env_cuda = os.popen(cmd).readlines()
    if len(env_cuda) == 0:
        return 0
    else:
        gpu_id = env_cuda[0].strip().split("=")[1]
        return int(gpu_id[0])


def draw_e2e_res(dt_boxes, strs, img_path):
    src_im = cv2.imread(img_path)
    for box, str in zip(dt_boxes, strs):
        box = box.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        cv2.putText(
            src_im,
            str,
            org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.7,
            color=(0, 255, 0),
            thickness=1)
    return src_im


def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im


def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img


def draw_ocr(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="./doc/fonts/simfang.ttf"):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or
                                   math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    if txts is not None:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=600,
            threshold=drop_score,
            font_path=font_path)
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./doc/simfang.ttf"):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)


def base64_to_cv2(b64str):
    import base64
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def check_gpu(use_gpu):
    if use_gpu and not paddle.is_compiled_with_cuda():
        use_gpu = False
    return use_gpu


if __name__ == '__main__':
    pass
