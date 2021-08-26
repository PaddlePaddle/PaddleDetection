# -*- coding: utf-8 -*-
# *******************************************************************************
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
# *******************************************************************************
"""

Authors: yanglijuan04@baidu.com
Date:     2021/8/4 下午7:20
"""
import sys
import json
import logging
import numpy as np

from ppdet.utils.logger import setup_logger
logger = setup_logger('sniper_params_stats')

def get_default_params(architecture):
    """get_default_params"""
    if architecture == "FasterRCNN":
        anchor_range = np.array([64., 256.])  # for frcnn-fpn
        # anchor_range = np.array([16., 373.])  # for yolov3
        # anchor_range = np.array([32., 373.])  # for yolov3
        default_crop_size = 1536  # mod 32 for frcnn-fpn
        default_stride = 352
    else:
        raise NotImplementedError

    return anchor_range, default_crop_size, default_stride


def get_box_ratios(anno_file):
    """
    get_size_ratios
    :param anno_file: coco anno flile
    :return: size_ratio: (box_long_size / pic_long_size)
    """
    coco_dict = json.load(open(anno_file))
    image_list = coco_dict['images']
    anno_list = coco_dict['annotations']

    image_id2hw = {}
    for im_dict in image_list:
        im_id = im_dict['id']
        h, w = im_dict['height'], im_dict['width']
        image_id2hw[im_id] = (h, w)

    box_ratios = []
    for a_dict in anno_list:
        im_id = a_dict['image_id']
        im_h, im_w = image_id2hw[im_id]
        bbox = a_dict['bbox']
        x1, y1, w, h = bbox
        pic_long = max(im_h, im_w)
        box_long = max(w, h)
        box_ratios.append(box_long / pic_long)

    return np.array(box_ratios)


def get_target_size_and_valid_box_ratios(anchor_range, box_ratio_p2, box_ratio_p98):
    """get_scale_and_ratios"""
    anchor_better_low, anchor_better_high = anchor_range  # (60., 512.)
    anchor_center = np.sqrt(anchor_better_high * anchor_better_low)

    anchor_log_range = np.log10(anchor_better_high) - np.log10(anchor_better_low)
    box_ratio_log_range = np.log10(box_ratio_p98) - np.log10(box_ratio_p2)
    logger.info("anchor_log_range:{}, box_ratio_log_range:{}".format(anchor_log_range, box_ratio_log_range))

    box_cut_num = int(np.ceil(box_ratio_log_range / anchor_log_range))
    box_ratio_log_window = box_ratio_log_range / box_cut_num
    logger.info("box_cut_num:{}, box_ratio_log_window:{}".format(box_cut_num, box_ratio_log_window))

    image_target_sizes = []
    valid_ratios = []
    for i in range(box_cut_num):
        # # method1: align center
        # box_ratio_log_center = np.log10(p2) + 0.5 * box_ratio_log_window + i * box_ratio_log_window
        # box_ratio_center = np.power(10, box_ratio_log_center)
        # scale = anchor_center / box_ratio_center
        # method2: align left low
        box_ratio_low = np.power(10, np.log10(box_ratio_p2) + i * box_ratio_log_window)
        image_target_size = anchor_better_low / box_ratio_low

        image_target_sizes.append(int(image_target_size))
        valid_ratio = anchor_range / image_target_size
        valid_ratios.append(valid_ratio.tolist())

        logger.info("Box cut {}".format(i))
        logger.info("box_ratio_low: {}".format(box_ratio_low))
        logger.info("image_target_size: {}".format(image_target_size))
        logger.info("valid_ratio: {}".format(valid_ratio))

    return image_target_sizes, valid_ratios


def get_valid_ranges(valid_ratios):
    """
    get_valid_box_ratios_range
    :param valid_ratios:
    :return:
    """
    valid_ranges = []
    if len(valid_ratios) == 1:
        valid_ranges.append([-1, -1])
    else:
        for i, vratio in enumerate(valid_ratios):
            if i == 0:
                valid_ranges.append([-1, vratio[1]])
            elif i == len(valid_ratios) - 1:
                valid_ranges.append([vratio[0], -1])
            else:
                valid_ranges.append(vratio)
    return valid_ranges


def get_percentile(a_array, low_percent, high_percent):
    """
    get_percentile
    :param low_percent:
    :param high_percent:
    :return:
    """
    array_p0 = min(a_array)
    array_p100 = max(a_array)
    array_plow = np.percentile(a_array, low_percent)
    array_phigh = np.percentile(a_array, high_percent)
    logger.info(
        "array_percentile(0): {},array_percentile low({}): {}, "
        "array_percentile high({}): {}, array_percentile 100: {}".format(
            array_p0, low_percent, array_plow, high_percent, array_phigh, array_p100))
    return array_plow, array_phigh


def sniper_anno_stats(architecture, anno_file):
    """
    sniper_anno_stats
    :param anno_file:
    :return:
    """

    anchor_range, default_crop_size, default_stride = get_default_params(architecture)

    box_ratios = get_box_ratios(anno_file)

    box_ratio_p8, box_ratio_p92 = get_percentile(box_ratios, 8, 92)

    image_target_sizes, valid_box_ratios = get_target_size_and_valid_box_ratios(anchor_range, box_ratio_p8, box_ratio_p92)

    valid_ranges = get_valid_ranges(valid_box_ratios)

    crop_size = min(default_crop_size, min([item for item in image_target_sizes]))
    crop_size = int(np.ceil(crop_size / 32.) * 32.)
    crop_stride = max(min(default_stride, crop_size), crop_size - default_stride)
    logger.info("Result".center(100, '-'))
    logger.info("image_target_sizes: {}".format(image_target_sizes))
    logger.info("valid_box_ratio_ranges: {}".format(valid_ranges))
    logger.info("chip_target_size: {}, chip_target_stride: {}".format(crop_size, crop_stride))

    return {
        "image_target_sizes": image_target_sizes,
        "valid_box_ratio_ranges": valid_ranges,
        "chip_target_size": crop_size,
        "chip_target_stride": crop_stride
    }

if __name__=="__main__":
    architecture, anno_file = sys.argv[1], sys.argv[2]
    sniper_anno_stats(architecture, anno_file)
