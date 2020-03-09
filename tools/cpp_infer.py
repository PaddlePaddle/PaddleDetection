import os
import time

import numpy as np
from PIL import Image, ImageDraw

import paddle.fluid as fluid

import argparse
import cv2
import yaml
import copy

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

precision_map = {
    'trt_int8': fluid.core.AnalysisConfig.Precision.Int8,
    'trt_fp32': fluid.core.AnalysisConfig.Precision.Float32,
    'trt_fp16': fluid.core.AnalysisConfig.Precision.Half
}


def create_config(model_path, mode='fluid', batch_size=1, min_subgraph_size=3):
    model_file = os.path.join(model_path, '__model__')
    params_file = os.path.join(model_path, '__params__')
    config = fluid.core.AnalysisConfig(model_file, params_file)
    config.enable_use_gpu(100, 0)
    logger.info('min_subgraph_size = %d.' % (min_subgraph_size))

    if mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[mode],
            use_static=False,
            use_calib_mode=mode == 'trt_int8')
        logger.info('Run inference by {}.'.format(mode))
    elif mode == 'fluid':
        logger.info('Run inference by Fluid FP32.')
    else:
        logger.fatal(
            'Wrong mode, only support trt_int8, trt_fp32, trt_fp16, fluid.')
    return config


def offset_to_lengths(lod):
    offset = lod[0]
    lengths = [offset[i + 1] - offset[i] for i in range(len(offset) - 1)]
    return [lengths]


def DecodeImage(im_path):
    assert os.path.exists(im_path), "Image path {} can not be found".format(
        im_path)
    with open(im_path, 'rb') as f:
        im = f.read()
    data = np.frombuffer(im, dtype='uint8')
    im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def get_extra_info(im, arch, shape, scale):
    info = []
    input_shape = []
    im_shape = []
    logger.info('The architecture is {}'.format(arch))
    if 'YOLO' in arch:
        im_size = np.array([shape[:2]]).astype('int32')
        logger.info('Extra info: im_size')
        info.append(im_size)
    elif 'SSD' in arch:
        im_shape = np.array([shape[:2]]).astype('int32')
        logger.info('Extra info: im_shape')
        info.append([im_shape])
    elif 'RetinaNet' in arch:
        input_shape.extend(im.shape[2:])
        im_info = np.array([input_shape + [scale]]).astype('float32')
        logger.info('Extra info: im_info')
        info.append(im_info)
    elif 'RCNN' in arch:
        input_shape.extend(im.shape[2:])
        im_shape.extend(shape[:2])
        im_info = np.array([input_shape + [scale]]).astype('float32')
        im_shape = np.array([im_shape + [1.]]).astype('float32')
        logger.info('Extra info: im_info, im_shape')
        info.append(im_info)
        info.append(im_shape)
    else:
        logger.error(
            "Unsupported arch: {}, expect YOLO, SSD, RetinaNet and RCNN".format(
                arch))
    return info


class Resize(object):
    def __init__(self,
                 target_size,
                 max_size=0,
                 interp=cv2.INTER_LINEAR,
                 use_cv2=True):
        super(Resize, self).__init__()
        self.target_size = target_size
        self.max_size = max_size
        self.interp = interp
        self.use_cv2 = use_cv2

    def __call__(self, im, use_trt=False):
        origin_shape = im.shape[:2]
        im_c = im.shape[2]
        if self.max_size != 0:
            im_size_min = np.min(origin_shape[0:2])
            im_size_max = np.max(origin_shape[0:2])
            im_scale = float(self.target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > self.max_size:
                im_scale = float(self.max_size) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
            resize_w = int(im_scale_x * float(origin_shape[1]))
            resize_h = int(im_scale_y * float(origin_shape[0]))
        else:
            im_scale_x = float(self.target_size) / float(origin_shape[1])
            im_scale_y = float(self.target_size) / float(origin_shape[0])
            resize_w = self.target_size
            resize_h = self.target_size
        if self.use_cv2:
            im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
        else:
            if self.max_size != 0:
                raise TypeError(
                    'If you set max_size to cap the maximum size of image,'
                    'please set use_cv2 to True to resize the image.')
            im = im.astype('uint8')
            im = Image.fromarray(im)
            im = im.resize((int(resize_w), int(resize_h)), self.interp)
            im = np.array(im)
        # padding im
        if self.max_size != 0 and use_trt:
            logger.warning('Due to the limitation of tensorRT, padding the'
                           'image shape to {} * {}'.format(self.max_size,
                                                           self.max_size))
            padding_im = np.zeros(
                (self.max_size, self.max_size, im_c), dtype=np.float32)
            im_h, im_w = im.shape[:2]
            padding_im[:im_h, :im_w, :] = im
            im = padding_im
        return im, im_scale_x


class Normalize(object):
    def __init__(self, mean, std, is_scale=True, is_channel_first=False):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first

    def __call__(self, im):
        im = im.astype(np.float32, copy=False)
        if self.is_channel_first:
            mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
            std = np.array(self.std)[:, np.newaxis, np.newaxis]
        else:
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
        if self.is_scale:
            im = im / 255.0
        im -= mean
        im /= std
        return im


class Permute(object):
    def __init__(self, to_bgr=False, channel_first=True):
        self.to_bgr = to_bgr
        self.channel_first = channel_first

    def __call__(self, im):
        if self.channel_first:
            im = im.transpose((2, 0, 1)).copy()
        if self.to_bgr:
            im = im[[2, 1, 0], :, :]
        return im


class PadStride(object):
    def __init__(self, stride=0):
        assert stride >= 0, "Unsupported stride: {}, the stride in PadStride must be greater or equal to 0".format(
            stride)
        self.coarsest_stride = stride

    def __call__(self, im):
        coarsest_stride = self.coarsest_stride
        if coarsest_stride == 0:
            return im
        im_c, im_h, im_w = im.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        return padding_im


def Preprocess(img_path, arch, config):
    img = DecodeImage(img_path)
    orig_shape = img.shape
    scale = 1.
    data = []
    data_config = copy.deepcopy(config)
    for data_aug_conf in data_config:
        obj = data_aug_conf.pop('type')
        preprocess = eval(obj)(**data_aug_conf)
        if obj == 'Resize':
            img, scale = preprocess(img, arch)
        else:
            img = preprocess(img)

    img = img[np.newaxis, :]  # N, C, H, W
    data.append(img)
    extra_info = get_extra_info(img, arch, orig_shape, scale)
    data += extra_info
    return data


def default_category_info(metric, with_background):
    logger.info('Use default label from {} dataset'.format(metric))
    voc_map = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }

    clsid2catid = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 13,
        13: 14,
        14: 15,
        15: 16,
        16: 17,
        17: 18,
        18: 19,
        19: 20,
        20: 21,
        21: 22,
        22: 23,
        23: 24,
        24: 25,
        25: 27,
        26: 28,
        27: 31,
        28: 32,
        29: 33,
        30: 34,
        31: 35,
        32: 36,
        33: 37,
        34: 38,
        35: 39,
        36: 40,
        37: 41,
        38: 42,
        39: 43,
        40: 44,
        41: 46,
        42: 47,
        43: 48,
        44: 49,
        45: 50,
        46: 51,
        47: 52,
        48: 53,
        49: 54,
        50: 55,
        51: 56,
        52: 57,
        53: 58,
        54: 59,
        55: 60,
        56: 61,
        57: 62,
        58: 63,
        59: 64,
        60: 65,
        61: 67,
        62: 70,
        63: 72,
        64: 73,
        65: 74,
        66: 75,
        67: 76,
        68: 77,
        69: 78,
        70: 79,
        71: 80,
        72: 81,
        73: 82,
        74: 84,
        75: 85,
        76: 86,
        77: 87,
        78: 88,
        79: 89,
        80: 90
    }

    coco_map = {
        0: 'background',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        27: 'backpack',
        28: 'umbrella',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
        34: 'frisbee',
        35: 'skis',
        36: 'snowboard',
        37: 'sports ball',
        38: 'kite',
        39: 'baseball bat',
        40: 'baseball glove',
        41: 'skateboard',
        42: 'surfboard',
        43: 'tennis racket',
        44: 'bottle',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        67: 'dining table',
        70: 'toilet',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink',
        82: 'refrigerator',
        84: 'book',
        85: 'clock',
        86: 'vase',
        87: 'scissors',
        88: 'teddy bear',
        89: 'hair drier',
        90: 'toothbrush'
    }
    catid2name = coco_map if metric == 'COCO' else voc_map
    if metric == 'VOC':
        offset = 0 if with_background else 1
        clsid2catid = {i: i + offset for i in range(len(voc_map) - offset)}
    elif not with_background:
        clsid2catid = {k - 1: v for k, v in clsid2catid.items()}

    return clsid2catid, catid2name


def get_category_info(with_background=True, metric='COCO', label_list=None):
    if label_list is None:
        return default_category_info(metric, with_background)
    logger.info("Load categories from {}".format(label_list))
    cats = []
    with open(label_list) as f:
        for line in f.readlines():
            cats.append(line.strip())

    if cats[0] != 'background' and with_background:
        cats.insert(0, 'background')
    if cats[0] == 'background' and not with_background:
        cats = cats[1:]
    clsid2catid = {i: i for i in range(len(cats))}
    catid2name = {i: name for i, name in enumerate(cats)}
    return clsid2catid, catid2name


def bbox2out(results, clsid2catid, is_bbox_normalized=False):
    """
    Args:
        results: request a dict, should include: `bbox`, `im_id`,
                 if is_bbox_normalized=True, also need `im_shape`.
        clsid2catid: class id to category id map of COCO2017 dataset.
        is_bbox_normalized: whether or not bbox is normalized.
    """
    xywh_res = []
    for t in results:
        bboxes = t['bbox'][0]
        lengths = t['bbox'][1][0]
        im_ids = np.array(t['im_id'][0]).flatten()
        if bboxes.shape == (1, 1) or bboxes is None:
            continue

        k = 0
        for i in range(len(lengths)):
            num = lengths[i]
            im_id = int(im_ids[i])
            for j in range(num):
                dt = bboxes[k]
                clsid, score, xmin, ymin, xmax, ymax = dt.tolist()
                catid = (clsid2catid[int(clsid)])

                if is_bbox_normalized:
                    xmin, ymin, xmax, ymax = \
                            clip_bbox([xmin, ymin, xmax, ymax])
                    w = xmax - xmin
                    h = ymax - ymin
                    im_shape = t['im_shape'][0][i].tolist()
                    im_height, im_width = int(im_shape[0]), int(im_shape[1])
                    xmin *= im_width
                    ymin *= im_height
                    w *= im_width
                    h *= im_height
                else:
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1

                bbox = [xmin, ymin, w, h]
                coco_res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'bbox': bbox,
                    'score': score
                }
                xywh_res.append(coco_res)
                k += 1
    return xywh_res


def draw_bbox(image, im_id, catid2name, bboxes, threshold):
    """
    draw bbox on image
    """
    draw = ImageDraw.Draw(image)

    catid2color = {}
    color_list = np.array([
        0.000,
        0.447,
        0.741,
        0.850,
        0.325,
        0.098,
        0.929,
        0.694,
        0.125,
        0.494,
        0.184,
        0.556,
        0.466,
        0.674,
        0.188,
        0.301,
        0.745,
        0.933,
        0.635,
        0.078,
        0.184,
        0.300,
        0.300,
        0.300,
        0.600,
        0.600,
        0.600,
        1.000,
        0.000,
        0.000,
        1.000,
        0.500,
        0.000,
        0.749,
        0.749,
        0.000,
        0.000,
        1.000,
        0.000,
        0.000,
        0.000,
        1.000,
        0.667,
        0.000,
        1.000,
        0.333,
        0.333,
        0.000,
        0.333,
        0.667,
        0.000,
        0.333,
        1.000,
        0.000,
        0.667,
        0.333,
        0.000,
        0.667,
        0.667,
        0.000,
        0.667,
        1.000,
        0.000,
        1.000,
        0.333,
        0.000,
        1.000,
        0.667,
        0.000,
        1.000,
        1.000,
        0.000,
        0.000,
        0.333,
        0.500,
        0.000,
        0.667,
        0.500,
        0.000,
        1.000,
        0.500,
        0.333,
        0.000,
        0.500,
        0.333,
        0.333,
        0.500,
        0.333,
        0.667,
        0.500,
        0.333,
        1.000,
        0.500,
        0.667,
        0.000,
        0.500,
        0.667,
        0.333,
        0.500,
        0.667,
        0.667,
        0.500,
        0.667,
        1.000,
        0.500,
        1.000,
        0.000,
        0.500,
        1.000,
        0.333,
        0.500,
        1.000,
        0.667,
        0.500,
        1.000,
        1.000,
        0.500,
        0.000,
        0.333,
        1.000,
    ]).astype(np.float32).reshape((-1, 3)) * 255
    for dt in np.array(bboxes):
        if im_id != dt['image_id']:
            continue
        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        if score < threshold:
            continue

        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h

        if catid not in catid2color:
            idx = np.random.randint(len(color_list))
            catid2color[catid] = color_list[idx]
        color = tuple(catid2color[catid])

        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=2,
            fill=color)

        # draw label
        text = "{} {:.2f}".format(catid2name[catid], score)
        tw, th = draw.textsize(text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

    return image


def get_bbox_result(outputs, result, conf, clsid2catid):
    is_bbox_normalized = True if 'SSD' in conf['arch'] else False

    out = outputs[-1]
    lod = out.lod() if conf['use_python_inference'] else out.lod
    lengths = offset_to_lengths(lod)
    np_data = np.array(out) if conf['use_python_inference'] else out.as_ndarray(
    )
    result['bbox'] = (np_data, lengths)
    result['im_id'] = np.array([[0]])

    bbox_results = bbox2out([result], clsid2catid, is_bbox_normalized)
    return bbox_results


def visualize(bbox_results, catid2name):
    image = Image.open(FLAGS.infer_img).convert('RGB')
    image = draw_bbox(image, 0, catid2name, bbox_results, 0.5)
    image_path = os.path.split(FLAGS.infer_img)[-1]
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    out_path = os.path.join(FLAGS.output_dir, image_path)
    image.save(out_path, quality=95)
    logger.info('Save visualize result to {}'.format(out_path))


def infer():
    model_path = FLAGS.model_path
    config_path = FLAGS.config_path
    res = {}
    assert model_path is not None, "Model path: {} does not exist!".format(
        model_path)
    assert config_path is not None, "Config path: {} does not exist!".format(
        config_path)
    with open(config_path) as f:
        conf = yaml.safe_load(f)

    img_data = Preprocess(FLAGS.infer_img, conf['arch'], conf['Preprocess'])
    if 'SSD' in conf['arch']:
        img_data, res['im_shape'] = img_data
        img_data = [img_data]

    if conf['use_python_inference']:
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        infer_prog, feed_var_names, fetch_targets = fluid.io.load_inference_model(
            dirname=model_path,
            executor=exe,
            model_filename='__model__',
            params_filename='__params__')
        data_dict = {k: v for k, v in zip(feed_var_names, img_data)}
    else:
        inputs = [fluid.core.PaddleTensor(d.copy()) for d in img_data]
        config = create_config(
            model_path,
            mode=conf['mode'],
            min_subgraph_size=conf['min_subgraph_size'])
        predict = fluid.core.create_paddle_predictor(config)

    logger.info('warmup...')
    for i in range(10):
        if conf['use_python_inference']:
            outs = exe.run(infer_prog,
                           feed=data_dict,
                           fetch_list=fetch_targets,
                           return_numpy=False)
        else:
            outs = predict.run(inputs)

    cnt = 100
    logger.info('run benchmark...')
    t1 = time.time()
    for i in range(cnt):
        if conf['use_python_inference']:
            outs = exe.run(infer_prog,
                           feed=data_dict,
                           fetch_list=fetch_targets,
                           return_numpy=False)
        else:
            outs = predict.run(inputs)
    t2 = time.time()

    ms = (t2 - t1) * 1000.0 / float(cnt)

    print("Inference: {} ms per batch image".format(ms))

    clsid2catid, catid2name = get_category_info(
        conf['with_background'], conf['metric'], FLAGS.label_list)
    bbox_result = get_bbox_result(outs, res, conf, clsid2catid)
    if FLAGS.visualize:
        visualize(bbox_result, catid2name)

    if FLAGS.dump_box:
        import json
        outfile = os.path.join(FLAGS.output_dir, 'bbox.json')
        logger.info('dump bbox to {}'.format(outfile))
        with open(outfile, 'w') as f:
            json.dump(bbox_result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path", type=str, default=None, help="model path.")
    parser.add_argument(
        "--config_path", type=str, default=None, help="preprocess config path.")
    parser.add_argument(
        "--infer_img", type=str, default=None, help="Image path")
    parser.add_argument(
        "--visualize",
        action='store_true',
        default=False,
        help="Whether to visualize detection output")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--label_list",
        type=str,
        default=None,
        help="Directory for label files.")
    parser.add_argument(
        "--dump_box",
        action='store_true',
        default=False,
        help="Whether to dump box")
    FLAGS = parser.parse_args()
    infer()
