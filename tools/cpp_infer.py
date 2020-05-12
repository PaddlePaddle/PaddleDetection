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
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
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
    elif arch in ['SSD', 'Face']:
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
            "Unsupported arch: {}, expect YOLO, SSD, RetinaNet, RCNN and Face".
            format(arch))
    return info


class Resize(object):
    def __init__(self,
                 target_size,
                 max_size=0,
                 interp=cv2.INTER_LINEAR,
                 use_cv2=True,
                 image_shape=None):
        super(Resize, self).__init__()
        self.target_size = target_size
        self.max_size = max_size
        self.interp = interp
        self.use_cv2 = use_cv2
        self.image_shape = image_shape

    def __call__(self, im):
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
        if self.max_size != 0 and self.image_shape is not None:
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
            im = im.transpose((2, 0, 1))
        if self.to_bgr:
            im = im[[2, 1, 0], :, :]
        return im.copy()


class PadStride(object):
    def __init__(self, stride=0):
        assert stride >= 0, "Unsupported stride: {},"
        " the stride in PadStride must be greater "
        "or equal to 0".format(stride)
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
            img, scale = preprocess(img)
        else:
            img = preprocess(img)

    img = img[np.newaxis, :]  # N, C, H, W
    data.append(img)
    extra_info = get_extra_info(img, arch, orig_shape, scale)
    data += extra_info
    return data


def get_category_info(with_background, label_list):
    if label_list[0] != 'background' and with_background:
        label_list.insert(0, 'background')
    if label_list[0] == 'background' and not with_background:
        label_list = label_list[1:]
    clsid2catid = {i: i for i in range(len(label_list))}
    catid2name = {i: name for i, name in enumerate(label_list)}
    return clsid2catid, catid2name


def clip_bbox(bbox):
    xmin = max(min(bbox[0], 1.), 0.)
    ymin = max(min(bbox[1], 1.), 0.)
    xmax = max(min(bbox[2], 1.), 0.)
    ymax = max(min(bbox[3], 1.), 0.)
    return xmin, ymin, xmax, ymax


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
        if bboxes.shape == (1, 1) or bboxes is None:
            continue

        k = 0
        for i in range(len(lengths)):
            num = lengths[i]
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
                coco_res = {'category_id': catid, 'bbox': bbox, 'score': score}
                xywh_res.append(coco_res)
                k += 1
    return xywh_res


def expand_boxes(boxes, scale):
    """
    Expand an array of boxes by a given scale.
    """
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


def mask2out(results, clsid2catid, resolution, thresh_binarize=0.5):
    import pycocotools.mask as mask_util
    scale = (resolution + 2.0) / resolution

    segm_res = []

    for t in results:
        bboxes = t['bbox'][0]
        lengths = t['bbox'][1][0]
        if bboxes.shape == (1, 1) or bboxes is None:
            continue
        if len(bboxes.tolist()) == 0:
            continue
        masks = t['mask'][0]

        s = 0
        # for each sample
        for i in range(len(lengths)):
            num = lengths[i]
            im_shape = t['im_shape'][i]

            bbox = bboxes[s:s + num][:, 2:]
            clsid_scores = bboxes[s:s + num][:, 0:2]
            mask = masks[s:s + num]
            s += num

            im_h = int(im_shape[0])
            im_w = int(im_shape[1])

            expand_bbox = expand_boxes(bbox, scale)
            expand_bbox = expand_bbox.astype(np.int32)

            padded_mask = np.zeros(
                (resolution + 2, resolution + 2), dtype=np.float32)

            for j in range(num):
                xmin, ymin, xmax, ymax = expand_bbox[j].tolist()
                clsid, score = clsid_scores[j].tolist()
                clsid = int(clsid)
                padded_mask[1:-1, 1:-1] = mask[j, clsid, :, :]

                catid = clsid2catid[clsid]

                w = xmax - xmin + 1
                h = ymax - ymin + 1
                w = np.maximum(w, 1)
                h = np.maximum(h, 1)

                resized_mask = cv2.resize(padded_mask, (w, h))
                resized_mask = np.array(
                    resized_mask > thresh_binarize, dtype=np.uint8)
                im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

                x0 = min(max(xmin, 0), im_w)
                x1 = min(max(xmax + 1, 0), im_w)
                y0 = min(max(ymin, 0), im_h)
                y1 = min(max(ymax + 1, 0), im_h)

                im_mask[y0:y1, x0:x1] = resized_mask[(y0 - ymin):(y1 - ymin), (
                    x0 - xmin):(x1 - xmin)]
                segm = mask_util.encode(
                    np.array(
                        im_mask[:, :, np.newaxis], order='F'))[0]
                catid = clsid2catid[clsid]
                segm['counts'] = segm['counts'].decode('utf8')
                coco_res = {
                    'category_id': catid,
                    'segmentation': segm,
                    'score': score
                }
                segm_res.append(coco_res)
    return segm_res


def color_map(num_classes):
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = np.array(color_map).reshape(-1, 3)
    return color_map


def draw_bbox(image, catid2name, bboxes, threshold, color_list):
    """
    draw bbox on image
    """
    draw = ImageDraw.Draw(image)

    for dt in np.array(bboxes):
        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        if score < threshold:
            continue

        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h

        color = tuple(color_list[catid])

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


def draw_mask(image, masks, threshold, color_list, alpha=0.7):
    """
    Draw mask on image
    """
    mask_color_id = 0
    w_ratio = .4
    img_array = np.array(image).astype('float32')
    for dt in np.array(masks):
        segm, score = dt['segmentation'], dt['score']
        if score < threshold:
            continue
        import pycocotools.mask as mask_util
        mask = mask_util.decode(segm) * 255
        color_mask = color_list[mask_color_id % len(color_list), 0:3]
        mask_color_id += 1
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        img_array[idx[0], idx[1], :] *= 1.0 - alpha
        img_array[idx[0], idx[1], :] += alpha * color_mask
    return Image.fromarray(img_array.astype('uint8'))


def get_bbox_result(output, result, conf, clsid2catid):
    is_bbox_normalized = True if conf['arch'] in ['SSD', 'Face'] else False
    lengths = offset_to_lengths(output.lod())
    np_data = np.array(output) if conf[
        'use_python_inference'] else output.copy_to_cpu()
    result['bbox'] = (np_data, lengths)
    result['im_id'] = np.array([[0]])

    bbox_results = bbox2out([result], clsid2catid, is_bbox_normalized)
    return bbox_results


def get_mask_result(output, result, conf, clsid2catid):
    resolution = conf['mask_resolution']
    bbox_out, mask_out = output
    lengths = offset_to_lengths(bbox_out.lod())
    bbox = np.array(bbox_out) if conf[
        'use_python_inference'] else bbox_out.copy_to_cpu()
    mask = np.array(mask_out) if conf[
        'use_python_inference'] else mask_out.copy_to_cpu()
    result['bbox'] = (bbox, lengths)
    result['mask'] = (mask, lengths)
    mask_results = mask2out([result], clsid2catid, conf['mask_resolution'])
    return mask_results


def visualize(bbox_results, catid2name, num_classes, mask_results=None):
    image = Image.open(FLAGS.infer_img).convert('RGB')
    color_list = color_map(num_classes)
    image = draw_bbox(image, catid2name, bbox_results, 0.5, color_list)
    if mask_results is not None:
        image = draw_mask(image, mask_results, 0.5, color_list)
    image_path = os.path.split(FLAGS.infer_img)[-1]
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    out_path = os.path.join(FLAGS.output_dir, image_path)
    image.save(out_path, quality=95)
    logger.info('Save visualize result to {}'.format(out_path))


def infer():
    logger.info("cpp_infer.py is deprecated since release/0.3. Please use"
                "deploy/python for your python deployment")
    model_path = FLAGS.model_path
    config_path = FLAGS.config_path
    res = {}
    assert model_path is not None, "Model path: {} does not exist!".format(
        model_path)
    assert config_path is not None, "Config path: {} does not exist!".format(
        config_path)
    with open(config_path) as f:
        conf = yaml.safe_load(f)

    use_trt = not conf['use_python_inference'] and 'trt' in conf['mode']
    if use_trt:
        logger.warning(
            "Due to the limitation of tensorRT, the image shape needs to set in export_model"
        )
    img_data = Preprocess(FLAGS.infer_img, conf['arch'], conf['Preprocess'])
    if conf['arch'] in ['SSD', 'Face']:
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
        config = create_config(
            model_path,
            mode=conf['mode'],
            min_subgraph_size=conf['min_subgraph_size'])
        predict = fluid.core.create_paddle_predictor(config)
        input_names = predict.get_input_names()
        for ind, d in enumerate(img_data):
            input_tensor = predict.get_input_tensor(input_names[ind])
            input_tensor.copy_from_cpu(d.copy())

    logger.info('warmup...')
    for i in range(10):
        if conf['use_python_inference']:
            outs = exe.run(infer_prog,
                           feed=data_dict,
                           fetch_list=fetch_targets,
                           return_numpy=False)
        else:
            predict.zero_copy_run()

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
            outs = []
            predict.zero_copy_run()
            output_names = predict.get_output_names()
            for o_name in output_names:
                outs.append(predict.get_output_tensor(o_name))
    t2 = time.time()

    ms = (t2 - t1) * 1000.0 / float(cnt)

    print("Inference: {} ms per batch image".format(ms))

    clsid2catid, catid2name = get_category_info(conf['with_background'],
                                                conf['label_list'])
    bbox_result = get_bbox_result(outs[0], res, conf, clsid2catid)

    mask_result = None
    if 'mask_resolution' in conf:
        res['im_shape'] = img_data[-1]
        mask_result = get_mask_result(outs, res, conf, clsid2catid)

    if FLAGS.visualize:
        visualize(bbox_result, catid2name, len(conf['label_list']), mask_result)

    if FLAGS.dump_result:
        import json
        bbox_file = os.path.join(FLAGS.output_dir, 'bbox.json')
        logger.info('dump bbox to {}'.format(bbox_file))
        with open(bbox_file, 'w') as f:
            json.dump(bbox_result, f)
        if mask_result is not None:
            mask_file = os.path.join(FLAGS.output_dir, 'mask.json')
            logger.info('dump mask to {}'.format(mask_file))
            with open(mask_file, 'w') as f:
                json.dump(mask_result, f)


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
        "--dump_result",
        action='store_true',
        default=False,
        help="Whether to dump result")
    FLAGS = parser.parse_args()
    infer()
