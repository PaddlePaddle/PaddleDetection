import numpy as np
import imgaug.augmenters as iaa
from .operators import BaseOperator, register_op
from ppdet.utils.logger import setup_logger
from ppdet.data.culane_utils import linestrings_to_lanes, transform_annotation

logger = setup_logger(__name__)

__all__ = [
    "CULaneTrainProcess", "CULaneDataProcess", "HorizontalFlip",
    "ChannelShuffle", "CULaneAffine", "CULaneResize", "OneOfBlur",
    "MultiplyAndAddToBrightness", "AddToHueAndSaturation"
]


def trainTransforms(img_h, img_w):
    transforms = [{
        'name': 'Resize',
        'parameters': dict(size=dict(
            height=img_h, width=img_w)),
        'p': 1.0
    }, {
        'name': 'HorizontalFlip',
        'parameters': dict(p=1.0),
        'p': 0.5
    }, {
        'name': 'ChannelShuffle',
        'parameters': dict(p=1.0),
        'p': 0.1
    }, {
        'name': 'MultiplyAndAddToBrightness',
        'parameters': dict(
            mul=(0.85, 1.15), add=(-10, 10)),
        'p': 0.6
    }, {
        'name': 'AddToHueAndSaturation',
        'parameters': dict(value=(-10, 10)),
        'p': 0.7
    }, {
        'name': 'OneOf',
        'transforms': [
            dict(
                name='MotionBlur', parameters=dict(k=(3, 5))), dict(
                    name='MedianBlur', parameters=dict(k=(3, 5)))
        ],
        'p': 0.2
    }, {
        'name': 'Affine',
        'parameters': dict(
            translate_percent=dict(
                x=(-0.1, 0.1), y=(-0.1, 0.1)),
            rotate=(-10, 10),
            scale=(0.8, 1.2)),
        'p': 0.7
    }, {
        'name': 'Resize',
        'parameters': dict(size=dict(
            height=img_h, width=img_w)),
        'p': 1.0
    }]
    return transforms


@register_op
class CULaneTrainProcess(BaseOperator):
    def __init__(self, img_w, img_h):
        super(CULaneTrainProcess, self).__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.transforms = trainTransforms(self.img_h, self.img_w)

        if self.transforms is not None:
            img_transforms = []
            for aug in self.transforms:
                p = aug['p']
                if aug['name'] != 'OneOf':
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=getattr(iaa, aug['name'])(**aug[
                                'parameters'])))
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf([
                                getattr(iaa, aug_['name'])(**aug_['parameters'])
                                for aug_ in aug['transforms']
                            ])))
        else:
            img_transforms = []
        self.iaa_transform = iaa.Sequential(img_transforms)

    def apply(self, sample, context=None):
        img, line_strings, seg = self.iaa_transform(
            image=sample['image'],
            line_strings=sample['lanes'],
            segmentation_maps=sample['mask'])
        sample['image'] = img
        sample['lanes'] = line_strings
        sample['mask'] = seg
        return sample


@register_op
class CULaneDataProcess(BaseOperator):
    def __init__(self, img_w, img_h, num_points, max_lanes):
        super(CULaneDataProcess, self).__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.num_points = num_points
        self.n_offsets = num_points
        self.n_strips = num_points - 1
        self.strip_size = self.img_h / self.n_strips

        self.max_lanes = max_lanes
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)

    def apply(self, sample, context=None):
        data = {}
        line_strings = sample['lanes']
        line_strings.clip_out_of_image_()
        new_anno = {'lanes': linestrings_to_lanes(line_strings)}

        for i in range(30):
            try:
                annos = transform_annotation(
                    self.img_w, self.img_h, self.max_lanes, self.n_offsets,
                    self.offsets_ys, self.n_strips, self.strip_size, new_anno)
                label = annos['label']
                lane_endpoints = annos['lane_endpoints']
                break
            except:
                if (i + 1) == 30:
                    logger.critical('Transform annotation failed 30 times :(')
                    exit()

        sample['image'] = sample['image'].astype(np.float32) / 255.
        data['image'] = sample['image'].transpose(2, 0, 1)
        data['lane_line'] = label
        data['seg'] = sample['seg']
        data['full_img_path'] = sample['full_img_path']
        data['img_name'] = sample['img_name']
        data['im_id'] = sample['im_id']

        if 'mask' in sample.keys():
            data['seg'] = sample['mask'].get_arr()

        data['im_shape'] = np.array([self.img_w, self.img_h], dtype=np.float32)
        data['scale_factor'] = np.array([1., 1.], dtype=np.float32)

        return data


@register_op
class CULaneResize(BaseOperator):
    def __init__(self, img_h, img_w, prob=0.5):
        super(CULaneResize, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.prob = prob

    def apply(self, sample, context=None):
        transform = iaa.Sometimes(self.prob,
                                  iaa.Resize({
                                      "height": self.img_h,
                                      "width": self.img_w
                                  }))
        if 'mask' in sample.keys():
            img, line_strings, seg = transform(
                image=sample['image'],
                line_strings=sample['lanes'],
                segmentation_maps=sample['mask'])
            sample['image'] = img
            sample['lanes'] = line_strings
            sample['mask'] = seg
        else:
            img, line_strings = transform(
                image=sample['image'].copy().astype(np.uint8),
                line_strings=sample['lanes'])
            sample['image'] = img
            sample['lanes'] = line_strings

        return sample


@register_op
class HorizontalFlip(BaseOperator):
    def __init__(self, prob=0.5):
        super(HorizontalFlip, self).__init__()
        self.prob = prob

    def apply(self, sample, context=None):
        transform = iaa.Sometimes(self.prob, iaa.HorizontalFlip(1.0))
        if 'mask' in sample.keys():
            img, line_strings, seg = transform(
                image=sample['image'],
                line_strings=sample['lanes'],
                segmentation_maps=sample['mask'])
            sample['image'] = img
            sample['lanes'] = line_strings
            sample['mask'] = seg
        else:
            img, line_strings = transform(
                image=sample['image'], line_strings=sample['lanes'])
            sample['image'] = img
            sample['lanes'] = line_strings

        return sample


@register_op
class ChannelShuffle(BaseOperator):
    def __init__(self, prob=0.1):
        super(ChannelShuffle, self).__init__()
        self.prob = prob

    def apply(self, sample, context=None):
        transform = iaa.Sometimes(self.prob, iaa.ChannelShuffle(1.0))
        if 'mask' in sample.keys():
            img, line_strings, seg = transform(
                image=sample['image'],
                line_strings=sample['lanes'],
                segmentation_maps=sample['mask'])
            sample['image'] = img
            sample['lanes'] = line_strings
            sample['mask'] = seg
        else:
            img, line_strings = transform(
                image=sample['image'], line_strings=sample['lanes'])
            sample['image'] = img
            sample['lanes'] = line_strings

        return sample


@register_op
class MultiplyAndAddToBrightness(BaseOperator):
    def __init__(self, mul=(0.85, 1.15), add=(-10, 10), prob=0.5):
        super(MultiplyAndAddToBrightness, self).__init__()
        self.mul = tuple(mul)
        self.add = tuple(add)
        self.prob = prob

    def apply(self, sample, context=None):
        transform = iaa.Sometimes(
            self.prob,
            iaa.MultiplyAndAddToBrightness(
                mul=self.mul, add=self.add))
        if 'mask' in sample.keys():
            img, line_strings, seg = transform(
                image=sample['image'],
                line_strings=sample['lanes'],
                segmentation_maps=sample['mask'])
            sample['image'] = img
            sample['lanes'] = line_strings
            sample['mask'] = seg
        else:
            img, line_strings = transform(
                image=sample['image'], line_strings=sample['lanes'])
            sample['image'] = img
            sample['lanes'] = line_strings

        return sample


@register_op
class AddToHueAndSaturation(BaseOperator):
    def __init__(self, value=(-10, 10), prob=0.5):
        super(AddToHueAndSaturation, self).__init__()
        self.value = tuple(value)
        self.prob = prob

    def apply(self, sample, context=None):
        transform = iaa.Sometimes(
            self.prob, iaa.AddToHueAndSaturation(value=self.value))
        if 'mask' in sample.keys():
            img, line_strings, seg = transform(
                image=sample['image'],
                line_strings=sample['lanes'],
                segmentation_maps=sample['mask'])
            sample['image'] = img
            sample['lanes'] = line_strings
            sample['mask'] = seg
        else:
            img, line_strings = transform(
                image=sample['image'], line_strings=sample['lanes'])
            sample['image'] = img
            sample['lanes'] = line_strings

        return sample


@register_op
class OneOfBlur(BaseOperator):
    def __init__(self, MotionBlur_k=(3, 5), MedianBlur_k=(3, 5), prob=0.5):
        super(OneOfBlur, self).__init__()
        self.MotionBlur_k = tuple(MotionBlur_k)
        self.MedianBlur_k = tuple(MedianBlur_k)
        self.prob = prob

    def apply(self, sample, context=None):
        transform = iaa.Sometimes(
            self.prob,
            iaa.OneOf([
                iaa.MotionBlur(k=self.MotionBlur_k),
                iaa.MedianBlur(k=self.MedianBlur_k)
            ]))

        if 'mask' in sample.keys():
            img, line_strings, seg = transform(
                image=sample['image'],
                line_strings=sample['lanes'],
                segmentation_maps=sample['mask'])
            sample['image'] = img
            sample['lanes'] = line_strings
            sample['mask'] = seg
        else:
            img, line_strings = transform(
                image=sample['image'], line_strings=sample['lanes'])
            sample['image'] = img
            sample['lanes'] = line_strings

        return sample


@register_op
class CULaneAffine(BaseOperator):
    def __init__(self,
                 translate_percent_x=(-0.1, 0.1),
                 translate_percent_y=(-0.1, 0.1),
                 rotate=(3, 5),
                 scale=(0.8, 1.2),
                 prob=0.5):
        super(CULaneAffine, self).__init__()
        self.translate_percent = {
            'x': tuple(translate_percent_x),
            'y': tuple(translate_percent_y)
        }
        self.rotate = tuple(rotate)
        self.scale = tuple(scale)
        self.prob = prob

    def apply(self, sample, context=None):
        transform = iaa.Sometimes(
            self.prob,
            iaa.Affine(
                translate_percent=self.translate_percent,
                rotate=self.rotate,
                scale=self.scale))

        if 'mask' in sample.keys():
            img, line_strings, seg = transform(
                image=sample['image'],
                line_strings=sample['lanes'],
                segmentation_maps=sample['mask'])
            sample['image'] = img
            sample['lanes'] = line_strings
            sample['mask'] = seg
        else:
            img, line_strings = transform(
                image=sample['image'], line_strings=sample['lanes'])
            sample['image'] = img
            sample['lanes'] = line_strings

        return sample
