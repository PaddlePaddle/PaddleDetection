import cv2
import numpy as np

from ppdet.data.culane_utils import clip_lanes, transform_annotation
from ppdet.utils.logger import setup_logger
from .operators import BaseOperator, register_op

logger = setup_logger(__name__)

__all__ = [
    "CULaneTrainProcess", "CULaneDataProcess", "HorizontalFlip",
    "ChannelShuffle", "CULaneAffine", "CULaneResize", "OneOfBlur",
    "MultiplyAndAddToBrightness", "AddToHueAndSaturation"
]

# The original CLRNet augmentations treated BGR input as RGB for color
# conversions. Keep that behavior for brightness and hue/saturation.
_BRIGHTNESS_COLORSPACES = (
    (cv2.COLOR_RGB2YCrCb, cv2.COLOR_YCrCb2RGB, 0),
    (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB, 2),
    (cv2.COLOR_RGB2HLS, cv2.COLOR_HLS2RGB, 1),
    (cv2.COLOR_RGB2Lab, cv2.COLOR_Lab2RGB, 0),
    (cv2.COLOR_RGB2Luv, cv2.COLOR_Luv2RGB, 0),
    (cv2.COLOR_RGB2YUV, cv2.COLOR_YUV2RGB, 0),
)


@register_op
class CULaneTrainProcess(BaseOperator):
    def __init__(self, img_w, img_h):
        super(CULaneTrainProcess, self).__init__()
        self.transforms = (
            CULaneResize(img_h, img_w, prob=1.0),
            HorizontalFlip(prob=0.5),
            ChannelShuffle(prob=0.1),
            MultiplyAndAddToBrightness(prob=0.6),
            AddToHueAndSaturation(prob=0.7),
            OneOfBlur(prob=0.2),
            CULaneAffine(rotate=(-10, 10), prob=0.7),
        )

    def apply(self, sample, context=None):
        for transform in self.transforms:
            sample = transform.apply(sample, context)
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
        height, width = sample['image'].shape[:2]
        try:
            new_anno = {'lanes': clip_lanes(sample['lanes'], height, width)}
            annos = transform_annotation(
                self.img_w, self.img_h, self.max_lanes, self.n_offsets,
                self.offsets_ys, self.n_strips, self.strip_size, new_anno)
        except Exception:
            logger.error('Failed to transform CULane annotations for %s',
                         sample.get('full_img_path', '<unknown>'))
            raise

        sample['image'] = sample['image'].astype(np.float32) / 255.
        data['image'] = sample['image'].transpose(2, 0, 1)
        data['lane_line'] = annos['label']
        if 'mask' in sample:
            data['seg'] = sample['mask']
        data['full_img_path'] = sample['full_img_path']
        data['img_name'] = sample['img_name']
        data['im_id'] = sample['im_id']
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
        if np.random.random() >= self.prob:
            return sample
        height, width = sample['image'].shape[:2]
        size = (self.img_w, self.img_h)
        if size == (width, height):
            return sample
        sample['image'] = cv2.resize(
            sample['image'], size, interpolation=cv2.INTER_CUBIC)
        if 'mask' in sample:
            sample['mask'] = cv2.resize(
                sample['mask'], size, interpolation=cv2.INTER_NEAREST)
        source_size = np.array((width, height), dtype=np.float32)
        target_size = np.array(size, dtype=np.float32)
        sample['lanes'] = [
            lane / source_size * target_size for lane in sample['lanes']
        ]
        return sample


@register_op
class HorizontalFlip(BaseOperator):
    def __init__(self, prob=0.5):
        super(HorizontalFlip, self).__init__()
        self.prob = prob

    def apply(self, sample, context=None):
        if np.random.random() < self.prob:
            width = sample['image'].shape[1]
            sample['image'] = cv2.flip(sample['image'], 1)
            if 'mask' in sample:
                sample['mask'] = cv2.flip(sample['mask'], 1)
            sample['lanes'] = [
                np.column_stack((width - lane[:, 0], lane[:, 1]))
                for lane in sample['lanes']
            ]
        return sample


@register_op
class ChannelShuffle(BaseOperator):
    def __init__(self, prob=0.1):
        super(ChannelShuffle, self).__init__()
        self.prob = prob

    def apply(self, sample, context=None):
        if np.random.random() < self.prob:
            image = sample['image']
            sample['image'] = np.take(
                image, np.random.permutation(image.shape[2]), axis=2)
        return sample


@register_op
class MultiplyAndAddToBrightness(BaseOperator):
    def __init__(self, mul=(0.85, 1.15), add=(-10, 10), prob=0.5):
        super(MultiplyAndAddToBrightness, self).__init__()
        self.mul = tuple(mul)
        self.add = tuple(add)
        self.prob = prob

    def apply(self, sample, context=None):
        if np.random.random() < self.prob:
            forward, inverse, channel = _BRIGHTNESS_COLORSPACES[
                np.random.randint(len(_BRIGHTNESS_COLORSPACES))]
            converted = cv2.cvtColor(sample['image'], forward)
            levels = np.arange(256, dtype=np.float32)
            mul = np.clip(levels * np.random.uniform(*self.mul), 0,
                          255).astype(np.uint8)
            add = np.clip(levels + np.rint(np.random.uniform(*self.add)), 0,
                          255).astype(np.uint8)
            lookup = add[mul] if np.random.random() < 0.5 else mul[add]
            converted[:, :, channel] = cv2.LUT(converted[:, :, channel], lookup)
            sample['image'] = cv2.cvtColor(converted, inverse)
        return sample


@register_op
class AddToHueAndSaturation(BaseOperator):
    def __init__(self, value=(-10, 10), prob=0.5):
        super(AddToHueAndSaturation, self).__init__()
        self.value = tuple(value)
        self.prob = prob

    def apply(self, sample, context=None):
        if np.random.random() < self.prob:
            delta = np.random.randint(self.value[0], self.value[1] + 1)
            hsv = cv2.cvtColor(sample['image'], cv2.COLOR_RGB2HSV)
            hue = int(delta * 180 / 255)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue) % 180
            hsv[:, :, 1] = np.clip(
                hsv[:, :, 1].astype(np.int16) + delta, 0, 255)
            sample['image'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return sample


@register_op
class OneOfBlur(BaseOperator):
    def __init__(self, MotionBlur_k=(3, 5), MedianBlur_k=(3, 5), prob=0.5):
        super(OneOfBlur, self).__init__()
        self.MotionBlur_k = tuple(MotionBlur_k)
        self.MedianBlur_k = tuple(MedianBlur_k)
        self.prob = prob

    def apply(self, sample, context=None):
        if np.random.random() >= self.prob:
            return sample
        image = sample['image']
        if np.random.random() < 0.5:
            size = np.random.randint(self.MotionBlur_k[0],
                                     self.MotionBlur_k[1] + 1) | 1
            direction = np.random.random()
            kernel = np.zeros((size, size), dtype=np.float32)
            kernel[:, size // 2] = np.linspace(direction, 1 - direction, size)
            center = ((size - 1) / 2, (size - 1) / 2)
            rotation = cv2.getRotationMatrix2D(
                center, -np.random.uniform(0, 360), 1.0)
            kernel = cv2.warpAffine((kernel * 255).astype(np.uint8), rotation,
                                    (size, size)).astype(np.float32)
            sample['image'] = cv2.filter2D(image, -1, kernel / kernel.sum())
        else:
            size = np.random.randint(self.MedianBlur_k[0],
                                     self.MedianBlur_k[1] + 1) | 1
            sample['image'] = cv2.medianBlur(image, size)
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
        self.translate_percent_x = tuple(translate_percent_x)
        self.translate_percent_y = tuple(translate_percent_y)
        self.rotate = tuple(rotate)
        self.scale = tuple(scale)
        self.prob = prob

    def apply(self, sample, context=None):
        if np.random.random() >= self.prob:
            return sample
        height, width = sample['image'].shape[:2]
        shift_x = np.float32(np.random.uniform(*self.translate_percent_x))
        shift_y = np.float32(np.random.uniform(*self.translate_percent_y))
        shift = np.array([shift_x, shift_y]) * np.array([width, height],
                                                       dtype=np.float32)
        angle = np.deg2rad(np.float32(np.random.uniform(*self.rotate)))
        scale = np.float32(np.random.uniform(*self.scale))
        cos_angle, sin_angle = scale * np.array([np.cos(angle), np.sin(angle)])
        rotation = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]],
                            dtype=np.float64)
        # Pixels use their centers. Lane vertices are continuous coordinates.
        lane_center = np.array([width / 2, height / 2])
        pixel_center = lane_center - 0.5
        pixel_offset = pixel_center - rotation @ pixel_center + shift
        pixel_matrix = np.column_stack((rotation, pixel_offset))
        sample['image'] = cv2.warpAffine(
            sample['image'], pixel_matrix, (width, height),
            flags=cv2.INTER_LINEAR)
        if 'mask' in sample:
            sample['mask'] = cv2.warpAffine(
                sample['mask'], pixel_matrix, (width, height),
                flags=cv2.INTER_NEAREST)
        lane_offset = lane_center - rotation @ lane_center + shift
        sample['lanes'] = [
            (lane @ rotation.T + lane_offset).astype(np.float32)
            for lane in sample['lanes']
        ]
        return sample
