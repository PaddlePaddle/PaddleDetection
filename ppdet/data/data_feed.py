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
from __future__ import print_function
from __future__ import division

import os
import inspect

from ppdet.core.workspace import register, serializable
from ppdet.utils.download import get_dataset_path

from ppdet.data.reader import Reader
# XXX these are for triggering the decorator
from ppdet.data.transform.operators import (
    DecodeImage, MixupImage, NormalizeBox, NormalizeImage, RandomDistort,
    RandomFlipImage, RandomInterpImage, ResizeImage, ExpandImage, CropImage,
    Permute)

from ppdet.data.transform.arrange_sample import (
    ArrangeRCNN, ArrangeTestRCNN, ArrangeSSD, ArrangeEvalSSD, ArrangeTestSSD, 
    ArrangeYOLO, ArrangeEvalYOLO, ArrangeTestYOLO)

__all__ = [
    'PadBatch', 'MultiScale', 'RandomShape', 'DataSet', 'CocoDataSet',
    'DataFeed', 'TrainFeed', 'EvalFeed', 'FasterRCNNTrainFeed',
    'MaskRCNNTrainFeed', 'FasterRCNNTestFeed', 'MaskRCNNTestFeed',
    'SSDTrainFeed', 'SSDEvalFeed', 'SSDTestFeed', 'YoloTrainFeed',
    'YoloEvalFeed', 'YoloTestFeed', 'create_reader'
]


def _prepare_data_config(feed, args_path):
    # if `DATASET_DIR` does not exists, search ~/.paddle/dataset for a directory
    # named `DATASET_DIR` (e.g., coco, pascal), if not present either, download
    dataset_home = args_path if args_path else feed.dataset.dataset_dir
    if dataset_home:
        annotation = getattr(feed.dataset, 'annotation', None)
        image_dir = getattr(feed.dataset, 'image_dir', None)
        dataset_dir = get_dataset_path(dataset_home, annotation, image_dir)
        if annotation:
            feed.dataset.annotation = os.path.join(dataset_dir, annotation)
        if image_dir:
            feed.dataset.image_dir = os.path.join(dataset_dir, image_dir)

    mixup_epoch = -1
    if getattr(feed, 'mixup_epoch', None) is not None:
        mixup_epoch = feed.mixup_epoch

    data_config = {
        'ANNO_FILE': feed.dataset.annotation,
        'IMAGE_DIR': feed.dataset.image_dir,
        'USE_DEFAULT_LABEL': feed.dataset.use_default_label,
        'IS_SHUFFLE': feed.shuffle,
        'SAMPLES': feed.samples,
        'WITH_BACKGROUND': feed.with_background,
        'MIXUP_EPOCH': mixup_epoch,
        'TYPE': type(feed.dataset).__source__
    }

    if len(getattr(feed.dataset, 'images', [])) > 0:
        data_config['IMAGES'] = feed.dataset.images

    return data_config


def create_reader(feed, max_iter=0, args_path=None, my_source=None):
    """
    Return iterable data reader.

    Args:
        max_iter (int): number of iterations.
        my_source (callable): callable function to create a source iterator
            which is used to provide source data in 'ppdet.data.reader'
    """

    # if `DATASET_DIR` does not exists, search ~/.paddle/dataset for a directory
    # named `DATASET_DIR` (e.g., coco, pascal), if not present either, download
    data_config = _prepare_data_config(feed, args_path)

    bufsize = getattr(feed, 'bufsize', 10)
    use_process = getattr(feed, 'use_process', False)
    memsize = getattr(feed, 'memsize', '3G')
    transform_config = {
        'WORKER_CONF': {
            'bufsize': bufsize,
            'worker_num': feed.num_workers,
            'use_process': use_process,
            'memsize': memsize
        },
        'BATCH_SIZE': feed.batch_size,
        'DROP_LAST': feed.drop_last,
        'USE_PADDED_IM_INFO': feed.use_padded_im_info,
    }

    batch_transforms = feed.batch_transforms
    pad = [t for t in batch_transforms if isinstance(t, PadBatch)]
    rand_shape = [t for t in batch_transforms if isinstance(t, RandomShape)]
    multi_scale = [t for t in batch_transforms if isinstance(t, MultiScale)]

    if any(pad):
        transform_config['IS_PADDING'] = True
        if pad[0].pad_to_stride != 0:
            transform_config['COARSEST_STRIDE'] = pad[0].pad_to_stride
    if any(rand_shape):
        transform_config['RANDOM_SHAPES'] = rand_shape[0].sizes
    if any(multi_scale):
        transform_config['MULTI_SCALES'] = multi_scale[0].scales

    if hasattr(inspect, 'getfullargspec'):
        argspec = inspect.getfullargspec
    else:
        argspec = inspect.getargspec

    ops = []
    for op in feed.sample_transforms:
        op_dict = op.__dict__.copy()
        argnames = [
            arg for arg in argspec(type(op).__init__).args if arg != 'self'
        ]
        op_dict = {k: v for k, v in op_dict.items() if k in argnames}
        op_dict['op'] = op.__class__.__name__
        ops.append(op_dict)
    transform_config['OPS'] = ops

    return Reader.create(feed.mode, data_config, transform_config, max_iter,
                         my_source)


# XXX batch transforms are only stubs for now, actually handled by `post_map`
@serializable
class PadBatch(object):
    """
    Pad a batch of samples to same dimensions

    Args:
        pad_to_stride (int): pad to multiple of strides, e.g., 32
    """

    def __init__(self, pad_to_stride=0):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride


@serializable
class MultiScale(object):
    """
    Randomly resize image by scale

    Args:
        scales (list): list of int, randomly resize to one of these scales
    """

    def __init__(self, scales=[]):
        super(MultiScale, self).__init__()
        self.scales = scales


@serializable
class RandomShape(object):
    """
    Randomly reshape a batch

    Args:
        sizes (list): list of int, random choose a size from these
    """

    def __init__(self, sizes=[]):
        super(RandomShape, self).__init__()
        self.sizes = sizes


@serializable
class DataSet(object):
    """
    Dataset, e.g., coco, pascal voc

    Args:
        annotation (str): annotation file path
        image_dir (str): directory where image files are stored
        shuffle (bool): shuffle samples
    """
    __source__ = 'RoiDbSource'

    def __init__(self,
                 annotation,
                 image_dir,
                 dataset_dir=None,
                 use_default_label=None):
        super(DataSet, self).__init__()
        self.dataset_dir = dataset_dir
        self.annotation = annotation
        self.image_dir = image_dir
        self.use_default_label = use_default_label


COCO_DATASET_DIR = 'coco'
COCO_TRAIN_ANNOTATION = 'annotations/instances_train2017.json'
COCO_TRAIN_IMAGE_DIR = 'train2017'
COCO_VAL_ANNOTATION = 'annotations/instances_val2017.json'
COCO_VAL_IMAGE_DIR = 'val2017'


@serializable
class CocoDataSet(DataSet):
    def __init__(self,
                 dataset_dir=COCO_DATASET_DIR,
                 annotation=COCO_TRAIN_ANNOTATION,
                 image_dir=COCO_TRAIN_IMAGE_DIR):
        super(CocoDataSet, self).__init__(
            dataset_dir=dataset_dir, annotation=annotation, image_dir=image_dir)


VOC_DATASET_DIR = 'pascalvoc'
VOC_TRAIN_ANNOTATION = 'VOCdevkit/VOC_all/ImageSets/Main/train.txt'
VOC_VAL_ANNOTATION = 'VOCdevkit/VOC_all/ImageSets/Main/val.txt'
VOC_TEST_ANNOTATION = 'VOCdevkit/VOC_all/ImageSets/Main/test.txt'
VOC_IMAGE_DIR = 'VOCdevkit/VOC_all/JPEGImages'
VOC_USE_DEFAULT_LABEL = None


@serializable
class VocDataSet(DataSet):
    __source__ = 'VOCSource'

    def __init__(self,
                 dataset_dir=VOC_DATASET_DIR,
                 annotation=VOC_TRAIN_ANNOTATION,
                 image_dir=VOC_IMAGE_DIR,
                 use_default_label=VOC_USE_DEFAULT_LABEL):
        super(VocDataSet, self).__init__(
            dataset_dir=dataset_dir,
            annotation=annotation,
            image_dir=image_dir,
            use_default_label=use_default_label)


@serializable
class SimpleDataSet(DataSet):
    __source__ = 'SimpleSource'

    def __init__(self,
                 dataset_dir=None,
                 annotation=None,
                 image_dir=None,
                 use_default_label=None):
        super(SimpleDataSet, self).__init__(
            dataset_dir=dataset_dir, annotation=annotation, image_dir=image_dir)
        self.images = []

    def add_images(self, images):
        self.images.extend(images)


@serializable
class DataFeed(object):
    """
    DataFeed encompasses all data loading related settings

    Args:
        dataset (object): a `Dataset` instance
        fields (list): list of data fields needed
        image_shape (list): list of image dims (C, MAX_DIM, MIN_DIM)
        sample_transforms (list): list of sample transformations to use
        batch_transforms (list): list of batch transformations to use
        batch_size (int): number of images per device
        shuffle (bool): if samples should be shuffled
        drop_last (bool): drop last batch if size is uneven
        num_workers (int): number of workers processes (or threads)
        bufsize (int): size of queue used to buffer results from workers
        use_process (bool): use process or thread as workers
        memsize (str): size of shared memory used in result queue
                        when 'use_process' is True, default to '3G'
    """
    __category__ = 'data'

    def __init__(self,
                 dataset,
                 fields,
                 image_shape,
                 sample_transforms=None,
                 batch_transforms=None,
                 batch_size=1,
                 shuffle=False,
                 samples=-1,
                 drop_last=False,
                 with_background=True,
                 num_workers=2,
                 bufsize=10,
                 use_process=False,
                 memsize=None,
                 use_padded_im_info=False):
        super(DataFeed, self).__init__()
        self.fields = fields
        self.image_shape = image_shape
        self.sample_transforms = sample_transforms
        self.batch_transforms = batch_transforms
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.samples = samples
        self.drop_last = drop_last
        self.with_background = with_background
        self.num_workers = num_workers
        self.bufsize = bufsize
        self.use_process = use_process
        self.memsize = memsize
        self.dataset = dataset
        self.use_padded_im_info = use_padded_im_info
        if isinstance(dataset, dict):
            self.dataset = DataSet(**dataset)


# for custom (i.e., Non-preset) datasets
@register
class TrainFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset,
                 fields,
                 image_shape,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=True,
                 samples=-1,
                 drop_last=False,
                 with_background=True,
                 num_workers=2,
                 bufsize=10,
                 use_process=True,
                 memsize=None):
        super(TrainFeed, self).__init__(
            dataset,
            fields,
            image_shape,
            sample_transforms,
            batch_transforms,
            batch_size=batch_size,
            shuffle=shuffle,
            samples=samples,
            drop_last=drop_last,
            with_background=with_background,
            num_workers=num_workers,
            bufsize=bufsize,
            use_process=use_process,
            memsize=memsize)


@register
class EvalFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset,
                 fields,
                 image_shape,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 samples=-1,
                 drop_last=False,
                 with_background=True,
                 num_workers=2):
        super(EvalFeed, self).__init__(
            dataset,
            fields,
            image_shape,
            sample_transforms,
            batch_transforms,
            batch_size=batch_size,
            shuffle=shuffle,
            samples=samples,
            drop_last=drop_last,
            with_background=with_background,
            num_workers=num_workers)


@register
class TestFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset,
                 fields,
                 image_shape,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 with_background=True,
                 num_workers=2):
        super(TestFeed, self).__init__(
            dataset,
            fields,
            image_shape,
            sample_transforms,
            batch_transforms,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            with_background=with_background,
            num_workers=num_workers)


# yapf: disable
@register
class FasterRCNNTrainFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=CocoDataSet().__dict__,
                 fields=[
                     'image', 'im_info', 'im_id', 'gt_box', 'gt_label',
                     'is_crowd'
                 ],
                 image_shape=[3, 800, 1333],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     RandomFlipImage(prob=0.5),
                     NormalizeImage(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    is_scale=True,
                                    is_channel_first=False),
                     ResizeImage(target_size=800, max_size=1333, interp=1),
                     Permute(to_bgr=False)
                 ],
                 batch_transforms=[PadBatch()],
                 batch_size=1,
                 shuffle=True,
                 samples=-1,
                 drop_last=False,
                 bufsize=10,
                 num_workers=2,
                 use_process=False,
                 memsize=None):
        # XXX this should be handled by the data loader, since `fields` is
        # given, just collect them
        sample_transforms.append(ArrangeRCNN())
        super(FasterRCNNTrainFeed, self).__init__(
            dataset,
            fields,
            image_shape,
            sample_transforms,
            batch_transforms,
            batch_size=batch_size,
            shuffle=shuffle,
            samples=samples,
            drop_last=drop_last,
            bufsize=bufsize,
            num_workers=num_workers,
            use_process=use_process,
            memsize=memsize)
        # XXX these modes should be unified
        self.mode = 'TRAIN'


@register
class FasterRCNNEvalFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=CocoDataSet(COCO_VAL_ANNOTATION,
                                     COCO_VAL_IMAGE_DIR).__dict__,
                 fields=['image', 'im_info', 'im_id', 'im_shape'],
                 image_shape=[3, 800, 1333],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     NormalizeImage(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    is_scale=True,
                                    is_channel_first=False),
                     ResizeImage(target_size=800, max_size=1333, interp=1),
                     Permute(to_bgr=False)
                 ],
                 batch_transforms=[PadBatch()],
                 batch_size=1,
                 shuffle=False,
                 samples=-1,
                 drop_last=False,
                 num_workers=2,
                 use_padded_im_info=True):
        sample_transforms.append(ArrangeTestRCNN())
        super(FasterRCNNEvalFeed, self).__init__(
            dataset,
            fields,
            image_shape,
            sample_transforms,
            batch_transforms,
            batch_size=batch_size,
            shuffle=shuffle,
            samples=samples,
            drop_last=drop_last,
            num_workers=num_workers,
            use_padded_im_info=use_padded_im_info)
        self.mode = 'VAL'


@register
class FasterRCNNTestFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=SimpleDataSet(COCO_VAL_ANNOTATION,
                                       COCO_VAL_IMAGE_DIR).__dict__,
                 fields=['image', 'im_info', 'im_id', 'im_shape'],
                 image_shape=[3, 800, 1333],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     NormalizeImage(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    is_scale=True,
                                    is_channel_first=False),
                     Permute(to_bgr=False)
                 ],
                 batch_transforms=[PadBatch()],
                 batch_size=1,
                 shuffle=False,
                 samples=-1,
                 drop_last=False,
                 num_workers=2,
                 use_padded_im_info=True):
        sample_transforms.append(ArrangeTestRCNN())
        if isinstance(dataset, dict):
            dataset = SimpleDataSet(**dataset)
        super(FasterRCNNTestFeed, self).__init__(
            dataset,
            fields,
            image_shape,
            sample_transforms,
            batch_transforms,
            batch_size=batch_size,
            shuffle=shuffle,
            samples=samples,
            drop_last=drop_last,
            num_workers=num_workers,
            use_padded_im_info=use_padded_im_info)
        self.mode = 'TEST'


# XXX currently use two presets, in the future, these should be combined into a
# single `RCNNTrainFeed`. Mask (and keypoint) should be processed
# automatically if `gt_mask` (or `gt_keypoints`) is in the required fields
@register
class MaskRCNNTrainFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=CocoDataSet().__dict__,
                 fields=[
                     'image', 'im_info', 'im_id', 'gt_box', 'gt_label',
                     'is_crowd', 'gt_mask'
                 ],
                 image_shape=[3, 800, 1333],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     RandomFlipImage(prob=0.5, is_mask_flip=True),
                     NormalizeImage(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    is_scale=True,
                                    is_channel_first=False),
                     ResizeImage(target_size=800,
                                 max_size=1333,
                                 interp=1,
                                 use_cv2=True),
                     Permute(to_bgr=False, channel_first=True)
                 ],
                 batch_transforms=[PadBatch()],
                 batch_size=1,
                 shuffle=True,
                 samples=-1,
                 drop_last=False,
                 num_workers=2,
                 use_process=False,
                 use_padded_im_info=False):
        sample_transforms.append(ArrangeRCNN(is_mask=True))
        super(MaskRCNNTrainFeed, self).__init__(
            dataset,
            fields,
            image_shape,
            sample_transforms,
            batch_transforms,
            batch_size=batch_size,
            shuffle=shuffle,
            samples=samples,
            drop_last=drop_last,
            num_workers=num_workers,
            use_process=use_process)
        self.mode = 'TRAIN'


@register
class MaskRCNNEvalFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=CocoDataSet(COCO_VAL_ANNOTATION,
                                     COCO_VAL_IMAGE_DIR).__dict__,
                 fields=['image', 'im_info', 'im_id', 'im_shape'],
                 image_shape=[3, 800, 1333],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     NormalizeImage(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    is_scale=True,
                                    is_channel_first=False),
                     ResizeImage(target_size=800,
                                 max_size=1333,
                                 interp=1,
                                 use_cv2=True),
                     Permute(to_bgr=False, channel_first=True)
                 ],
                 batch_transforms=[PadBatch()],
                 batch_size=1,
                 shuffle=False,
                 samples=-1,
                 drop_last=False,
                 num_workers=2,
                 use_process=False,
                 use_padded_im_info=True):
        sample_transforms.append(ArrangeTestRCNN())
        super(MaskRCNNEvalFeed, self).__init__(
            dataset,
            fields,
            image_shape,
            sample_transforms,
            batch_transforms,
            batch_size=batch_size,
            shuffle=shuffle,
            samples=samples,
            drop_last=drop_last,
            num_workers=num_workers,
            use_process=use_process,
            use_padded_im_info=use_padded_im_info)
        self.mode = 'VAL'


@register
class MaskRCNNTestFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=SimpleDataSet(COCO_VAL_ANNOTATION,
                                       COCO_VAL_IMAGE_DIR).__dict__,
                 fields=['image', 'im_info', 'im_id', 'im_shape'],
                 image_shape=[3, 800, 1333],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     NormalizeImage(
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         is_scale=True,
                         is_channel_first=False),
                     Permute(to_bgr=False, channel_first=True)
                 ],
                 batch_transforms=[PadBatch()],
                 batch_size=1,
                 shuffle=False,
                 samples=-1,
                 drop_last=False,
                 num_workers=2,
                 use_process=False,
                 use_padded_im_info=True):
        sample_transforms.append(ArrangeTestRCNN())
        if isinstance(dataset, dict):
            dataset = SimpleDataSet(**dataset)
        super(MaskRCNNTestFeed, self).__init__(
            dataset,
            fields,
            image_shape,
            sample_transforms,
            batch_transforms,
            batch_size=batch_size,
            shuffle=shuffle,
            samples=samples,
            drop_last=drop_last,
            num_workers=num_workers,
            use_process=use_process,
            use_padded_im_info=use_padded_im_info)
        self.mode = 'TEST'


@register
class SSDTrainFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=VocDataSet().__dict__,
                 fields=['image', 'gt_box', 'gt_label'],
                 image_shape=[3, 300, 300],
                 sample_transforms=[
                     DecodeImage(to_rgb=True, with_mixup=False),
                     NormalizeBox(),
                     RandomDistort(brightness_lower=0.875,
                                   brightness_upper=1.125,
                                   is_order=True),
                     ExpandImage(max_ratio=4, prob=0.5),
                     CropImage(batch_sampler=[[1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 0.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 0.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 0.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 0.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 0.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0]],
                               satisfy_all=False, avoid_no_bbox=False),
                     ResizeImage(target_size=300, use_cv2=False, interp=1),
                     RandomFlipImage(is_normalized=True),
                     Permute(),
                     NormalizeImage(mean=[127.5, 127.5, 127.5],
                                    std=[127.502231, 127.502231, 127.502231],
                                    is_scale=False)
                 ],
                 batch_transforms=[],
                 batch_size=32,
                 shuffle=True,
                 samples=-1,
                 drop_last=True,
                 num_workers=8,
                 bufsize=10,
                 use_process=True,
                 memsize=None):
        sample_transforms.append(ArrangeSSD())
        super(SSDTrainFeed, self).__init__(
            dataset,
            fields,
            image_shape,
            sample_transforms,
            batch_transforms,
            batch_size=batch_size,
            shuffle=shuffle,
            samples=samples,
            drop_last=drop_last,
            num_workers=num_workers,
            bufsize=bufsize,
            use_process=use_process,
            memsize=None)
        self.mode = 'TRAIN'


@register
class SSDEvalFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(
            self,
            dataset=VocDataSet(VOC_VAL_ANNOTATION).__dict__,
            fields=['image', 'im_shape', 'im_id', 'gt_box',
                         'gt_label', 'is_difficult'],
            image_shape=[3, 300, 300],
            sample_transforms=[
                DecodeImage(to_rgb=True, with_mixup=False),
                NormalizeBox(),
                ResizeImage(target_size=300, use_cv2=False, interp=1),
                Permute(),
                NormalizeImage(
                    mean=[127.5, 127.5, 127.5],
                    std=[127.502231, 127.502231, 127.502231],
                    is_scale=False)
            ],
            batch_transforms=[],
            batch_size=64,
            shuffle=False,
            samples=-1,
            drop_last=True,
            num_workers=8,
            bufsize=10,
            use_process=False,
            memsize=None):
        sample_transforms.append(ArrangeEvalSSD())
        super(SSDEvalFeed, self).__init__(
            dataset,
            fields,
            image_shape,
            sample_transforms,
            batch_transforms,
            batch_size=batch_size,
            shuffle=shuffle,
            samples=samples,
            drop_last=drop_last,
            num_workers=num_workers,
            bufsize=bufsize,
            use_process=use_process,
            memsize=memsize)
        self.mode = 'VAL'


@register
class SSDTestFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=SimpleDataSet(VOC_TEST_ANNOTATION).__dict__,
                 fields=['image', 'im_id', 'im_shape'],
                 image_shape=[3, 300, 300],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     ResizeImage(target_size=300, use_cv2=False, interp=1),
                     Permute(),
                     NormalizeImage(
                         mean=[127.5, 127.5, 127.5],
                         std=[127.502231, 127.502231, 127.502231],
                         is_scale=False)
                 ],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 samples=-1,
                 drop_last=False,
                 num_workers=8,
                 bufsize=10,
                 use_process=False,
                 memsize=None):
        sample_transforms.append(ArrangeTestSSD())
        if isinstance(dataset, dict):
            dataset = SimpleDataSet(**dataset)
        super(SSDTestFeed, self).__init__(
            dataset,
            fields,
            image_shape,
            sample_transforms,
            batch_transforms,
            batch_size=batch_size,
            shuffle=shuffle,
            samples=samples,
            drop_last=drop_last,
            num_workers=num_workers,
            bufsize=bufsize,
            use_process=use_process,
            memsize=memsize)
        self.mode = 'TEST'


@register
class YoloTrainFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=CocoDataSet().__dict__,
                 fields=['image', 'gt_box', 'gt_label', 'gt_score'],
                 image_shape=[3, 608, 608],
                 sample_transforms=[
                     DecodeImage(to_rgb=True, with_mixup=True),
                     MixupImage(alpha=1.5, beta=1.5),
                     NormalizeBox(),
                     RandomDistort(),
                     ExpandImage(max_ratio=4., prob=.5,
                                 mean=[123.675, 116.28, 103.53]),
                     CropImage([[1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 1.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 1.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 1.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 1.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 1.0],
                                [1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0]]),
                     RandomInterpImage(target_size=608),
                     RandomFlipImage(is_normalized=True),
                     NormalizeImage(
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         is_scale=True,
                         is_channel_first=False),
                     Permute(to_bgr=False),
                 ],
                 batch_transforms=[
                     RandomShape(sizes=[
                         320, 352, 384, 416, 448, 480, 512, 544, 576, 608
                     ])
                 ],
                 batch_size=8,
                 shuffle=True,
                 samples=-1,
                 drop_last=True,
                 with_background=False,
                 num_workers=8,
                 bufsize=128,
                 use_process=True,
                 memsize=None,
                 num_max_boxes=50,
                 mixup_epoch=250):
        sample_transforms.append(ArrangeYOLO())
        super(YoloTrainFeed, self).__init__(
            dataset,
            fields,
            image_shape,
            sample_transforms,
            batch_transforms,
            batch_size=batch_size,
            shuffle=shuffle,
            samples=samples,
            drop_last=drop_last,
            with_background=with_background,
            num_workers=num_workers,
            bufsize=bufsize,
            use_process=use_process,
            memsize=memsize)
        self.num_max_boxes = num_max_boxes
        self.mixup_epoch = mixup_epoch
        self.mode = 'TRAIN'


@register
class YoloEvalFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=CocoDataSet(COCO_VAL_ANNOTATION,
                                     COCO_VAL_IMAGE_DIR).__dict__,
                 fields=['image', 'im_size', 'im_id', 'gt_box',
                         'gt_label', 'is_difficult'],
                 image_shape=[3, 608, 608],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     ResizeImage(target_size=608, interp=2),
                     NormalizeImage(
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         is_scale=True,
                         is_channel_first=False),
                     Permute(to_bgr=False),
                 ],
                 batch_transforms=[],
                 batch_size=8,
                 shuffle=False,
                 samples=-1,
                 drop_last=False,
                 with_background=False,
                 num_workers=8,
                 num_max_boxes=50,
                 use_process=False,
                 memsize=None):
        sample_transforms.append(ArrangeEvalYOLO())
        super(YoloEvalFeed, self).__init__(
            dataset,
            fields,
            image_shape,
            sample_transforms,
            batch_transforms,
            batch_size=batch_size,
            shuffle=shuffle,
            samples=samples,
            drop_last=drop_last,
            with_background=with_background,
            num_workers=num_workers,
            use_process=use_process,
            memsize=memsize)
        self.num_max_boxes = num_max_boxes
        self.mode = 'VAL'
        self.bufsize = 128

        # support image shape config, resize image with image_shape
        for i, trans in enumerate(sample_transforms):
            if isinstance(trans, ResizeImage):
                sample_transforms[i] = ResizeImage(
                        target_size=self.image_shape[-1],
                        interp=trans.interp)


@register
class YoloTestFeed(DataFeed):
    __doc__ = DataFeed.__doc__

    def __init__(self,
                 dataset=SimpleDataSet(COCO_VAL_ANNOTATION,
                                       COCO_VAL_IMAGE_DIR).__dict__,
                 fields=['image', 'im_size', 'im_id'],
                 image_shape=[3, 608, 608],
                 sample_transforms=[
                     DecodeImage(to_rgb=True),
                     ResizeImage(target_size=608, interp=2),
                     NormalizeImage(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225],
                                    is_scale=True,
                                    is_channel_first=False),
                     Permute(to_bgr=False),
                 ],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 samples=-1,
                 drop_last=False,
                 with_background=False,
                 num_workers=8,
                 num_max_boxes=50,
                 use_process=False,
                 memsize=None):
        sample_transforms.append(ArrangeTestYOLO())
        if isinstance(dataset, dict):
            dataset = SimpleDataSet(**dataset)
        super(YoloTestFeed, self).__init__(
            dataset,
            fields,
            image_shape,
            sample_transforms,
            batch_transforms,
            batch_size=batch_size,
            shuffle=shuffle,
            samples=samples,
            drop_last=drop_last,
            with_background=with_background,
            num_workers=num_workers,
            use_process=use_process,
            memsize=memsize)
        self.mode = 'TEST'
        self.bufsize = 128

        # support image shape config, resize image with image_shape
        for i, trans in enumerate(sample_transforms):
            if isinstance(trans, ResizeImage):
                sample_transforms[i] = ResizeImage(
                        target_size=self.image_shape[-1],
                        interp=trans.interp)
# yapf: enable
