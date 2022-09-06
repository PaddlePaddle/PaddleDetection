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

import os
import copy
import numpy as np
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from paddle.io import Dataset
from ppdet.core.workspace import register, serializable
from ppdet.utils.download import get_dataset_path
from ppdet.data import source

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


@serializable
class DetDataset(Dataset):
    """
    Load detection dataset.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        use_default_label (bool): whether to load default label list.
        repeat (int): repeat times for dataset, use in benchmark.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 use_default_label=None,
                 repeat=1,
                 **kwargs):
        super(DetDataset, self).__init__()
        self.dataset_dir = dataset_dir if dataset_dir is not None else ''
        self.anno_path = anno_path
        self.image_dir = image_dir if image_dir is not None else ''
        self.data_fields = data_fields
        self.sample_num = sample_num
        self.use_default_label = use_default_label
        self.repeat = repeat
        self._epoch = 0
        self._curr_iter = 0

    def __len__(self, ):
        return len(self.roidbs) * self.repeat

    def __call__(self, *args, **kwargs):
        return self

    def __getitem__(self, idx):
        n = len(self.roidbs)
        if self.repeat > 1:
            idx %= n
        # data batch
        roidb = copy.deepcopy(self.roidbs[idx])
        if self.mixup_epoch == 0 or self._epoch < self.mixup_epoch:
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.cutmix_epoch == 0 or self._epoch < self.cutmix_epoch:
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.mosaic_epoch == 0 or self._epoch < self.mosaic_epoch:
            roidb = [roidb, ] + [
                copy.deepcopy(self.roidbs[np.random.randint(n)])
                for _ in range(4)
            ]
        if isinstance(roidb, Sequence):
            for r in roidb:
                r['curr_iter'] = self._curr_iter
        else:
            roidb['curr_iter'] = self._curr_iter
        self._curr_iter += 1

        return self.transform(roidb)

    def check_or_download_dataset(self):
        self.dataset_dir = get_dataset_path(self.dataset_dir, self.anno_path,
                                            self.image_dir)

    def set_kwargs(self, **kwargs):
        self.mixup_epoch = kwargs.get('mixup_epoch', -1)
        self.cutmix_epoch = kwargs.get('cutmix_epoch', -1)
        self.mosaic_epoch = kwargs.get('mosaic_epoch', -1)

    def set_transform(self, transform):
        self.transform = transform

    def set_epoch(self, epoch_id):
        self._epoch = epoch_id

    def parse_dataset(self, ):
        raise NotImplementedError(
            "Need to implement parse_dataset method of Dataset")

    def get_anno(self):
        if self.anno_path is None:
            return
        return os.path.join(self.dataset_dir, self.anno_path)


def _is_valid_file(f, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    return f.lower().endswith(extensions)


def _make_dataset(dir):
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        raise ('{} should be a dir'.format(dir))
    images = []
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if _is_valid_file(path):
                images.append(path)
    return images


@register
@serializable
class ImageFolder(DetDataset):
    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 use_default_label=None,
                 **kwargs):
        super(ImageFolder, self).__init__(
            dataset_dir,
            image_dir,
            anno_path,
            sample_num=sample_num,
            use_default_label=use_default_label)
        self._imid2path = {}
        self.roidbs = None
        self.sample_num = sample_num

    def check_or_download_dataset(self):
        return

    def get_anno(self):
        if self.anno_path is None:
            return
        if self.dataset_dir:
            return os.path.join(self.dataset_dir, self.anno_path)
        else:
            return self.anno_path

    def parse_dataset(self, ):
        if not self.roidbs:
            self.roidbs = self._load_images()

    def _parse(self):
        image_dir = self.image_dir
        if not isinstance(image_dir, Sequence):
            image_dir = [image_dir]
        images = []
        for im_dir in image_dir:
            if os.path.isdir(im_dir):
                im_dir = os.path.join(self.dataset_dir, im_dir)
                images.extend(_make_dataset(im_dir))
            elif os.path.isfile(im_dir) and _is_valid_file(im_dir):
                images.append(im_dir)
        return images

    def _load_images(self):
        images = self._parse()
        ct = 0
        records = []
        for image in images:
            assert image != '' and os.path.isfile(image), \
                    "Image {} not found".format(image)
            if self.sample_num > 0 and ct >= self.sample_num:
                break
            rec = {'im_id': np.array([ct]), 'im_file': image}
            self._imid2path[ct] = image
            ct += 1
            records.append(rec)
        assert len(records) > 0, "No image file found"
        return records

    def get_imid2path(self):
        return self._imid2path

    def set_images(self, images):
        self.image_dir = images
        self.roidbs = self._load_images()

    def set_slice_images(self,
                         images,
                         slice_size=[640, 640],
                         overlap_ratio=[0.25, 0.25]):
        self.image_dir = images
        ori_records = self._load_images()
        try:
            import sahi
            from sahi.slicing import slice_image
        except Exception as e:
            logger.error(
                'sahi not found, plaese install sahi. '
                'for example: `pip install sahi`, see https://github.com/obss/sahi.'
            )
            raise e

        sub_img_ids = 0
        ct = 0
        ct_sub = 0
        records = []
        for i, ori_rec in enumerate(ori_records):
            im_path = ori_rec['im_file']
            slice_image_result = sahi.slicing.slice_image(
                image=im_path,
                slice_height=slice_size[0],
                slice_width=slice_size[1],
                overlap_height_ratio=overlap_ratio[0],
                overlap_width_ratio=overlap_ratio[1])

            sub_img_num = len(slice_image_result)
            for _ind in range(sub_img_num):
                im = slice_image_result.images[_ind]
                rec = {
                    'image': im,
                    'im_id': np.array([sub_img_ids + _ind]),
                    'h': im.shape[0],
                    'w': im.shape[1],
                    'ori_im_id': np.array([ori_rec['im_id'][0]]),
                    'st_pix': np.array(
                        slice_image_result.starting_pixels[_ind],
                        dtype=np.float32),
                    'is_last': 1 if _ind == sub_img_num - 1 else 0,
                } if 'image' in self.data_fields else {}
                records.append(rec)
            ct_sub += sub_img_num
            ct += 1
        print('{} samples and slice to {} sub_samples'.format(ct, ct_sub))
        self.roidbs = records

    def get_label_list(self):
        # Only VOC dataset needs label list in ImageFold 
        return self.anno_path


@register
class CommonDataset(object):
    def __init__(self, **dataset_args):
        super(CommonDataset, self).__init__()
        dataset_args = copy.deepcopy(dataset_args)
        type = dataset_args.pop("name")
        self.dataset = getattr(source, type)(**dataset_args)

    def __call__(self):
        return self.dataset


@register
class TrainDataset(CommonDataset):
    pass


@register
class EvalMOTDataset(CommonDataset):
    pass


@register
class TestMOTDataset(CommonDataset):
    pass


@register
class EvalDataset(CommonDataset):
    pass


@register
class TestDataset(CommonDataset):
    pass
