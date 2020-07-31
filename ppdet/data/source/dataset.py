import os
import numpy as np
from collections import OrderedDict
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from paddle.io import Dataset
from ppdet.core.workspace import register, serializable
from ppdet.utils.download import get_dataset_path


@serializable
class DetDataset(Dataset):
    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 with_background=True,
                 sample_num=-1,
                 **kwargs):
        super(DetDataset, self).__init__()
        self.dataset_dir = dataset_dir if dataset_dir is not None else ''
        self.anno_path = anno_path
        self.image_dir = image_dir if image_dir is not None else ''
        self.with_background = with_background
        self.sample_num = sample_num

    def __len__(self, ):
        return len(self.roidbs)

    def __getitem__(self, idx):
        # data batch
        roidb = self.roidbs[idx]
        # data augment
        roidb = self.transform(roidb)
        # data item 
        out = OrderedDict()
        for k in self.fields:
            out[k] = roidb[k]
        return out.values()

    def set_out(self, sample_transform, fields):
        self.transform = sample_transform
        self.fields = fields

    def parse_dataset(self):
        raise NotImplemented(
            "Need to implement parse_dataset method of Dataset")


@register
@serializable
class ImageFolder(DetDataset):
    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 with_background=True,
                 sample_num=-1,
                 **kwargs):
        super(ImageFolder, self).__init__(dataset_dir, image_dir, anno_path,
                                          with_background, sample_num)

    def parse_dataset(self):
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
        self.roidbs = images
