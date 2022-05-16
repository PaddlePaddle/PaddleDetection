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

# This code is just for retinaface version of WIDERFace, which contains 5 
# kerpoints for each valid face
import os
import numpy as np

from ppdet.core.workspace import register, serializable
from .dataset import DetDataset

from ppdet.utils.logger import setup_logger

logger = setup_logger(__name__)


@register
@serializable
class WIDERFaceDataSetV2(DetDataset):
    """
    Load WiderFace records with 'anno_path'

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): WiderFace annotation data.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        with_lmk (bool): whether to load face landmark keypoint labels.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 min_size=None,
                 test_mode=False,
                 nk=5,
                 with_lmk=True):
        super(WIDERFaceDataSetV2, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            with_lmk=with_lmk)
        self.anno_path = anno_path
        self.sample_num = sample_num
        self.roidbs = None
        self.cname2cid = None
        self.with_lmk = with_lmk
        self.NK = nk
        self.min_size = min_size
        self.test_mode = test_mode

    def _parse_ann_line(self, line):
        values = [float(x) for x in line.strip().split()]
        bbox = np.array(values[0:4], dtype=np.float32)
        kps = np.zeros((self.NK, 3), dtype=np.float32)
        ignore = False
        if self.min_size is not None:
            assert not self.test_mode
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < self.min_size or h < self.min_size:
                ignore = True
        if len(values) > 4:
            if len(values) > 5:
                #print(values)
                kps = np.array(
                    values[4:19], dtype=np.float32).reshape((self.NK, 3))
                for li in range(kps.shape[0]):
                    if (kps[li, :] == -1).all():
                        #assert kps[li][2]==-1
                        kps[li][2] = 0.0  #weight = 0, ignore
                    else:
                        assert kps[li][2] >= 0
                        kps[li][2] = 1.0  #weight
                        #if li==0:
                        #  landmark_num+=1
                        #if kps[li][2]==0.0:#visible
                        #  kps[li][2] = 1.0
                        #else:
                        #  kps[li][2] = 0.0
            else:  #len(values)==5
                if not ignore:
                    ignore = (values[4] == 1)
        else:
            assert self.test_mode
        return [bbox, kps, ignore]

    def parse_dataset(self):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        ann_file = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)
        name = None
        bbox_map = {}
        for line in open(ann_file, 'r'):
            line = line.strip()
            if line.startswith('#'):
                value = line[1:].strip().split()
                name = value[0]
                width = int(value[1])
                height = int(value[2])

                bbox_map[name] = dict(width=width, height=height, objs=[])
                continue
            assert name is not None
            assert name in bbox_map
            bbox_map[name]['objs'].append(line)
        print('origin image size', len(bbox_map))
        data_infos = []
        ct = 0
        for name in bbox_map:
            item = bbox_map[name]
            width = item['width']
            height = item['height']
            vals = item['objs']
            objs = []
            bbox = np.zeros((len(vals), 4), dtype=np.float32)
            lms = np.zeros((len(vals), self.NK, 3), dtype=np.float32)
            for i, line in enumerate(vals):
                data = self._parse_ann_line(line)
                if data is None:
                    continue
                objs.append(data)  #data is (bbox, kps, cat)
                bbox[i] = data[0]
                #  lms[i] = data[1][..., :2].reshape(-1)
                lms[i] = data[1]

            if len(bbox) == 0 and not self.test_mode:
                continue
            gt_class = np.zeros((len(objs), 1), dtype=np.int32)
            im_fname = os.path.join(image_dir, name) if image_dir else name
            data_infos.append(
                dict(
                    im_file=im_fname,
                    im_id=np.array([ct]),
                    gt_class=gt_class,
                    gt_bbox=bbox,
                    gt_keypoint=lms))
            ct += 1
        assert len(data_infos)
        self.roidbs, self.cname2cid = data_infos, widerface_label()


def widerface_label():
    labels_map = {'face': 0}
    return labels_map
