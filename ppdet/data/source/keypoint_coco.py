# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
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
import cv2
import numpy as np
import copy
# TODO: unify xtococotools and pycocotools
import pycocotools
from pycocotools.coco import COCO
from .dataset import DetDataset
from ppdet.core.workspace import register, serializable


@serializable
class KeypointBottomUpBaseDataset(DetDataset):
    """Base class for bottom-up datasets.

    All datasets should subclass it.
    All subclasses should overwrite:
        Methods:`_get_imganno`

    Args:
        dataset_dir (str): Root path to the dataset.
        anno_path (str): Relative path to the annotation file.
        image_dir (str): Path to a directory where images are held.
            Default: None.
        num_joints (int): keypoint numbers
        transform (composed(operators)): A sequence of data transforms.
        shard (list): [rank, worldsize], the distributed env params
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 dataset_dir,
                 image_dir,
                 anno_path,
                 num_joints,
                 transform=[],
                 shard=[0, 1],
                 test_mode=False):
        super().__init__(dataset_dir, image_dir, anno_path)
        self.image_info = {}
        self.ann_info = {}

        self.img_prefix = os.path.join(dataset_dir, image_dir)
        self.transform = transform
        self.test_mode = test_mode

        self.ann_info['num_joints'] = num_joints

        self.img_ids = []

    def __len__(self):
        """Get dataset length."""
        return len(self.img_ids)

    def _get_imganno(self, idx):
        """Get anno for a single image."""
        raise NotImplementedError

    def __getitem__(self, idx):
        """Prepare image for training given the index."""
        records = copy.deepcopy(self._get_imganno(idx))
        records['image'] = cv2.imread(records['image_file'])
        records['image'] = cv2.cvtColor(records['image'], cv2.COLOR_BGR2RGB)
        records['mask'] = (records['mask'] + 0).astype('uint8')
        records = self.transform(records)
        return records

    def parse_dataset(self):
        return


@register
@serializable
class KeypointBottomUpCocoDataset(KeypointBottomUpBaseDataset):
    """COCO dataset for bottom-up pose estimation.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    COCO keypoint indexes::

        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'

    Args:
        dataset_dir (str): Root path to the dataset.
        anno_path (str): Relative path to the annotation file.
        image_dir (str): Path to a directory where images are held.
            Default: None.
        num_joints (int): keypoint numbers
        transform (composed(operators)): A sequence of data transforms.
        shard (list): [rank, worldsize], the distributed env params
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 dataset_dir,
                 image_dir,
                 anno_path,
                 num_joints,
                 transform=[],
                 shard=[0, 1],
                 test_mode=False):
        super().__init__(dataset_dir, image_dir, anno_path, num_joints,
                         transform, shard, test_mode)

        ann_file = os.path.join(dataset_dir, anno_path)
        self.coco = COCO(ann_file)

        self.img_ids = self.coco.getImgIds()
        if not test_mode:
            self.img_ids = [
                img_id for img_id in self.img_ids
                if len(self.coco.getAnnIds(
                    imgIds=img_id, iscrowd=None)) > 0
            ]
        blocknum = int(len(self.img_ids) / shard[1])
        self.img_ids = self.img_ids[(blocknum * shard[0]):(blocknum * (shard[0]
                                                                       + 1))]
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.dataset_name = 'coco'

        cat_ids = self.coco.getCatIds()
        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        print(f'=> num_images: {self.num_images}')

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def _get_imganno(self, idx):
        """Get anno for a single image.

        Args:
            idx (int): image idx

        Returns:
            dict: info for model training
        """
        coco = self.coco
        img_id = self.img_ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)

        mask = self._get_mask(anno, idx)
        anno = [
            obj for obj in anno
            if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
        ]

        joints, orgsize = self._get_joints(anno, idx)

        db_rec = {}
        db_rec['im_id'] = img_id
        db_rec['image_file'] = os.path.join(self.img_prefix,
                                            self.id2name[img_id])
        db_rec['mask'] = mask
        db_rec['joints'] = joints
        db_rec['im_shape'] = orgsize

        return db_rec

    def _get_joints(self, anno, idx):
        """Get joints for all people in an image."""
        num_people = len(anno)

        joints = np.zeros(
            (num_people, self.ann_info['num_joints'], 3), dtype=np.float32)

        for i, obj in enumerate(anno):
            joints[i, :self.ann_info['num_joints'], :3] = \
                np.array(obj['keypoints']).reshape([-1, 3])

        img_info = self.coco.loadImgs(self.img_ids[idx])[0]
        joints[..., 0] /= img_info['width']
        joints[..., 1] /= img_info['height']
        orgsize = np.array([img_info['height'], img_info['width']])

        return joints, orgsize

    def _get_mask(self, anno, idx):
        """Get ignore masks to mask out losses."""
        coco = self.coco
        img_info = coco.loadImgs(self.img_ids[idx])[0]

        m = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)

        for obj in anno:
            if 'segmentation' in obj:
                if obj['iscrowd']:
                    rle = pycocotools.mask.frPyObjects(obj['segmentation'],
                                                       img_info['height'],
                                                       img_info['width'])
                    m += pycocotools.mask.decode(rle)
                elif obj['num_keypoints'] == 0:
                    rles = pycocotools.mask.frPyObjects(obj['segmentation'],
                                                        img_info['height'],
                                                        img_info['width'])
                    for rle in rles:
                        m += pycocotools.mask.decode(rle)

        return m < 0.5


@register
@serializable
class KeypointBottomUpCrowdPoseDataset(KeypointBottomUpCocoDataset):
    """CrowdPose dataset for bottom-up pose estimation.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    CrowdPose keypoint indexes::

        0: 'left_shoulder',
        1: 'right_shoulder',
        2: 'left_elbow',
        3: 'right_elbow',
        4: 'left_wrist',
        5: 'right_wrist',
        6: 'left_hip',
        7: 'right_hip',
        8: 'left_knee',
        9: 'right_knee',
        10: 'left_ankle',
        11: 'right_ankle',
        12: 'top_head',
        13: 'neck'

    Args:
        dataset_dir (str): Root path to the dataset.
        anno_path (str): Relative path to the annotation file.
        image_dir (str): Path to a directory where images are held.
            Default: None.
        num_joints (int): keypoint numbers
        transform (composed(operators)): A sequence of data transforms.
        shard (list): [rank, worldsize], the distributed env params
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 dataset_dir,
                 image_dir,
                 anno_path,
                 num_joints,
                 transform=[],
                 shard=[0, 1],
                 test_mode=False):
        super().__init__(dataset_dir, image_dir, anno_path, num_joints,
                         transform, shard, test_mode)

        ann_file = os.path.join(dataset_dir, anno_path)

        self.coco = COCO(ann_file)

        self.img_ids = self.coco.getImgIds()
        if not test_mode:
            self.img_ids = [
                img_id for img_id in self.img_ids
                if len(self.coco.getAnnIds(
                    imgIds=img_id, iscrowd=None)) > 0
            ]
        blocknum = int(len(self.img_ids) / shard[1])
        self.img_ids = self.img_ids[(blocknum * shard[0]):(blocknum * (shard[0]
                                                                       + 1))]
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)

        self.dataset_name = 'crowdpose'
        print(f'=> num_images: {self.num_images}')
