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
"""
this code is base on https://github.com/open-mmlab/mmpose
"""
import os
import cv2
import numpy as np
import json
import copy
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

    def parse_dataset(self):
        pass

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
        if 'mask' in records:
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
                 test_mode=False,
                 return_mask=True,
                 return_bbox=True,
                 return_area=True,
                 return_class=True):
        super().__init__(dataset_dir, image_dir, anno_path, num_joints,
                         transform, shard, test_mode)

        self.ann_file = os.path.join(dataset_dir, anno_path)
        self.shard = shard
        self.test_mode = test_mode
        self.return_mask = return_mask
        self.return_bbox = return_bbox
        self.return_area = return_area
        self.return_class = return_class

    def parse_dataset(self):
        self.coco = COCO(self.ann_file)

        self.img_ids = self.coco.getImgIds()
        if not self.test_mode:
            self.img_ids_tmp = []
            for img_id in self.img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anno = self.coco.loadAnns(ann_ids)
                anno = [obj for obj in anno if obj['iscrowd'] == 0]
                if len(anno) == 0:
                    continue
                self.img_ids_tmp.append(img_id)
            self.img_ids = self.img_ids_tmp

        blocknum = int(len(self.img_ids) / self.shard[1])
        self.img_ids = self.img_ids[(blocknum * self.shard[0]):(blocknum * (
            self.shard[0] + 1))]
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.dataset_name = 'coco'

        cat_ids = self.coco.getCatIds()
        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        print('=> num_images: {}'.format(self.num_images))

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

        anno = [
            obj for obj in anno
            if obj['iscrowd'] == 0 and obj['num_keypoints'] > 0
        ]

        db_rec = {}
        joints, orgsize = self._get_joints(anno, idx)
        db_rec['gt_joints'] = joints
        db_rec['im_shape'] = orgsize

        if self.return_bbox:
            db_rec['gt_bbox'] = self._get_bboxs(anno, idx)

        if self.return_class:
            db_rec['gt_class'] = self._get_labels(anno, idx)

        if self.return_area:
            db_rec['gt_areas'] = self._get_areas(anno, idx)

        if self.return_mask:
            db_rec['mask'] = self._get_mask(anno, idx)

        db_rec['im_id'] = img_id
        db_rec['image_file'] = os.path.join(self.img_prefix,
                                            self.id2name[img_id])

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
        orgsize = np.array([img_info['height'], img_info['width'], 1])

        return joints, orgsize

    def _get_bboxs(self, anno, idx):
        num_people = len(anno)
        gt_bboxes = np.zeros((num_people, 4), dtype=np.float32)

        for idx, obj in enumerate(anno):
            if 'bbox' in obj:
                gt_bboxes[idx, :] = obj['bbox']

        gt_bboxes[:, 2] += gt_bboxes[:, 0]
        gt_bboxes[:, 3] += gt_bboxes[:, 1]
        return gt_bboxes

    def _get_labels(self, anno, idx):
        num_people = len(anno)
        gt_labels = np.zeros((num_people, 1), dtype=np.float32)

        for idx, obj in enumerate(anno):
            if 'category_id' in obj:
                catid = obj['category_id']
                gt_labels[idx, 0] = self.catid2clsid[catid]
        return gt_labels

    def _get_areas(self, anno, idx):
        num_people = len(anno)
        gt_areas = np.zeros((num_people, ), dtype=np.float32)

        for idx, obj in enumerate(anno):
            if 'area' in obj:
                gt_areas[idx, ] = obj['area']
        return gt_areas

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

        self.ann_file = os.path.join(dataset_dir, anno_path)
        self.shard = shard
        self.test_mode = test_mode

    def parse_dataset(self):
        self.coco = COCO(self.ann_file)

        self.img_ids = self.coco.getImgIds()
        if not self.test_mode:
            self.img_ids = [
                img_id for img_id in self.img_ids
                if len(self.coco.getAnnIds(
                    imgIds=img_id, iscrowd=None)) > 0
            ]
        blocknum = int(len(self.img_ids) / self.shard[1])
        self.img_ids = self.img_ids[(blocknum * self.shard[0]):(blocknum * (
            self.shard[0] + 1))]
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)

        self.dataset_name = 'crowdpose'
        print('=> num_images: {}'.format(self.num_images))


@serializable
class KeypointTopDownBaseDataset(DetDataset):
    """Base class for top_down datasets.

    All datasets should subclass it.
    All subclasses should overwrite:
        Methods:`_get_db`

    Args:
        dataset_dir (str): Root path to the dataset.
        image_dir (str): Path to a directory where images are held.
        anno_path (str): Relative path to the annotation file.
        num_joints (int): keypoint numbers
        transform (composed(operators)): A sequence of data transforms.
    """

    def __init__(self,
                 dataset_dir,
                 image_dir,
                 anno_path,
                 num_joints,
                 transform=[]):
        super().__init__(dataset_dir, image_dir, anno_path)
        self.image_info = {}
        self.ann_info = {}

        self.img_prefix = os.path.join(dataset_dir, image_dir)
        self.transform = transform

        self.ann_info['num_joints'] = num_joints
        self.db = []

    def __len__(self):
        """Get dataset length."""
        return len(self.db)

    def _get_db(self):
        """Get a sample"""
        raise NotImplementedError

    def __getitem__(self, idx):
        """Prepare sample for training given the index."""
        records = copy.deepcopy(self.db[idx])
        records['image'] = cv2.imread(records['image_file'], cv2.IMREAD_COLOR |
                                      cv2.IMREAD_IGNORE_ORIENTATION)
        records['image'] = cv2.cvtColor(records['image'], cv2.COLOR_BGR2RGB)
        records['score'] = records['score'] if 'score' in records else 1
        records = self.transform(records)
        # print('records', records)
        return records


@register
@serializable
class KeypointTopDownCocoDataset(KeypointTopDownBaseDataset):
    """COCO dataset for top-down pose estimation. 

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    COCO keypoint indexes:

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
        image_dir (str): Path to a directory where images are held.
        anno_path (str): Relative path to the annotation file.
        num_joints (int): Keypoint numbers
        trainsize (list):[w, h] Image target size
        transform (composed(operators)): A sequence of data transforms.
        bbox_file (str): Path to a detection bbox file
            Default: None.
        use_gt_bbox (bool): Whether to use ground truth bbox
            Default: True.
        pixel_std (int): The pixel std of the scale
            Default: 200.
        image_thre (float): The threshold to filter the detection box
            Default: 0.0.
    """

    def __init__(self,
                 dataset_dir,
                 image_dir,
                 anno_path,
                 num_joints,
                 trainsize,
                 transform=[],
                 bbox_file=None,
                 use_gt_bbox=True,
                 pixel_std=200,
                 image_thre=0.0,
                 center_scale=None):
        super().__init__(dataset_dir, image_dir, anno_path, num_joints,
                         transform)

        self.bbox_file = bbox_file
        self.use_gt_bbox = use_gt_bbox
        self.trainsize = trainsize
        self.pixel_std = pixel_std
        self.image_thre = image_thre
        self.center_scale = center_scale
        self.dataset_name = 'coco'
        self.num_joints = num_joints

    def parse_dataset(self):
        if self.use_gt_bbox:
            self.db = self._load_coco_keypoint_annotations()
        else:
            self.db = self._load_coco_person_detection_results()

    def _load_coco_keypoint_annotations(self):
        coco = COCO(self.get_anno())
        img_ids = coco.getImgIds()
        gt_db = []
        for index in img_ids:
            im_ann = coco.loadImgs(index)[0]
            width = im_ann['width']
            height = im_ann['height']
            file_name = im_ann['file_name']
            im_id = int(im_ann["id"])

            annIds = coco.getAnnIds(imgIds=index, iscrowd=False)
            objs = coco.loadAnns(annIds)

            valid_objs = []
            for obj in objs:
                x, y, w, h = obj['bbox']
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    valid_objs.append(obj)
            objs = valid_objs

            rec = []
            for obj in objs:
                if max(obj['keypoints']) == 0:
                    continue

                joints = np.zeros(
                    (self.ann_info['num_joints'], 3), dtype=np.float32)
                joints_vis = np.zeros(
                    (self.ann_info['num_joints'], 3), dtype=np.float32)
                for ipt in range(self.ann_info['num_joints']):
                    joints[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                    joints[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                    joints[ipt, 2] = 0
                    t_vis = obj['keypoints'][ipt * 3 + 2]
                    if t_vis > 1:
                        t_vis = 1
                    joints_vis[ipt, 0] = t_vis
                    joints_vis[ipt, 1] = t_vis
                    joints_vis[ipt, 2] = 0

                center, scale = self._box2cs(obj['clean_bbox'][:4])
                rec.append({
                    'image_file': os.path.join(self.img_prefix, file_name),
                    'center': center,
                    'scale': scale,
                    'gt_joints': joints,
                    'joints_vis': joints_vis,
                    'im_id': im_id,
                })
            gt_db.extend(rec)

        return gt_db

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        aspect_ratio = self.trainsize[0] * 1.0 / self.trainsize[1]

        if self.center_scale is not None and np.random.rand() < 0.3:
            center += self.center_scale * (np.random.rand(2) - 0.5) * [w, h]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def _load_coco_person_detection_results(self):
        all_boxes = None
        bbox_file_path = os.path.join(self.dataset_dir, self.bbox_file)
        with open(bbox_file_path, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            print('=> Load %s fail!' % bbox_file_path)
            return None

        kpt_db = []
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            file_name = det_res[
                'filename'] if 'filename' in det_res else '%012d.jpg' % det_res[
                    'image_id']
            img_name = os.path.join(self.img_prefix, file_name)
            box = det_res['bbox']
            score = det_res['score']
            im_id = int(det_res['image_id'])

            if score < self.image_thre:
                continue

            center, scale = self._box2cs(box)
            joints = np.zeros(
                (self.ann_info['num_joints'], 3), dtype=np.float32)
            joints_vis = np.ones(
                (self.ann_info['num_joints'], 3), dtype=np.float32)
            kpt_db.append({
                'image_file': img_name,
                'im_id': im_id,
                'center': center,
                'scale': scale,
                'score': score,
                'gt_joints': joints,
                'joints_vis': joints_vis,
            })

        return kpt_db


@register
@serializable
class KeypointTopDownCocoWholeBodyHandDataset(KeypointTopDownBaseDataset):
    """CocoWholeBody dataset for top-down hand pose estimation. 

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    COCO-WholeBody Hand keypoint indexes:

        0: 'wrist',
        1: 'thumb1',
        2: 'thumb2',
        3: 'thumb3',
        4: 'thumb4',
        5: 'forefinger1',
        6: 'forefinger2',
        7: 'forefinger3',
        8: 'forefinger4',
        9: 'middle_finger1',
        10: 'middle_finger2',
        11: 'middle_finger3',
        12: 'middle_finger4',
        13: 'ring_finger1',
        14: 'ring_finger2',
        15: 'ring_finger3',
        16: 'ring_finger4',
        17: 'pinky_finger1',
        18: 'pinky_finger2',
        19: 'pinky_finger3',
        20: 'pinky_finger4'

    Args:
        dataset_dir (str): Root path to the dataset.
        image_dir (str): Path to a directory where images are held.
        anno_path (str): Relative path to the annotation file.
        num_joints (int): Keypoint numbers
        trainsize (list):[w, h] Image target size
        transform (composed(operators)): A sequence of data transforms.
        pixel_std (int): The pixel std of the scale
            Default: 200.
    """

    def __init__(self,
                 dataset_dir,
                 image_dir,
                 anno_path,
                 num_joints,
                 trainsize,
                 transform=[],
                 pixel_std=200):
        super().__init__(dataset_dir, image_dir, anno_path, num_joints,
                         transform)

        self.trainsize = trainsize
        self.pixel_std = pixel_std
        self.dataset_name = 'coco_wholebady_hand'

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        aspect_ratio = self.trainsize[0] * 1.0 / self.trainsize[1]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def parse_dataset(self):
        gt_db = []
        num_joints = self.ann_info['num_joints']
        coco = COCO(self.get_anno())
        img_ids = list(coco.imgs.keys())
        for img_id in img_ids:
            im_ann = coco.loadImgs(img_id)[0]
            image_file = os.path.join(self.img_prefix, im_ann['file_name'])
            im_id = int(im_ann["id"])

            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = coco.loadAnns(ann_ids)

            for obj in objs:
                for type in ['left', 'right']:
                    if (obj[f'{type}hand_valid'] and
                            max(obj[f'{type}hand_kpts']) > 0):

                        joints = np.zeros((num_joints, 3), dtype=np.float32)
                        joints_vis = np.zeros((num_joints, 3), dtype=np.float32)

                        keypoints = np.array(obj[f'{type}hand_kpts'])
                        keypoints = keypoints.reshape(-1, 3)
                        joints[:, :2] = keypoints[:, :2]
                        joints_vis[:, :2] = np.minimum(1, keypoints[:, 2:3])

                        center, scale = self._box2cs(obj[f'{type}hand_box'][:4])
                        gt_db.append({
                            'image_file': image_file,
                            'center': center,
                            'scale': scale,
                            'gt_joints': joints,
                            'joints_vis': joints_vis,
                            'im_id': im_id,
                        })

        self.db = gt_db


@register
@serializable
class KeypointTopDownMPIIDataset(KeypointTopDownBaseDataset):
    """MPII dataset for topdown pose estimation.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    MPII keypoint indexes::

        0: 'right_ankle',
        1: 'right_knee',
        2: 'right_hip',
        3: 'left_hip',
        4: 'left_knee',
        5: 'left_ankle',
        6: 'pelvis',
        7: 'thorax',
        8: 'upper_neck',
        9: 'head_top',
        10: 'right_wrist',
        11: 'right_elbow',
        12: 'right_shoulder',
        13: 'left_shoulder',
        14: 'left_elbow',
        15: 'left_wrist',

    Args:
        dataset_dir (str): Root path to the dataset.
        image_dir (str): Path to a directory where images are held.
        anno_path (str): Relative path to the annotation file.
        num_joints (int): Keypoint numbers
        trainsize (list):[w, h] Image target size
        transform (composed(operators)): A sequence of data transforms.
    """

    def __init__(self,
                 dataset_dir,
                 image_dir,
                 anno_path,
                 num_joints,
                 transform=[]):
        super().__init__(dataset_dir, image_dir, anno_path, num_joints,
                         transform)

        self.dataset_name = 'mpii'

    def parse_dataset(self):
        with open(self.get_anno()) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']
            im_id = a['image_id'] if 'image_id' in a else int(
                os.path.splitext(image_name)[0])

            c = np.array(a['center'], dtype=np.float32)
            s = np.array([a['scale'], a['scale']], dtype=np.float32)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25
            c = c - 1

            joints = np.zeros(
                (self.ann_info['num_joints'], 3), dtype=np.float32)
            joints_vis = np.zeros(
                (self.ann_info['num_joints'], 3), dtype=np.float32)
            if 'gt_joints' in a:
                joints_ = np.array(a['gt_joints'])
                joints_[:, 0:2] = joints_[:, 0:2] - 1
                joints_vis_ = np.array(a['joints_vis'])
                assert len(joints_) == self.ann_info[
                    'num_joints'], 'joint num diff: {} vs {}'.format(
                        len(joints_), self.ann_info['num_joints'])

                joints[:, 0:2] = joints_[:, 0:2]
                joints_vis[:, 0] = joints_vis_[:]
                joints_vis[:, 1] = joints_vis_[:]

            gt_db.append({
                'image_file': os.path.join(self.img_prefix, image_name),
                'im_id': im_id,
                'center': c,
                'scale': s,
                'gt_joints': joints,
                'joints_vis': joints_vis
            })
        print("number length: {}".format(len(gt_db)))
        self.db = gt_db
