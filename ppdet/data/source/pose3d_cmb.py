# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
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
class Pose3DDataset(DetDataset):
    """Pose3D Dataset class. 

    Args:
        dataset_dir (str): Root path to the dataset.
        anno_list (list of str): each of the element is a relative path to the annotation file.
        image_dirs (list of str): each of path is a relative path where images are held.
        transform (composed(operators)): A sequence of data transforms.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
        24 joints order:
        0-2: 'R_Ankle', 'R_Knee', 'R_Hip', 
        3-5:'L_Hip', 'L_Knee', 'L_Ankle', 
        6-8:'R_Wrist', 'R_Elbow', 'R_Shoulder', 
        9-11:'L_Shoulder','L_Elbow','L_Wrist',
        12-14:'Neck','Top_of_Head','Pelvis',
        15-18:'Thorax','Spine','Jaw','Head',
        19-23:'Nose','L_Eye','R_Eye','L_Ear','R_Ear'
    """

    def __init__(self,
                 dataset_dir,
                 image_dirs,
                 anno_list,
                 transform=[],
                 num_joints=24,
                 test_mode=False):
        super().__init__(dataset_dir, image_dirs, anno_list)
        self.image_info = {}
        self.ann_info = {}
        self.num_joints = num_joints

        self.transform = transform
        self.test_mode = test_mode

        self.img_ids = []
        self.dataset_dir = dataset_dir
        self.image_dirs = image_dirs
        self.anno_list = anno_list

    def get_mask(self, mvm_percent=0.3):
        num_joints = self.num_joints
        mjm_mask = np.ones((num_joints, 1)).astype(np.float32)
        if self.test_mode == False:
            pb = np.random.random_sample()
            masked_num = int(
                pb * mvm_percent *
                num_joints)  # at most x% of the joints could be masked
            indices = np.random.choice(
                np.arange(num_joints), replace=False, size=masked_num)
            mjm_mask[indices, :] = 0.0

        mvm_mask = np.ones((10, 1)).astype(np.float32)
        if self.test_mode == False:
            num_vertices = 10
            pb = np.random.random_sample()
            masked_num = int(
                pb * mvm_percent *
                num_vertices)  # at most x% of the vertices could be masked
            indices = np.random.choice(
                np.arange(num_vertices), replace=False, size=masked_num)
            mvm_mask[indices, :] = 0.0

        mjm_mask = np.concatenate([mjm_mask, mvm_mask], axis=0)
        return mjm_mask

    def filterjoints(self, x):
        if self.num_joints == 24:
            return x
        elif self.num_joints == 14:
            return x[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18], :]
        elif self.num_joints == 17:
            return x[
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 19], :]
        else:
            raise ValueError(
                "unsupported joint numbers, only [24 or 17 or 14] is supported!")

    def parse_dataset(self):
        print("Loading annotations..., please wait")
        self.annos = []
        im_id = 0
        for idx, annof in enumerate(self.anno_list):
            img_prefix = os.path.join(self.dataset_dir, self.image_dirs[idx])
            dataf = os.path.join(self.dataset_dir, annof)
            with open(dataf, 'r') as rf:
                anno_data = json.load(rf)
                annos = anno_data['data']
                new_annos = []
                print("{} has annos numbers: {}".format(dataf, len(annos)))
                for anno in annos:
                    new_anno = {}
                    new_anno['im_id'] = im_id
                    im_id += 1
                    imagename = anno['imageName']
                    if imagename.startswith("COCO_train2014_"):
                        imagename = imagename[len("COCO_train2014_"):]
                    elif imagename.startswith("COCO_val2014_"):
                        imagename = imagename[len("COCO_val2014_"):]
                    imagename = os.path.join(img_prefix, imagename)
                    if not os.path.exists(imagename):
                        if "train2017" in imagename:
                            imagename = imagename.replace("train2017",
                                                          "val2017")
                            if not os.path.exists(imagename):
                                print("cannot find imagepath:{}".format(
                                    imagename))
                                continue
                        else:
                            print("cannot find imagepath:{}".format(imagename))
                            continue
                    new_anno['imageName'] = imagename
                    new_anno['bbox_center'] = anno['bbox_center']
                    new_anno['bbox_scale'] = anno['bbox_scale']
                    new_anno['joints_2d'] = np.array(anno[
                        'gt_keypoint_2d']).astype(np.float32)
                    if new_anno['joints_2d'].shape[0] == 49:
                        #if the joints_2d is in SPIN format(which generated by eft), choose the last 24 public joints
                        #for detail please refer: https://github.com/nkolot/SPIN/blob/master/constants.py
                        new_anno['joints_2d'] = new_anno['joints_2d'][25:]
                    new_anno['joints_3d'] = np.array(anno[
                        'pose3d'])[:, :3].astype(np.float32)
                    new_anno['mjm_mask'] = self.get_mask()
                    if not 'has_3d_joints' in anno:
                        new_anno['has_3d_joints'] = int(1)
                        new_anno['has_2d_joints'] = int(1)
                    else:
                        new_anno['has_3d_joints'] = int(anno['has_3d_joints'])
                        new_anno['has_2d_joints'] = int(anno['has_2d_joints'])
                    new_anno['joints_2d'] = self.filterjoints(new_anno[
                        'joints_2d'])
                    self.annos.append(new_anno)
                del annos

    def __len__(self):
        """Get dataset length."""
        return len(self.annos)

    def _get_imganno(self, idx):
        """Get anno for a single image."""
        return self.annos[idx]

    def __getitem__(self, idx):
        """Prepare image for training given the index."""
        records = copy.deepcopy(self._get_imganno(idx))
        imgpath = records['imageName']
        assert os.path.exists(imgpath), "cannot find image {}".format(imgpath)
        records['image'] = cv2.imread(imgpath)
        records['image'] = cv2.cvtColor(records['image'], cv2.COLOR_BGR2RGB)
        records = self.transform(records)
        return records

    def check_or_download_dataset(self):
        alldatafind = True
        for image_dir in self.image_dirs:
            image_dir = os.path.join(self.dataset_dir, image_dir)
            if not os.path.isdir(image_dir):
                print("dataset [{}] is not found".format(image_dir))
                alldatafind = False
        if not alldatafind:
            raise ValueError(
                "Some dataset is not valid and cannot download automatically now, please prepare the dataset first"
            )
