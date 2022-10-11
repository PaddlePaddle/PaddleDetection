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
from paddle.io import DistributedBatchSampler


class CustomTempDBSampler(DistributedBatchSampler):
    """
    Custom DistributedBatchSampler for 3dpose temporal sample
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=True):
        self.tempsize = 4
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, \
                "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
                "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
                "drop_last should be a boolean number"

        from paddle.fluid.dygraph.parallel import ParallelEnv

        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, \
                    "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = ParallelEnv().nranks

        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, \
                    "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = ParallelEnv().local_rank

        self.drop_last = drop_last
        self.epoch = 0

        self.temp_num = self.dataset.get_temp_num()
        batch_size_temp = int(self.batch_size *
                              (self.temp_num / len(self.dataset)))
        self.batch_size_temp = (batch_size_temp //
                                self.tempsize) * self.tempsize
        self.batch_size_indep = self.batch_size - self.batch_size_temp
        self.total_batch = self.temp_num // (self.batch_size_temp * self.nranks)
        self.num_samples = self.total_batch * self.batch_size
        self.total_size = self.num_samples * self.nranks
        print("bs:{}, bst:{}, bsi:{}, totalbc:{}".format(
            self.batch_size, self.batch_size_temp, self.batch_size_indep,
            self.total_batch))

    def __iter__(self):
        tempblock = self.batch_size_temp // self.tempsize
        num_samples = len(self.dataset)
        indicestemp = np.arange(self.temp_num).tolist()
        indiceindep = np.arange(self.temp_num, num_samples).tolist()
        if len(indicestemp) % (self.batch_size_temp * self.nranks) != 0:
            indicestemp = indicestemp[:-(len(indicestemp) % (
                self.batch_size_temp * self.nranks))]
        subindicestemp = indicestemp[0:len(indicestemp):self.tempsize]
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(subindicestemp)
            np.random.RandomState(self.epoch).shuffle(indiceindep)
            self.epoch += 1
        if self.total_batch * self.batch_size_indep * self.nranks > len(
                indiceindep):
            indiceindep += indiceindep[:self.total_batch * self.batch_size_indep
                                       * self.nranks - len(indiceindep)]
        tempindices = []

        for idx in range(self.total_batch * self.nranks):
            for i in range(tempblock):
                tempindices.extend(indicestemp[subindicestemp[
                    idx * tempblock + i]:subindicestemp[idx * tempblock + i] +
                                               self.tempsize])
            tempindices.extend(indiceindep[idx * self.batch_size_indep:(idx + 1)
                                           * self.batch_size_indep])
            # print(tempindices[-self.batch_size:])

        print(
            len(tempindices), self.total_size,
            len(indicestemp), len(subindicestemp))

        # import pdb;pdb.set_trace()
        # assert len(tempindices) == self.total_size

        # subsample
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            last_batch_size = self.total_size % (self.batch_size * self.nranks)
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(self.local_rank * self.batch_size,
                           len(indices) - last_batch_size,
                           self.batch_size * self.nranks):
                subsampled_indices.extend(indices[i:i + self.batch_size])

            indices = indices[len(indices) - last_batch_size:]
            subsampled_indices.extend(indices[
                self.local_rank * last_local_batch_size:(
                    self.local_rank + 1) * last_local_batch_size])
            return subsampled_indices

        if self.nranks > 1:
            tempindices = _get_indices_by_batch_size(tempindices)

        # assert len(tempindices) == self.num_samples
        _sample_iter = iter(tempindices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices


class CustomTempDBSampler_onetempblock(DistributedBatchSampler):
    """
    Custom DistributedBatchSampler for 3dpose temporal sample
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=True):
        tempsize = 4
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, \
                "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
                "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
                "drop_last should be a boolean number"

        from paddle.fluid.dygraph.parallel import ParallelEnv

        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, \
                    "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = ParallelEnv().nranks

        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, \
                    "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = ParallelEnv().local_rank

        self.drop_last = drop_last
        self.epoch = 0

        self.temp_num = self.dataset.get_temp_num()
        self.batch_size_temp = int(self.batch_size *
                                   (self.temp_num / len(self.dataset)))
        self.batch_size_indep = self.batch_size - self.batch_size_temp
        self.total_batch = self.temp_num // (self.batch_size_temp * self.nranks)
        self.num_samples = self.total_batch * self.batch_size
        self.total_size = self.num_samples * self.nranks
        # print(
        #     len(self.dataset), self.temp_num, self.batch_size,
        #     self.batch_size_temp, self.batch_size_indep, self.total_size,
        #     self.num_samples, self.num_samples / self.batch_size)

    def __iter__(self):
        num_samples = len(self.dataset)
        indicestemp = np.arange(self.temp_num).tolist()
        indiceindep = np.arange(self.temp_num, num_samples).tolist()
        if len(indicestemp) % (self.batch_size_temp * self.nranks) != 0:
            indicestemp = indicestemp[:-(len(indicestemp) % (
                self.batch_size_temp * self.nranks))]
        subindicestemp = indicestemp[0:len(indicestemp):self.batch_size_temp]
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(subindicestemp)
            np.random.RandomState(self.epoch).shuffle(indiceindep)
            self.epoch += 1
        if self.total_batch * self.batch_size_indep * self.nranks > len(
                indiceindep):
            indiceindep += indiceindep[:self.total_batch * self.batch_size_indep
                                       * self.nranks - len(indiceindep)]
        tempindices = []

        for idx, index in enumerate(subindicestemp):
            tempindices.extend(indicestemp[index:index + self.batch_size_temp])
            tempindices.extend(indiceindep[idx * self.batch_size_indep:(idx + 1)
                                           * self.batch_size_indep])

        assert len(tempindices) == self.total_size

        # subsample
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            last_batch_size = self.total_size % (self.batch_size * self.nranks)
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(self.local_rank * self.batch_size,
                           len(indices) - last_batch_size,
                           self.batch_size * self.nranks):
                subsampled_indices.extend(indices[i:i + self.batch_size])

            indices = indices[len(indices) - last_batch_size:]
            subsampled_indices.extend(indices[
                self.local_rank * last_local_batch_size:(
                    self.local_rank + 1) * last_local_batch_size])
            return subsampled_indices

        if self.nranks > 1:
            tempindices = _get_indices_by_batch_size(tempindices)

        assert len(tempindices) == self.num_samples
        _sample_iter = iter(tempindices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices


class CustomTempDBSampler_alltemp(DistributedBatchSampler):
    """
    Custom DistributedBatchSampler for 3dpose temporal sample
    """

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        subindices = indices[0:len(indices):self.batch_size]
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(subindices)
            self.epoch += 1
        tempindices = []
        for index in subindices:
            tempindices.extend(indices[index:index + self.batch_size])
        assert len(tempindices) == self.total_size

        # subsample
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            last_batch_size = self.total_size % (self.batch_size * self.nranks)
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(self.local_rank * self.batch_size,
                           len(indices) - last_batch_size,
                           self.batch_size * self.nranks):
                subsampled_indices.extend(indices[i:i + self.batch_size])

            indices = indices[len(indices) - last_batch_size:]
            subsampled_indices.extend(indices[
                self.local_rank * last_local_batch_size:(
                    self.local_rank + 1) * last_local_batch_size])
            return subsampled_indices

        if self.nranks > 1:
            tempindices = _get_indices_by_batch_size(tempindices)

        assert len(tempindices) == self.num_samples
        _sample_iter = iter(tempindices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices


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
        self.human36m_num = 0
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
                    if 'human3.6m' in imagename:
                        self.human36m_num += 1
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

    def get_temp_num(self):
        """get temporal data number, like human3.6m"""
        return self.human36m_num

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
