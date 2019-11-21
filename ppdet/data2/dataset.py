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

from ppdet.core.workspace import register, serializable


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
                 image_dir=None,
                 dataset_dir=None,
                 use_default_label=None,
                 data_dir=None):
        super(DataSet, self).__init__()
        self.annotation = annotation
        self.image_dir = image_dir
        self.dataset_dir = dataset_dir
        self.use_default_label = use_default_label

    def get_roidb(self):
        """get dataset dict in this dataset"""
        raise NotImplementedError('%s.get_roidb not available' %
                                  (self.__class__.__name__))

    def get_cname2cid(self):
        """get mapping between category to classid in this dataset"""
        raise NotImplementedError('%s.get_cname2cid not available' %
                                  (self.__class__.__name__))
