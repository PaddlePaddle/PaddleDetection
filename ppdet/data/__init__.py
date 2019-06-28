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

# function:
#    module to prepare data for detection model training
#
# implementation notes:
# - Dateset
#    basic interface to accessing data samples in stream mode
#
# - xxxSource (RoiDbSource)
#    * subclass of 'Dataset'
#    * load data from local files and other source data
#
# - xxxOperator (DecodeImage)
#    * subclass of 'BaseOperator'
#    * each op can transform a sample, eg: decode/resize/crop image
#    * each op must obey basic rules defined in transform.operator.base
#
# - transformer
#    * subclass of 'Dataset'
#    * 'MappedDataset' accept a 'xxxSource' and a list of 'xxxOperator'
#       to build a transformed 'Dataset'

from __future__ import absolute_import

from .dataset import Dataset
from .reader import Reader
from .data_feed import create_reader

__all__ = ['Dataset', 'Reader', 'create_reader']
