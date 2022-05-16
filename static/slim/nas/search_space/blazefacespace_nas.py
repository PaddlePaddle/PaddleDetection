# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from __future__ import division
from __future__ import print_function

import numpy as np
from paddleslim.nas.search_space.search_space_base import SearchSpaceBase
from paddleslim.nas.search_space.search_space_registry import SEARCHSPACE
from ppdet.modeling.backbones.blazenet import BlazeNet
from ppdet.modeling.architectures.blazeface import BlazeFace


@SEARCHSPACE.register
class BlazeFaceNasSpace(SearchSpaceBase):
    def __init__(self, input_size, output_size, block_num, block_mask):
        super(BlazeFaceNasSpace, self).__init__(input_size, output_size,
                                                block_num, block_mask)
        self.blaze_filter_num1 = np.array([4, 8, 12, 16, 24, 32])
        self.blaze_filter_num2 = np.array([8, 12, 16, 24, 32, 40, 48, 64])
        self.mid_filter_num = np.array([8, 12, 16, 20, 24, 32])
        self.double_filter_num = np.array(
            [8, 12, 16, 24, 32, 40, 48, 64, 72, 80, 88, 96])
        self.use_5x5kernel = np.array(
            [0]
        )  ### if constraint is latency, use 3x3 kernel, otherwise self.use_5x5kernel = np.array([0, 1])

    def init_tokens(self):
        return [2, 1, 3, 8, 2, 1, 2, 1, 1]

    def range_table(self):
        return [
            len(self.blaze_filter_num1), len(self.blaze_filter_num2),
            len(self.double_filter_num), len(self.double_filter_num),
            len(self.mid_filter_num), len(self.mid_filter_num),
            len(self.mid_filter_num), len(self.mid_filter_num),
            len(self.use_5x5kernel)
        ]

    def get_nas_cnf(self, tokens=None):
        if tokens is None:
            tokens = self.init_tokens()

        blaze_filters = [[
            self.blaze_filter_num1[tokens[0]], self.blaze_filter_num1[tokens[0]]
        ], [
            self.blaze_filter_num1[tokens[0]],
            self.blaze_filter_num2[tokens[1]], 2
        ], [
            self.blaze_filter_num2[tokens[1]], self.blaze_filter_num2[tokens[1]]
        ]]

        double_blaze_filters = [[
            self.blaze_filter_num2[tokens[1]], self.mid_filter_num[tokens[4]],
            self.double_filter_num[tokens[2]], 2
        ], [
            self.double_filter_num[tokens[2]], self.mid_filter_num[tokens[5]],
            self.double_filter_num[tokens[2]]
        ], [
            self.double_filter_num[tokens[2]], self.mid_filter_num[tokens[6]],
            self.double_filter_num[tokens[3]], 2
        ], [
            self.double_filter_num[tokens[3]], self.mid_filter_num[tokens[7]],
            self.double_filter_num[tokens[3]]
        ]]

        ### if constraint is latency, use 3x3 kernel, otherwise is_5x5kernel = True if self.use_5x5kernel[tokens[8]] else False
        is_5x5kernel = False  ###True if self.use_5x5kernel[tokens[8]] else False
        return blaze_filters, double_blaze_filters, is_5x5kernel

    def token2arch(self, tokens=None):

        blaze_filters, double_blaze_filters, is_5x5kernel = self.get_nas_cnf(
            tokens)
        self.print_nas_structure(tokens)

        def net_arch(input, mode, cfg):
            self.output_decoder = cfg.BlazeFace['output_decoder']
            self.min_sizes = cfg.BlazeFace['min_sizes']
            self.use_density_prior_box = cfg.BlazeFace['use_density_prior_box']

            my_backbone = BlazeNet(
                blaze_filters=blaze_filters,
                double_blaze_filters=double_blaze_filters,
                use_5x5kernel=is_5x5kernel)
            my_blazeface = BlazeFace(
                my_backbone,
                output_decoder=self.output_decoder,
                min_sizes=self.min_sizes,
                use_density_prior_box=self.use_density_prior_box)
            return my_blazeface.build(input, mode=mode)

        return net_arch

    def print_nas_structure(self, tokens=None):
        blaze_filters, double_filters, is_5x5kernel = self.get_nas_cnf(tokens)
        print('---------->>> BlazeFace-NAS structure start: <<<------------')
        print('BlazeNet:')
        print('  blaze_filters: {}'.format(blaze_filters))
        print('  double_blaze_filters: {}'.format(double_filters))
        print('  use_5x5kernel: {}'.format(is_5x5kernel))
        print('  with_extra_blocks: true')
        print('  lite_edition: false')
        print('---------->>> BlazeFace-NAS structure end! <<<------------')
