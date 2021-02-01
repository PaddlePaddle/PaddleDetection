# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle.utils import try_import

from ppdet.core.workspace import register, serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


def print_prune_params(model):
    model_dict = model.state_dict()
    for key in model_dict.keys():
        weight_name = model_dict[key].name
        logger.info('Parameter name: {}, shape: {}'.format(
            weight_name, model_dict[key].shape))


@register
@serializable
class Pruner(object):
    def __init__(self,
                 criterion,
                 pruned_params,
                 pruned_ratios,
                 print_params=False):
        super(Pruner, self).__init__()
        assert criterion in ['l1_norm', 'fpgm'], \
            "unsupported prune criterion: {}".format(criterion)
        self.criterion = criterion
        self.pruned_params = pruned_params
        self.pruned_ratios = pruned_ratios
        self.print_params = print_params

    def __call__(self, model):
        # FIXME: adapt to network graph when Training and inference are
        # inconsistent, now only supports prune inference network graph.
        model.eval()
        paddleslim = try_import('paddleslim')
        from paddleslim.analysis import dygraph_flops as flops
        input_spec = [{
            "image": paddle.ones(
                shape=[1, 3, 640, 640], dtype='float32'),
            "im_shape": paddle.full(
                [1, 2], 640, dtype='float32'),
            "scale_factor": paddle.ones(
                shape=[1, 2], dtype='float32')
        }]
        if self.print_params:
            print_prune_params(model)

        ori_flops = flops(model, input_spec) / 1000
        logger.info("FLOPs before pruning: {}GFLOPs".format(ori_flops))
        if self.criterion == 'fpgm':
            pruner = paddleslim.dygraph.FPGMFilterPruner(model, input_spec)
        elif self.criterion == 'l1_norm':
            pruner = paddleslim.dygraph.L1NormFilterPruner(model, input_spec)

        logger.info("pruned params: {}".format(self.pruned_params))
        pruned_ratios = [float(n) for n in self.pruned_ratios]
        ratios = {}
        for i, param in enumerate(self.pruned_params):
            ratios[param] = pruned_ratios[i]
        pruner.prune_vars(ratios, [0])
        pruned_flops = flops(model, input_spec) / 1000
        logger.info("FLOPs after pruning: {}GFLOPs; pruned ratio: {}".format(
            pruned_flops, (ori_flops - pruned_flops) / ori_flops))

        return model
