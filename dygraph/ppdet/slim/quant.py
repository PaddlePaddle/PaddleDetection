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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle.utils import try_import

from ppdet.core.workspace import register, serializable
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


@register
@serializable
class QAT(object):
    def __init__(self, quant_config, print_model):
        super(QAT, self).__init__()
        self.quant_config = quant_config
        self.print_model = print_model

    def __call__(self, model):
        paddleslim = try_import('paddleslim')
        self.quanter = paddleslim.dygraph.quant.QAT(config=self.quant_config)
        if self.print_model:
            logger.info("Model before quant:")
            logger.info(model)

        self.quanter.quantize(model)

        if self.print_model:
            logger.info("Quantized model:")
            logger.info(model)

        return model
