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

import numpy as np
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import KaimingUniform, Uniform
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from ppdet.modeling.backbones.dla import ConvLayer


@register
class FairReIDHead(nn.Layer):
    def __init__(self, in_channels, ch_head=256, ch_emb=128, num_id=14455):
        super(FairReIDHead, self).__init__()
        self.reid = nn.Sequential(
            ConvLayer(
                in_channels,
                ch_head,
                kernel_size=3,
                padding=1,
                bias=True,
                name="id.0"),
            nn.ReLU(),
            ConvLayer(
                ch_head,
                ch_emb,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                name="id.2"))
        param_attr = paddle.ParamAttr(
            initializer=KaimingUniform(), name="classifier.weight")
        bound = 1 / math.sqrt(ch_emb)
        bias_attr = paddle.ParamAttr(
            initializer=Uniform(-bound, bound), name="classifier.bias")
        self.classifier = nn.Linear(
            ch_emb, num_id, weight_attr=param_attr, bias_attr=bias_attr)
        self.reid_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(num_id - 1)

    @classmethod
    def from_config(cls, cfg, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channels': input_shape.channels}

    def forward(self, feat, inputs):
        output = dict()
        reid_feat = self.reid(feat)
        #print('---------reid_feat', np.mean(reid_feat.numpy()))
        if self.training:
            loss = self.get_loss(reid_feat, inputs)
            output.update(loss)
        else:
            reid_feat = F.normalize(feat)
            output['embedding'] = reid_feat
        return output

    def get_loss(self, feat, inputs):
        index = inputs['index']
        mask = inputs['index_mask']
        target = inputs['reid']
        #reid_feat = np.load('/rrpn/FairMOT/src/reid_feat.npy')
        #feat = paddle.to_tensor(reid_feat)
        #print('---------origin reid', np.mean(target.numpy()))
        target = paddle.masked_select(target, mask > 0)
        target = paddle.unsqueeze(target, 1)

        feat = paddle.transpose(feat, perm=[0, 2, 3, 1])
        feat_n, feat_h, feat_w, feat_c = feat.shape
        feat = paddle.reshape(feat, shape=[feat_n, -1, feat_c])
        index = paddle.unsqueeze(index, 2)
        batch_inds = list()
        for i in range(feat_n):
            batch_ind = paddle.full(
                shape=[1, index.shape[1], 1], fill_value=i, dtype='int64')
            batch_inds.append(batch_ind)
        batch_inds = paddle.concat(batch_inds, axis=0)
        index = paddle.concat(x=[batch_inds, index], axis=2)
        feat = paddle.gather_nd(feat, index=index)

        #print('---------gather reid feat', np.mean(feat.numpy()))
        mask = paddle.unsqueeze(mask, axis=2)
        mask = paddle.expand_as(mask, feat)
        mask.stop_gradient = True
        feat = paddle.masked_select(feat, mask > 0)
        #print('---------mask reid feat', np.mean(feat.numpy()))
        feat = paddle.reshape(feat, shape=[-1, feat_c])
        feat = F.normalize(feat)
        #print('---------normalize reid feat', np.mean(feat.numpy()))
        feat = self.emb_scale * feat
        #feat = paddle.to_tensor(np.load('/rrpn/new/FairMOT/src/classifier_feat.npy'))
        #print('-----------emb_scale', self.emb_scale)
        #print('----------classifier feat', np.mean(feat.numpy()))
        #print('----------classifier weight', np.mean(self.classifier.weight.numpy()))
        #print('----------classifier bias', np.mean(self.classifier.bias.numpy()))
        #weight = np.load('/rrpn/FairMOT/src/classifier_weight.npy')
        #bias = np.load('/rrpn/FairMOT/src/classifier_bias.npy')
        #self.classifier.weight[:] = weight
        #self.classifier.bias[:] = bias
        logit = self.classifier(feat)
        #logit = feat
        #logit = paddle.to_tensor(np.load('/rrpn/new/FairMOT/src/classifier_logit.npy'))
        #target = paddle.to_tensor(np.expand_dims(np.load('/rrpn/new/FairMOT/src/classifier_target.npy'), axis=-1))

        #print('----------classifier logit', np.mean(logit.numpy()))
        #print('----------classifier target', np.mean(target.numpy()))
        target.stop_gradient = True
        #logit = paddle.to_tensor(np.load('/rrpn/FairMOT/src/id_output.npy'))
        #target = paddle.to_tensor(np.expand_dims(np.load('/rrpn/FairMOT/src/id_target.npy'), 1))
        #loss = F.softmax_with_cross_entropy(logit, target)
        #loss = loss.mean()
        #np.save('logit.npy', logit.numpy())
        #np.save('target.npy', target.numpy())
        loss = self.reid_loss(logit, target)
        return {'reid_loss': loss}
