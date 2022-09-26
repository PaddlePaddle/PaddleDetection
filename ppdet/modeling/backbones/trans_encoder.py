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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import ReLU, Swish, GELU
import math

from ppdet.core.workspace import register
from ..shape_spec import ShapeSpec

__all__ = ['TransEncoder']


class BertEmbeddings(nn.Layer):
    def __init__(self, word_size, position_embeddings_size, word_type_size,
                 hidden_size, dropout_prob):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            word_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(position_embeddings_size,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(word_type_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size, epsilon=1e-8)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, token_type_ids=None, position_ids=None):
        seq_len = paddle.shape(x)[1]
        if position_ids is None:
            position_ids = paddle.arange(seq_len).unsqueeze(0).expand_as(x)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(paddle.shape(x))

        word_embs = self.word_embeddings(x)
        position_embs = self.position_embeddings(position_ids)
        token_type_embs = self.token_type_embeddings(token_type_ids)

        embs_cmb = word_embs + position_embs + token_type_embs
        embs_out = self.layernorm(embs_cmb)
        embs_out = self.dropout(embs_out)
        return embs_out


class BertSelfAttention(nn.Layer):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_probs_dropout_prob,
                 output_attentions=False):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden_size must be a multiple of the number of attention "
                "heads, but got {} % {} != 0" %
                (hidden_size, num_attention_heads))

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.output_attentions = output_attentions

    def forward(self, x, attention_mask, head_mask=None):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query_dim1, query_dim2 = paddle.shape(query)[:-1]
        new_shape = [
            query_dim1, query_dim2, self.num_attention_heads,
            self.attention_head_size
        ]
        query = query.reshape(new_shape).transpose(perm=(0, 2, 1, 3))
        key = key.reshape(new_shape).transpose(perm=(0, 2, 3, 1))
        value = value.reshape(new_shape).transpose(perm=(0, 2, 1, 3))

        attention = paddle.matmul(query,
                                  key) / math.sqrt(self.attention_head_size)
        attention = attention + attention_mask
        attention_value = F.softmax(attention, axis=-1)
        attention_value = self.dropout(attention_value)

        if head_mask is not None:
            attention_value = attention_value * head_mask

        context = paddle.matmul(attention_value, value).transpose(perm=(0, 2, 1,
                                                                        3))
        ctx_dim1, ctx_dim2 = paddle.shape(context)[:-2]
        new_context_shape = [
            ctx_dim1,
            ctx_dim2,
            self.all_head_size,
        ]
        context = context.reshape(new_context_shape)

        if self.output_attentions:
            return (context, attention_value)
        else:
            return (context, )


class BertAttention(nn.Layer):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_probs_dropout_prob,
                 fc_dropout_prob,
                 output_attentions=False):
        super(BertAttention, self).__init__()
        self.bert_selfattention = BertSelfAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob,
            output_attentions)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size, epsilon=1e-8)
        self.dropout = nn.Dropout(fc_dropout_prob)

    def forward(self, x, attention_mask, head_mask=None):
        attention_feats = self.bert_selfattention(x, attention_mask, head_mask)
        features = self.fc(attention_feats[0])
        features = self.dropout(features)
        features = self.layernorm(features + x)
        if len(attention_feats) == 2:
            return (features, attention_feats[1])
        else:
            return (features, )


class BertFeedForward(nn.Layer):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 num_attention_heads,
                 attention_probs_dropout_prob,
                 fc_dropout_prob,
                 act_fn='ReLU',
                 output_attentions=False):
        super(BertFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act_fn = eval(act_fn)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size, epsilon=1e-8)
        self.dropout = nn.Dropout(fc_dropout_prob)

    def forward(self, x):
        features = self.fc1(x)
        features = self.act_fn(features)
        features = self.fc2(features)
        features = self.dropout(features)
        features = self.layernorm(features + x)
        return features


class BertLayer(nn.Layer):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 num_attention_heads,
                 attention_probs_dropout_prob,
                 fc_dropout_prob,
                 act_fn='ReLU',
                 output_attentions=False):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads,
                                       attention_probs_dropout_prob,
                                       output_attentions)
        self.feed_forward = BertFeedForward(
            hidden_size, intermediate_size, num_attention_heads,
            attention_probs_dropout_prob, fc_dropout_prob, act_fn,
            output_attentions)

    def forward(self, x, attention_mask, head_mask=None):
        attention_feats = self.attention(x, attention_mask, head_mask)
        features = self.feed_forward(attention_feats[0])
        if len(attention_feats) == 2:
            return (features, attention_feats[1])
        else:
            return (features, )


class BertEncoder(nn.Layer):
    def __init__(self,
                 num_hidden_layers,
                 hidden_size,
                 intermediate_size,
                 num_attention_heads,
                 attention_probs_dropout_prob,
                 fc_dropout_prob,
                 act_fn='ReLU',
                 output_attentions=False,
                 output_hidden_feats=False):
        super(BertEncoder, self).__init__()
        self.output_attentions = output_attentions
        self.output_hidden_feats = output_hidden_feats
        self.layers = nn.LayerList([
            BertLayer(hidden_size, intermediate_size, num_attention_heads,
                      attention_probs_dropout_prob, fc_dropout_prob, act_fn,
                      output_attentions) for _ in range(num_hidden_layers)
        ])

    def forward(self, x, attention_mask, head_mask=None):
        all_features = (x, )
        all_attentions = ()

        for i, layer in enumerate(self.layers):
            mask = head_mask[i] if head_mask is not None else None
            layer_out = layer(x, attention_mask, mask)

            if self.output_hidden_feats:
                all_features = all_features + (x, )
            x = layer_out[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_out[1], )

        outputs = (x, )
        if self.output_hidden_feats:
            outputs += (all_features, )
        if self.output_attentions:
            outputs += (all_attentions, )
        return outputs


class BertPooler(nn.Layer):
    def __init__(self, hidden_size):
        super(BertPooler, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.act = nn.Tanh()

    def forward(self, x):
        first_token = x[:, 0]
        pooled_output = self.fc(first_token)
        pooled_output = self.act(pooled_output)
        return pooled_output


class METROEncoder(nn.Layer):
    def __init__(self,
                 vocab_size,
                 num_hidden_layers,
                 features_dims,
                 position_embeddings_size,
                 hidden_size,
                 intermediate_size,
                 output_feature_dim,
                 num_attention_heads,
                 attention_probs_dropout_prob,
                 fc_dropout_prob,
                 act_fn='ReLU',
                 output_attentions=False,
                 output_hidden_feats=False,
                 use_img_layernorm=False):
        super(METROEncoder, self).__init__()
        self.img_dims = features_dims
        self.num_hidden_layers = num_hidden_layers
        self.use_img_layernorm = use_img_layernorm
        self.output_attentions = output_attentions
        self.embedding = BertEmbeddings(vocab_size, position_embeddings_size, 2,
                                        hidden_size, fc_dropout_prob)
        self.encoder = BertEncoder(
            num_hidden_layers, hidden_size, intermediate_size,
            num_attention_heads, attention_probs_dropout_prob, fc_dropout_prob,
            act_fn, output_attentions, output_hidden_feats)
        self.pooler = BertPooler(hidden_size)
        self.position_embeddings = nn.Embedding(position_embeddings_size,
                                                hidden_size)
        self.img_embedding = nn.Linear(
            features_dims, hidden_size, bias_attr=True)
        self.dropout = nn.Dropout(fc_dropout_prob)
        self.cls_head = nn.Linear(hidden_size, output_feature_dim)
        self.residual = nn.Linear(features_dims, output_feature_dim)

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.set_value(
                paddle.normal(
                    mean=0.0, std=0.02, shape=module.weight.shape))
        elif isinstance(module, nn.LayerNorm):
            module.bias.set_value(paddle.zeros(shape=module.bias.shape))
            module.weight.set_value(
                paddle.full(
                    shape=module.weight.shape, fill_value=1.0))
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.set_value(paddle.zeros(shape=module.bias.shape))

    def forward(self, x):
        batchsize, seq_len = paddle.shape(x)[:2]
        input_ids = paddle.zeros((batchsize, seq_len), dtype="int64")
        position_ids = paddle.arange(
            seq_len, dtype="int64").unsqueeze(0).expand_as(input_ids)

        attention_mask = paddle.ones_like(input_ids).unsqueeze(1).unsqueeze(2)
        head_mask = [None] * self.num_hidden_layers

        position_embs = self.position_embeddings(position_ids)
        attention_mask = (1.0 - attention_mask) * -10000.0

        img_features = self.img_embedding(x)

        # We empirically observe that adding an additional learnable position embedding leads to more stable training
        embeddings = position_embs + img_features
        if self.use_img_layernorm:
            embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        encoder_outputs = self.encoder(
            embeddings, attention_mask, head_mask=head_mask)

        pred_score = self.cls_head(encoder_outputs[0])
        res_img_feats = self.residual(x)
        pred_score = pred_score + res_img_feats

        if self.output_attentions and self.output_hidden_feats:
            return pred_score, encoder_outputs[1], encoder_outputs[-1]
        else:
            return pred_score


def gelu(x):
    """Implementation of the gelu activation function.
        https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + paddle.erf(x / math.sqrt(2.0)))


@register
class TransEncoder(nn.Layer):
    def __init__(self,
                 vocab_size=30522,
                 num_hidden_layers=4,
                 num_attention_heads=4,
                 position_embeddings_size=512,
                 intermediate_size=3072,
                 input_feat_dim=[2048, 512, 128],
                 hidden_feat_dim=[1024, 256, 128],
                 attention_probs_dropout_prob=0.1,
                 fc_dropout_prob=0.1,
                 act_fn='gelu',
                 output_attentions=False,
                 output_hidden_feats=False):
        super(TransEncoder, self).__init__()
        output_feat_dim = input_feat_dim[1:] + [3]
        trans_encoder = []
        for i in range(len(output_feat_dim)):
            features_dims = input_feat_dim[i]
            output_feature_dim = output_feat_dim[i]
            hidden_size = hidden_feat_dim[i]

            # init a transformer encoder and append it to a list
            assert hidden_size % num_attention_heads == 0
            model = METROEncoder(vocab_size, num_hidden_layers, features_dims,
                                 position_embeddings_size, hidden_size,
                                 intermediate_size, output_feature_dim,
                                 num_attention_heads,
                                 attention_probs_dropout_prob, fc_dropout_prob,
                                 act_fn, output_attentions, output_hidden_feats)
            trans_encoder.append(model)
        self.trans_encoder = paddle.nn.Sequential(*trans_encoder)

    def forward(self, x):
        out = self.trans_encoder(x)
        return out
