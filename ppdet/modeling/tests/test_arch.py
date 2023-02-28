from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import os
import unittest

import ppdet.core.workspace
from ppdet.core.workspace import *
from ppdet.engine import Trainer
from ppdet.utils.checkpoint import match_state_dict

# model_list
model_list = get_registered_modules()
# print(model_list)

# backbone = create('ResNet')
# kwargs = {'input_shape': backbone.out_shape}
# ov_transformer = create('DeformableTransformer', **kwargs)
# # print(ov_transformer)
# # channel = 256
# # srcs = []
# # for i in range(4):
# #     srcs.append(paddle.ones([1, channel, 60, 80]))
# #     channel *= 2
# out = ov_transformer(srcs)
#
# print(out)
dataset_path = '/home/a401-2/PycharmProjects/PaddleDetection/configs/datasets/zeroshot_coco_detection.yml'
reader_path = '/home/a401-2/PycharmProjects/PaddleDetection/configs/ov_detr/_base_/ov_deformable_detr_reader.yml'
model_path = '/home/a401-2/PycharmProjects/PaddleDetection/configs/ov_detr/ov_deformable_detr_r50_1x_coco.yml'
# model_path = '/home/a401-2/PycharmProjects/PaddleDetection/configs/deformable_detr/deformable_detr_r50_1x_coco.yml'
# checkpoint = "/home/a401-2/PycharmProjects/PaddleDetection/pretrained_models/coco_model.pdparams"
checkpoint = "/home/a401-2/PycharmProjects/PaddleDetection/pretrained_models/coco_model_new.pdparams"
# global_config = ppdet.core.workspace.AttrDict()
ppdet.core.workspace.load_config(model_path)
ppdet.core.workspace.load_config(dataset_path)
ppdet.core.workspace.load_config(reader_path)
print(global_config)

# dataset
coco_dataset = create('TrainDataset')
coco_dataset.parse_dataset()
# print(coco_dataset[0])
dataloader = create('TrainReader')
# coco_dataset = create('EvalDataset')
# coco_dataset.parse_dataset()
# # print(coco_dataset[0])
# dataloader = create('EvalReader')
dataloader = dataloader(coco_dataset, 0)

# model
model = create('OVDETR')
# print(model)

checkpoint_model = paddle.load(checkpoint)
# print('backbone', checkpoint_model.keys())
# print(checkpoint_model['encoder.layers.0.self_attn.value_proj.weight'])
# print(checkpoint_model.keys())
model.set_state_dict(checkpoint_model)
# print(model.backbone.state_dict().keys())


res = match_state_dict(model.state_dict(), checkpoint_model)
# print(res)

i = 0
for data in dataloader:
    if i > 1:
        exit()
    # print('data', data)
    # model.eval()
    out = model(data)

    # backbone_out = backbone(data)
    # # print(backbone_out)
    # pad_mask = data['pad_mask']
    # out = ov_transformer(backbone_out, pad_mask, data)
    print(out)
    out['loss'].backward()
    # for n, p in model.named_parameters():
    #     print(n)
    #     if p.grad is None:
    #         print(p.grad)
    #     else:
    #         print(p.grad.abs().sum())
    exit()
    i += 1
