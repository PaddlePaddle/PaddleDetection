简体中文 | [English](README_en.md)

# S2ANet

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [使用说明](#使用说明)
- [预测部署](#预测部署)
- [引用](#引用)

## 简介

[S2ANet](https://arxiv.org/pdf/2008.09397.pdf)是用于检测旋转框的模型.

## 模型库

| 模型 | Conv类型 | mAP | 学习率策略 | 角度表示 | 数据增广 | GPU数目 | 每GPU图片数目 | 模型下载 | 配置文件 |
|:---:|:------:|:----:|:---------:|:-----:|:--------:|:-----:|:------------:|:-------:|:------:|
| S2ANet | Conv | 71.45 | 2x | le135 | - | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/s2anet_conv_2x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/s2anet/s2anet_conv_2x_dota.yml) |
| S2ANet | AlignConv | 73.84 | 2x | le135 | - | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/s2anet_alignconv_2x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/s2anet/s2anet_alignconv_2x_dota.yml) |

**注意：**

- 如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- 模型库中的模型默认使用单尺度训练单尺度测试。如果数据增广一栏标明MS，意味着使用多尺度训练和多尺度测试。如果数据增广一栏标明RR，意味着使用RandomRotate数据增广进行训练。
- 这里使用`multiclass_nms`，与原作者使用nms略有不同。


## 使用说明

参考[数据准备](../README.md#数据准备)准备数据。

### 1. 训练

GPU单卡训练
```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rotate/s2anet/s2anet_1x_spine.yml
```

GPU多卡训练
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rotate/s2anet/s2anet_1x_spine.yml
```

可以通过`--eval`开启边训练边测试。

### 2. 评估
```bash
python tools/eval.py -c configs/rotate/s2anet/s2anet_1x_spine.yml -o weights=output/s2anet_1x_spine/model_final.pdparams

# 使用提供训练好的模型评估
python tools/eval.py -c configs/rotate/s2anet/s2anet_1x_spine.yml -o weights=https://paddledet.bj.bcebos.com/models/s2anet_1x_spine.pdparams
```

### 3. 预测
执行如下命令，会将图像预测结果保存到`output`文件夹下。
```bash
python tools/infer.py -c configs/rotate/s2anet/s2anet_1x_spine.yml -o weights=output/s2anet_1x_spine/model_final.pdparams --infer_img=demo/39006.jpg --draw_threshold=0.3
```
使用提供训练好的模型预测：
```bash
python tools/infer.py -c configs/rotate/s2anet/s2anet_1x_spine.yml -o weights=https://paddledet.bj.bcebos.com/models/s2anet_1x_spine.pdparams --infer_img=demo/39006.jpg --draw_threshold=0.3
```

### 4. DOTA数据评估
执行如下命令，会在`output`文件夹下将每个图像预测结果保存到同文件夹名的txt文本中。
```
python tools/infer.py -c configs/rotate/s2anet/s2anet_alignconv_2x_dota.yml -o weights=https://paddledet.bj.bcebos.com/models/s2anet_alignconv_2x_dota.pdparams --infer_dir=/path/to/test/images --output_dir=output --visualize=False --save_results=True
```
参考[DOTA Task](https://captain-whu.github.io/DOTA/tasks.html), 评估DOTA数据集需要生成一个包含所有检测结果的zip文件，每一类的检测结果储存在一个txt文件中，txt文件中每行格式为：`image_name score x1 y1 x2 y2 x3 y3 x4 y4`。将生成的zip文件提交到[DOTA Evaluation](https://captain-whu.github.io/DOTA/evaluation.html)的Task1进行评估。你可以执行以下命令生成评估文件
```
python configs/rotate/tools/generate_result.py --pred_txt_dir=output/ --output_dir=submit/ --data_type=dota10

zip -r submit.zip submit
```

## 预测部署

Paddle中`multiclass_nms`算子的输入支持四边形输入，因此部署时可以不需要依赖旋转框IOU计算算子。

部署教程请参考[预测部署](../../../deploy/README.md)


## 引用
```
@article{han2021align,  
  author={J. {Han} and J. {Ding} and J. {Li} and G. -S. {Xia}},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},  
  title={Align Deep Features for Oriented Object Detection},  
  year={2021},
  pages={1-11},  
  doi={10.1109/TGRS.2021.3062048}}

@inproceedings{xia2018dota,
  title={DOTA: A large-scale dataset for object detection in aerial images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3974--3983},
  year={2018}
}
```
