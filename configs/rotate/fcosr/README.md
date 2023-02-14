简体中文 | [English](README_en.md)

# FCOSR

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [使用说明](#使用说明)
- [预测部署](#预测部署)
- [引用](#引用)

## 简介

[FCOSR](https://arxiv.org/abs/2111.10780)是基于[FCOS](https://arxiv.org/abs/1904.01355)的单阶段Anchor-Free的旋转框检测算法。FCOSR主要聚焦于旋转框的标签匹配策略，提出了椭圆中心采样和模糊样本标签匹配的方法。在loss方面，FCOSR使用了[ProbIoU](https://arxiv.org/abs/2106.06072)避免边界不连续性问题。

## 模型库

| 模型 | Backbone | mAP | 学习率策略 | 角度表示 | 数据增广 | GPU数目 | 每GPU图片数目 | 模型下载 | 配置文件 |
|:---:|:--------:|:----:|:---------:|:-----:|:--------:|:-----:|:------------:|:-------:|:------:|
| FCOSR-M | ResNeXt-50 | 76.62 | 3x | oc | RR | 4 | 4 | [model](https://paddledet.bj.bcebos.com/models/fcosr_x50_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/fcosr/fcosr_x50_3x_dota.yml) |

**注意:**

- 如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- 模型库中的模型默认使用单尺度训练单尺度测试。如果数据增广一栏标明MS，意味着使用多尺度训练和多尺度测试。如果数据增广一栏标明RR，意味着使用RandomRotate数据增广进行训练。

## 使用说明

参考[数据准备](../README.md#数据准备)准备数据。

### 训练

GPU单卡训练
``` bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rotate/fcosr/fcosr_x50_3x_dota.yml
```

GPU多卡训练
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rotate/fcosr/fcosr_x50_3x_dota.yml
```

### 预测

执行以下命令预测单张图片，图片预测结果会默认保存在`output`文件夹下面
``` bash
python tools/infer.py -c configs/rotate/fcosr/fcosr_x50_3x_dota.yml -o weights=https://paddledet.bj.bcebos.com/models/fcosr_x50_3x_dota.pdparams --infer_img=demo/P0861__1.0__1154___824.png --draw_threshold=0.5
```

### DOTA数据集评估

参考[DOTA Task](https://captain-whu.github.io/DOTA/tasks.html), 评估DOTA数据集需要生成一个包含所有检测结果的zip文件，每一类的检测结果储存在一个txt文件中，txt文件中每行格式为：`image_name score x1 y1 x2 y2 x3 y3 x4 y4`。将生成的zip文件提交到[DOTA Evaluation](https://captain-whu.github.io/DOTA/evaluation.html)的Task1进行评估。你可以执行以下命令得到test数据集的预测结果：
``` bash
python tools/infer.py -c configs/rotate/fcosr/fcosr_x50_3x_dota.yml -o weights=https://paddledet.bj.bcebos.com/models/fcosr_x50_3x_dota.pdparams --infer_dir=/path/to/test/images --output_dir=output_fcosr --visualize=False --save_results=True
```
将预测结果处理成官网评估所需要的格式：
``` bash
python configs/rotate/tools/generate_result.py --pred_txt_dir=output_fcosr/ --output_dir=submit/ --data_type=dota10

zip -r submit.zip submit
```

## 预测部署

部署教程请参考[预测部署](../../../deploy/README.md)

## 引用

```
@article{li2021fcosr,
  title={Fcosr: A simple anchor-free rotated detector for aerial object detection},
  author={Li, Zhonghua and Hou, Biao and Wu, Zitong and Jiao, Licheng and Ren, Bo and Yang, Chen},
  journal={arXiv preprint arXiv:2111.10780},
  year={2021}
}

@inproceedings{tian2019fcos,
  title={Fcos: Fully convolutional one-stage object detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={9627--9636},
  year={2019}
}

@article{llerena2021gaussian,
  title={Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection},
  author={Llerena, Jeffri M and Zeni, Luis Felipe and Kristen, Lucas N and Jung, Claudio},
  journal={arXiv preprint arXiv:2106.06072},
  year={2021}
}
```
