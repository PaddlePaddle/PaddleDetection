# S2ANet模型

## 内容
- [简介](#简介)
- [模型库与基线](#模型库与基线)
- [使用说明](#使用说明)
- [未来工作](#未来工作)
- [附录](#附录)

## 简介

[S2ANet](https://arxiv.org/pdf/2008.09397.pdf)是用于检测旋转框的模型，要求使用PaddlePaddle 2.0.1(可使用pip安装) 或适当的[develop版本](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#whl-release)。


## DOTA Dataset
[DOTA Dataset]是航空影像中物体检测的数据集，包含2806张图像，每张图像4000*4000分辨率。

|  数据版本  |  类别数  |   图像数   |  图像尺寸  |    实例数    |     标注方式     |
|:--------:|:-------:|:---------:|:---------:| :---------:| :------------: |
|   v1.0   |   15    |   2806    | 800~4000  |   118282    |   OBB + HBB     |
|   v1.5   |   16    |   2806    | 800~4000  |   400000    |   OBB + HBB     |

注：OBB标注方式是指标注任意四边形；顶点按顺时针顺序排列。HBB标注方式是指标注示例的外接矩形。

DOTA数据集中总共有2806张图像，其中1411张图像作为训练集，458张图像作为评估集，剩余937张图像作为测试集。

如果需要切割图像数据，请参考[DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) 。

设置`crop_size=1024, stride=824, gap=200`参数切割数据后，训练集15749张图像，评估集5297张图像，测试集10833张图像。

## 模型库

### S2ANet模型

|     模型     | GPU个数  |  Conv类型  |   mAP    |   模型下载   |   配置文件   |
|:-----------:|:-------:|:----------:|:--------:| :----------:| :---------: |
|   S2ANet    |    8    |   Conv     |   71.27  |  [model](https://paddledet.bj.bcebos.com/models/.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/dota/s2anet_1x_dota.yml)                   |


## 训练说明

### 1. 旋转框IOU计算OP

旋转框IOU计算OP[ext_op](ppdet/ext_op)是参考Paddle[自定义外部算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_custom_op.html) 的方式开发。

若使用旋转框IOU计算OP，需要环境满足：
- Paddle >= 2.0.1
- gcc 8.2

推荐使用docker镜像[paddle:2.0.1-gpu-cuda10.1-cudnn7](registry.baidubce.com/paddlepaddle/paddle:2.0.1-gpu-cuda10.1-cudnn7)。

使用镜像启动容器后，paddle2.0.1已安装好，按照如下命令测试op：
```
cd PaddleDetecetion/ppdet/ext_op
python3.7 test.py
```

### 2. 数据格式
DOTA 数据集中实例是按照任意四边形标注，在进行训练模型前，需要参考[DOTA2COCO](https://github.com/CAPTAIN-WHU/DOTA_devkit/blob/master/DOTA2COCO.py) 转换成`[xc, yc, bow_w, bow_h, angle]`格式，并以coco数据格式存储。

## 预测部署

Paddle中`multiclass_nms`算子的输入支持四边形输入，因此部署时可以不不需要依赖旋转框IOU计算算子。

```bash
# 预测
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/dota/s2anet_1x_dota.yml -o weights=model.pdparams --infer_img=demo/P0072__1.0__0___0.png --use_gpu=True
```
