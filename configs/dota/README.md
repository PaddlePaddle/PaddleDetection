# S2ANet模型

## 内容
- [简介](#简介)
- [准备数据](#准备数据)
- [开始训练](#开始训练)
- [模型库](#模型库)
- [预测部署](#预测部署)

## 简介

[S2ANet](https://arxiv.org/pdf/2008.09397.pdf)是用于检测旋转框的模型，要求使用PaddlePaddle 2.1.1(可使用pip安装) 或适当的[develop版本](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#whl-release)。


## 准备数据

### DOTA数据
[DOTA Dataset]是航空影像中物体检测的数据集，包含2806张图像，每张图像4000*4000分辨率。

|  数据版本  |  类别数  |   图像数   |  图像尺寸  |    实例数    |     标注方式     |
|:--------:|:-------:|:---------:|:---------:| :---------:| :------------: |
|   v1.0   |   15    |   2806    | 800~4000  |   118282    |   OBB + HBB     |
|   v1.5   |   16    |   2806    | 800~4000  |   400000    |   OBB + HBB     |

注：OBB标注方式是指标注任意四边形；顶点按顺时针顺序排列。HBB标注方式是指标注示例的外接矩形。

DOTA数据集中总共有2806张图像，其中1411张图像作为训练集，458张图像作为评估集，剩余937张图像作为测试集。

如果需要切割图像数据，请参考[DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) 。

设置`crop_size=1024, stride=824, gap=200`参数切割数据后，训练集15749张图像，评估集5297张图像，测试集10833张图像。

### 自定义数据

数据标注有两种方式：

- 第一种是标注旋转矩形，可以通过旋转矩形标注工具[roLabelImg](https://github.com/cgvict/roLabelImg) 来标注旋转矩形框。

- 第二种是标注四边形，通过脚本转成外接旋转矩形，这样得到的标注可能跟真实的物体框有一定误差。

然后将标注结果转换成coco标注格式，其中每个`bbox`的格式为 `[x_center, y_center, width, height, angle]`，这里角度以弧度表示。

参考[脊椎间盘数据集](https://aistudio.baidu.com/aistudio/datasetdetail/85885) ，我们将数据集划分为训练集(230)、测试集(57)，数据地址为：[spine_coco](https://paddledet.bj.bcebos.com/data/spine_coco.tar) 。该数据集图像数量比较少，使用这个数据集可以快速训练S2ANet模型。


## 开始训练

### 1. 安装旋转框IOU计算OP

旋转框IOU计算OP[ext_op](../../ppdet/ext_op)是参考Paddle[自定义外部算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_custom_op.html) 的方式开发。

若使用旋转框IOU计算OP，需要环境满足：
- PaddlePaddle >= 2.1.1
- GCC == 8.2

推荐使用docker镜像[paddle:2.1.1-gpu-cuda10.1-cudnn7](registry.baidubce.com/paddlepaddle/paddle:2.1.1-gpu-cuda10.1-cudnn7)。

执行如下命令下载镜像并启动容器：
```
sudo nvidia-docker run -it --name paddle_s2anet -v $PWD:/paddle --network=host registry.baidubce.com/paddlepaddle/paddle:2.1.1-gpu-cuda10.1-cudnn7 /bin/bash
```

镜像中paddle已安装好，进入python3.7，执行如下代码检查paddle安装是否正常：
```
import paddle
print(paddle.__version__)
paddle.utils.run_check()
```

进入到`ppdet/ext_op`文件夹，安装：
```
python3.7 setup.py install
```

Windows环境请按照如下步骤安装：

（1）准备Visual Studio (版本需要>=Visual Studio 2015 update3)，这里以VS2017为例；

（2）点击开始-->Visual Studio 2017-->适用于 VS 2017 的x64本机工具命令提示；

（3）设置环境变量：`set DISTUTILS_USE_SDK=1`

（4）进入`PaddleDetection/ppdet/ext_op`目录，通过`python3.7 setup.py install`命令进行安装。

安装完成后，测试自定义op是否可以正常编译以及计算结果：
```
cd PaddleDetecetion/ppdet/ext_op
python3.7 test.py
```

### 2. 训练
**注意：**
配置文件中学习率是按照8卡GPU训练设置的，如果使用单卡GPU训练，请将学习率设置为原来的1/8。

GPU单卡训练
```bash
export CUDA_VISIBLE_DEVICES=0
python3.7 tools/train.py -c configs/dota/s2anet_1x_spine.yml
```

GPU多卡训练
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/dota/s2anet_1x_spine.yml
```

可以通过`--eval`开启边训练边测试。

### 3. 评估
```bash
python3.7 tools/eval.py -c configs/dota/s2anet_1x_spine.yml -o weights=output/s2anet_1x_spine/model_final.pdparams

# 使用提供训练好的模型评估
python3.7 tools/eval.py -c configs/dota/s2anet_1x_spine.yml -o weights=https://paddledet.bj.bcebos.com/models/s2anet_1x_spine.pdparams
```
** 注意：**
(1) dota数据集中是train和val数据作为训练集一起训练的，对dota数据集进行评估时需要自定义设置评估数据集配置。

(2) 骨骼数据集是由分割数据转换而来，由于椎间盘不同类别对于检测任务而言区别很小，且s2anet算法最后得出的分数较低，评估时默认阈值为0.5，mAP较低是正常的。建议通过可视化查看检测结果。

### 4. 预测
执行如下命令，会将图像预测结果保存到`output`文件夹下。
```bash
python3.7 tools/infer.py -c configs/dota/s2anet_1x_spine.yml -o weights=output/s2anet_1x_spine/model_final.pdparams --infer_img=demo/39006.jpg --draw_threshold=0.3
```
使用提供训练好的模型预测：
```bash
python3.7 tools/infer.py -c configs/dota/s2anet_1x_spine.yml -o weights=https://paddledet.bj.bcebos.com/models/s2anet_1x_spine.pdparams --infer_img=demo/39006.jpg --draw_threshold=0.3
```

### 5. DOTA数据评估
执行如下命令，会在`output`文件夹下将每个图像预测结果保存到同文件夹名的txt文本中。
```
python3.7 tools/infer.py -c configs/dota/s2anet_alignconv_2x_dota.yml -o weights=./weights/s2anet_alignconv_2x_dota.pdparams  --infer_dir=dota_test_images --draw_threshold=0.05 --save_txt=True --output_dir=output
```

请参考[DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) 生成评估文件，评估文件格式请参考[DOTA Test](http://captain.whu.edu.cn/DOTAweb/tasks.html) ，生成zip文件，每个类一个txt文件，txt文件中每行格式为：`image_id score x1 y1 x2 y2 x3 y3 x4 y4`，提交服务器进行评估。您也可以参考`dataset/dota_coco/dota_generate_test_result.py`脚本生成评估文件，提交到服务器。

## 模型库

### S2ANet模型

|     模型     |  Conv类型  |   mAP    |   模型下载   |   配置文件   |
|:-----------:|:----------:|:--------:| :----------:| :---------: |
|   S2ANet    |   Conv     |   71.42  |  [model](https://paddledet.bj.bcebos.com/models/s2anet_conv_2x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/dota/s2anet_conv_2x_dota.yml)                   |
|   S2ANet    |  AlignConv |   74.0   |  [model](https://paddledet.bj.bcebos.com/models/s2anet_alignconv_2x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/dota/s2anet_alignconv_2x_dota.yml)                   |

**注意：** 这里使用`multiclass_nms`，与原作者使用nms略有不同。


## 预测部署

Paddle中`multiclass_nms`算子的输入支持四边形输入，因此部署时可以不需要依赖旋转框IOU计算算子。

部署教程请参考[预测部署](../../deploy/README.md)


## Citations
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
