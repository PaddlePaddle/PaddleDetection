# S2ANet模型

## 内容
- [简介](#简介)
- [DOTA数据集](#DOTA数据集)
- [模型库](#模型库)
- [训练说明](#训练说明)

## 简介

[S2ANet](https://arxiv.org/pdf/2008.09397.pdf)是用于检测旋转框的模型，要求使用PaddlePaddle 2.0.1(可使用pip安装) 或适当的[develop版本](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#whl-release)。


## DOTA数据集
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
|   S2ANet    |    8    |   Conv     |   71.42  |  [model](https://paddledet.bj.bcebos.com/models/s2anet_conv_1x.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/dota/s2anet_conv_1x_dota.yml)                   |

**注意：**这里使用`multiclass_nms`，与原作者使用nms略有不同，精度相比原始论文中高0.15 (71.27-->71.42)。

## 训练说明

### 1. 旋转框IOU计算OP

旋转框IOU计算OP[ext_op](../../ppdet/ext_op)是参考Paddle[自定义外部算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_custom_op.html) 的方式开发。

若使用旋转框IOU计算OP，需要环境满足：
- Paddle >= 2.0.1
- gcc == 8.2

推荐使用docker镜像[paddle:2.0.1-gpu-cuda10.1-cudnn7](registry.baidubce.com/paddlepaddle/paddle:2.0.1-gpu-cuda10.1-cudnn7)。

执行如下命令下载镜像并启动容器：
```
sudo nvidia-docker run -it --name paddle_s2anet -v $PWD:/paddle --network=host registry.baidubce.com/paddlepaddle/paddle:2.0.1-gpu-cuda10.1-cudnn7 /bin/bash
```

进入容器后，安装必要的python包：
```
python3.7 -m pip install Cython wheel tqdm opencv-python==4.2.0.32 scipy  PyYAML shapely pycocotools
```

镜像中paddle2.0.1已安装好，进入python3.7，执行如下代码检查paddle安装是否正常：
```
import paddle
print(paddle.__version__)
paddle.utils.run_check()
```

进入到`ext_op`文件夹，安装：
```
python3.7 setup.py install
```

安装完成后，测试自定义op是否可以正常编译以及计算结果：
```
cd PaddleDetecetion/ppdet/ext_op
python3.7 test.py
```

### 2. 数据格式
DOTA 数据集中实例是按照任意四边形标注，在进行训练模型前，需要参考[DOTA2COCO](https://github.com/CAPTAIN-WHU/DOTA_devkit/blob/master/DOTA2COCO.py) 转换成`[xc, yc, bow_w, bow_h, angle]`格式，并以coco数据格式存储。

## 评估

执行如下命令，会在`output_dir`文件夹下将每个图像预测结果保存到同文件夹名的txt文本中。
```
python3.7 tools/infer.py -c configs/dota/s2anet_1x_dota.yml -o weights=./weights/s2anet_1x_dota.pdparams  --infer_dir=dota_test_images --draw_threshold=0.05 --save_txt=True --output_dir=output
```


请参考[DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) 生成评估文件，评估文件格式请参考[DOTA Test](http://captain.whu.edu.cn/DOTAweb/tasks.html) ，生成zip文件，每个类一个txt文件，txt文件中每行格式为：`image_id score x1 y1 x2 y2 x3 y3 x4 y4`，提交服务器进行评估。

## 预测部署

Paddle中`multiclass_nms`算子的输入支持四边形输入，因此部署时可以不不需要依赖旋转框IOU计算算子。

```bash
# 预测
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/dota/s2anet_1x_dota.yml -o weights=model.pdparams --infer_img=demo/P0072__1.0__0___0.png --use_gpu=True
```


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
