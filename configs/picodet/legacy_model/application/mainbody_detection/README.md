# 更多应用


## 1. 主体检测任务

主体检测技术是目前应用非常广泛的一种检测技术，它指的是检测出图片中一个或者多个主体的坐标位置，然后将图像中的对应区域裁剪下来，进行识别，从而完成整个识别过程。主体检测是识别任务的前序步骤，可以有效提升识别精度。

主体检测是图像识别的前序步骤，被用于PaddleClas的PP-ShiTu图像识别系统中。PP-ShiTu中使用的主体检测模型基于PP-PicoDet。更多关于PP-ShiTu的介绍与使用可以参考：[PP-ShiTu](https://github.com/PaddlePaddle/PaddleClas)。


### 1.1 数据集

PP-ShiTu图像识别任务中，训练主体检测模型时主要用到了以下几个数据集。

| 数据集       | 数据量   | 主体检测任务中使用的数据量   | 场景  | 数据集地址 |
| :------------:  | :-------------: | :-------: | :-------: | :--------: |
| Objects365 | 1700K | 173k | 通用场景 | [地址](https://www.objects365.org/overview.html) |
| COCO2017 | 118K | 118k  | 通用场景 | [地址](https://cocodataset.org/) |
| iCartoonFace | 48k | 48k | 动漫人脸检测 | [地址](https://github.com/luxiangju-PersonAI/iCartoonFace) |
| LogoDet-3k | 155k | 155k | Logo检测 | [地址](https://github.com/Wangjing1551/LogoDet-3K-Dataset) |
| RPC | 54k | 54k  | 商品检测 | [地址](https://rpc-dataset.github.io/) |

在实际训练的过程中，将所有数据集混合在一起。由于是主体检测，这里将所有标注出的检测框对应的类别都修改为 `前景` 的类别，最终融合的数据集中只包含 1 个类别，即前景，数据集定义配置可以参考[mainbody_detection.yml](./mainbody_detection.yml)。


### 1.2 模型库

| 模型     | 图像输入尺寸 | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 |  下载地址  | config |
| :-------- | :--------: | :---------------------: | :----------------: | :----------------: | :---------------: |
| PicoDet-LCNet_x2_5 |  640*640   |          41.5   |    62.0     | [trained model](https://paddledet.bj.bcebos.com/models/picodet_lcnet_x2_5_640_mainbody.pdparams) &#124; [inference model](https://paddledet.bj.bcebos.com/models/picodet_lcnet_x2_5_640_mainbody_infer.tar) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_lcnet_x2_5_640_mainbody.log) | [config](./picodet_lcnet_x2_5_640_mainbody.yml) |
