## 大规模实用目标检测模型

### 简介

* 与图像分类任务不同，目标检测任务中，不仅需要标注图像中物体所属类别，还要标注其边框位置，因此标注成本相对更高。目前已开源的目标检测数据集中，应用比较广泛的有Open Images V5、Objects365和COCO数据集，这三个数据集的基本信息如下。


|   Dataset          | Classes | Images    | Bounding boxes |
|--------------------|---------|-----------|----------------|
| COCO               | 80      | 123,287   | 886,284        |
| Objects365 | 365     | 600,000   | 10,000,000     |
| Open Images V5              | 500     | 1,743,042 | 14,610,229     |


上述数据集中包含的类别均不多(相比于ImageNet1k分类数据集的1000个类别)。为了提供更加实用的服务器端目标检测模型，方便用户在不需要任何微调的情况下就可以直接使用，PaddleDetection结合[服务器端实用目标检测方案](./SERVER_SIDE.md)，融合Open Images V5和Objects365训练集数据(二者包含许多重复类别)，生成了包含676个类别的新数据集，类别映射关系可以在这里查看: [676个类别的标签文件](../../dataset/voc/generic_det_label_list_zh.txt)。并训练了服务器端实用目标检测模型，适用于绝大部分应用场景，方便用户直接部署使用，用户也可以根据提供的预训练模型，在自己的数据集上进行模型微调，加快收敛并获得更高的精度指标。


### 模型库


| 骨架网络       | 网络类型     |      下载       | 配置文件 |
| :---------------| :---------------| :---------------| :---------------
| ResNet50-vd-FPN-Dcnv2         | Cascade Faster     |  [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r50_vd_fpn_gen_server_side.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/rcnn_enhance/generic/cascade_rcnn_dcn_r50_vd_fpn_gen_server_side.yml) |
| ResNet101-vd-FPN-Dcnv2         | Cascade Faster     |  [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r101_vd_fpn_gen_server_side.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/rcnn_enhance/generic/cascade_rcnn_dcn_r101_vd_fpn_gen_server_side.yml) |
| CBResNet101-vd-FPN         | Cascade Faster     |  [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_cbr101_vd_fpn_server_side.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/rcnn_enhance/generic/cascade_rcnn_cbr101_vd_fpn_server_side.yml) |
