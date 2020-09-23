# 印刷电路板（PCB）瑕疵数据集模型
- 印刷电路板（PCB）瑕疵数据集：[数据下载链接](http://robotics.pkusz.edu.cn/resources/dataset/)，是一个公共的合成PCB数据集，由北京大学发布，其中包含1386张图像以及6种缺陷（缺失孔，鼠标咬伤，开路，短路，杂散，伪铜），用于检测，分类和配准任务。我们选取了其中适用与检测任务的693张图像，随机选择593张图像作为训练集，100张图像作为验证集。AIStudio数据集链接：[印刷电路板（PCB）瑕疵数据集](https://aistudio.baidu.com/aistudio/datasetdetail/52914)

## 模型
| 模型  | mAP(Iou=0.50:0.95)           | mAP(Iou=0.50)          | 配置文件 |
| :-------: | :-------: | :----: | :-----: |
| Faster-RCNN-R50_vd_FPN_3x | 52.7  | 97.7  | faster_rcnn_r50_vd_fpn_3x.yml |
| YOLOv3_darknet | 44.8  | 94.6 | yolov3_darknet.yml |
| FCOS_R50_FPN_multiscale_3x | 54.9 | 98.5 | fcos_r50_fpn_multiscale_3x.yml |
