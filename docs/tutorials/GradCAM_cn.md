# 目标检测热力图

## 1.简介

基于backbone特征图计算物体预测框的cam(类激活图)

## 2.使用方法
* 以PP-YOLOE为例，准备好数据之后，指定网络配置文件、模型权重地址和图片路径以及输出文件夹路径，使用脚本调用tools/cam_ppdet.py计算图片中物体预测框的grad_cam热力图。下面为运行脚本示例。
```shell
python tools/cam_ppdet.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml --infer_img demo/000000014439.jpg --cam_out cam_ppyoloe -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
```

* **参数**

|         FLAG             |                                                  用途                                                   |
| :----------------------: |:-----------------------------------------------------------------------------------------------------:|
|          -c              |                                                指定配置文件                                                 |
|          --infer_img              |                                               用于预测的图片路径                                               |
|          --cam_out              |                                                指定输出路径                                                 |
|          -o              | 设置或更改配置文件里的参数内容, 如 -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams |

* 运行效果

<center>
<img src="../images/grad_cam_ppyoloe_demo.jpg" width="500" >
</center>
<br><center>cam_ppyoloe/225.jpg</center></br>

## 3. 目前支持基于FasterRCNN和YOLOv3系列的网络。
* FasterRCNN网络热图可视化脚本
```bash
python tools/cam_ppdet.py -c configs/faster_rcnn/faster_rcnn_r50_vd_fpn_2x_coco.yml --infer_img demo/000000014439.jpg  --cam_out cam_faster_rcnn -o weights=https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_fpn_ssld_2x_coco.pdparams
```
* PPYOLOE网络热图可视化脚本
```bash
python tools/cam_ppdet.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml --infer_img demo/000000014439.jpg --cam_out cam_ppyoloe -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
```