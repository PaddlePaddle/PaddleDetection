简体中文 | [English](README.md)

# SNIPER: Efficient Multi-Scale Training

## 模型库
| 有无sniper   | GPU个数    | 每张GPU图片个数 | 骨架网络             | 网络类型       | 学习率策略 | Box AP |                           模型下载                          | 配置文件 |
| :---------------- | :-------------------: | :------------------: | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| w/o sniper   |    4    |    1    | ResNet-r50-FPN      | Faster Rcnn         |   1x    |  23.3  | [faster_rcnn_r50_fpn_1x_visdrone](https://bj.bcebos.com/v1/paddledet/models/faster_rcnn_r50_fpn_1x_visdrone.pdparams ) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/sniper/faster_rcnn_r50_fpn_1x_sniper_coco.yml) |
| w sniper |    4    |    1    | ResNet-r50-FPN      | Faster Rcnn         |   1x    |  29.7  | [faster_rcnn_r50_fpn_1x_sniper_visdrone](https://bj.bcebos.com/v1/paddledet/models/faster_rcnn_r50_fpn_1x_sniper_visdrone.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/sniper/faster_rcnn_r50_fpn_2x_sniper_coco.yml) |




## 使用说明
### 1. 训练
a.可选：统计数据集信息，获得数据缩放尺度、有效框范围、chip尺度和步长等参数，修改configs/datasets/sniper_coco_detection.yml中对应参数
```bash
python tools/sniper_params_stats.py FasterRCNN annotations/instances_train2017.json
```
b.可选：训练检测器，生成负样本
```bash
python -m paddle.distributed.launch --log_dir=./sniper/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/sniper/faster_rcnn_r50_fpn_2x_sniper_coco.yml --save_proposals --proposals_path=./proposals.json &>sniper.log 2>&1 &
```
c.训练模型
```bash
python -m paddle.distributed.launch --log_dir=./sniper/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/sniper/faster_rcnn_r50_fpn_2x_sniper_coco.yml --eval &>sniper.log 2>&1 &
```

### 2. 评估
使用单GPU通过如下命令一键式评估模型在COCO val2017数据集效果
```bash
# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/sniper/faster_rcnn_r50_fpn_2x_sniper_coco.yml -o weights=output/faster_rcnn_r50_fpn_2x_sniper_coco/model_final
```

###3.推理
使用单GPU通过如下命令一键式推理图像，通过`--infer_img`指定图像路径，或通过`--infer_dir`指定目录并推理目录下所有图像

```bash
# 推理单张图像
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/sniper/faster_rcnn_r50_fpn_2x_sniper_coco.yml -o weights=output/faster_rcnn_r50_fpn_2x_sniper_coco/model_final --infer_img=demo/P0861__1.0__1154___824.png

# 推理目录下所有图像
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/sniper/faster_rcnn_r50_fpn_2x_sniper_coco.yml -o weights=output/faster_rcnn_r50_fpn_2x_sniper_coco/model_final --infer_dir=demo
```

## Citations
@misc{1805.09300,
Author = {Bharat Singh and Mahyar Najibi and Larry S. Davis},
Title = {SNIPER: Efficient Multi-Scale Training},
Year = {2018},
Eprint = {arXiv:1805.09300},
}
