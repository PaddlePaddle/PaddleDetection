# 入门使用

## 安装

关于安装配置运行环境，请参考[安装指南](INSTALL_cn.md)


## 准备数据
- 首先按照[准备数据文档](PrepareDataSet.md) 准备数据。  
- 然后设置`configs/datasets`中相应的coco或voc等数据配置文件中的数据路径。


## 训练/评估/预测

PaddleDetection在[tools](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/tools)中提供了`训练`/`评估`/`预测`/`导出模型`等功能，支持通过传入不同可选参数实现特定功能

### 参数列表

以下列表可以通过`--help`查看

|         FLAG             |     支持脚本    |        用途        |      默认值       |         备注         |
| :----------------------: | :------------: | :---------------: | :--------------: | :-----------------: |
|          -c              |      ALL       |  指定配置文件  |  None  |  **必选**，例如-c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml |
|        --eval            |     train      |  是否边训练边测试  |  False  |  可选，如需指定，直接`--eval`即可 |
|          -o              |      ALL       |  设置或更改配置文件里的参数内容  |  None  |  可选，例如：`-o use_gpu=False`  |
|       --slim_config             |     ALL      |  模型压缩策略配置文件  |  None  |  可选，例如`--slim_config configs/slim/prune/yolov3_prune_l1_norm.yml`  |
|       --output_dir       |      infer/export_model     |  预测后结果或导出模型保存路径  |  `./output`  |  可选，例如`--output_dir=output`  |
|    --draw_threshold      |      infer     |  可视化时分数阈值  |  0.5  |  可选，`--draw_threshold=0.7`  |
|      --infer_dir         |       infer     |  用于预测的图片文件夹路径  |  None  |  可选  |
|      --infer_img         |       infer     |  用于预测的图片路径  |  None  |  可选，`--infer_img`和`--infer_dir`必须至少设置一个  |


### 训练

- 单卡训练
```bash
# 通过CUDA_VISIBLE_DEVICES指定GPU卡号
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml
```
- 多卡训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml
```

- 边训练边评估
```bash
python tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml --eval
```

### 评估
```bash
# 目前只支持单卡评估
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml
```

### 预测
```bash
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml --infer_img={IMAGE_PATH}
```

## 预测部署

（1）导出模型

```bash
python tools/export_model.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
        -o weights=output/faster_rcnn_r50_fpn_1x_coco/model_final \
        --output_dir=output_inference
```

（2）预测部署

参考[预测部署文档](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/deploy)。


## 模型压缩

参考[模型压缩文档](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/slim)。
