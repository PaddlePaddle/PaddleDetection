# 入门使用

## 安装

`dygraph`分支需要安装每日版本的PaddlePaddle，PaddlePaddle中`c0a991c8740b413559bfc894aa5ae1d5ed3704b5`这个commit会影响精度，建议安装这个commit之前的版本。


## 准备数据
请按照[如何准备训练数据](PrepareDataSet.md) 准备训练数据。  
数据准备好之后，设置数据配置文件`configs/_base_/datasets/coco.yml`中的数据路径。


## 训练/评估/预测

PaddleDetection提供了训练/评估/预测，支持通过不同可选参数实现特定功能

#### 训练
```bash
# GPU训练 支持单卡，多卡训练，通过CUDA_VISIBLE_DEVICES指定卡号
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --selected_gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/faster_rcnn_r50_fpn_1x_coco.yml
```

#### 评估
```bash
# 使用单卡评估
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/faster_rcnn_r50_fpn_1x_coco.yml
```

#### 预测
```bash
# 预测
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/faster_rcnn_r50_fpn_1x_coco.yml --infer_img=demo/000000570688.jpg
```
