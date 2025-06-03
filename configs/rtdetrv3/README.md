## RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Positive Supervision

## 简介

RTDETRv3采用了模块化的设计架构，充分继承和发展了RT-DETR系列的优势，最显著的创新点是引入了一对多(One-to-Many, O2M)分支策略，这是本模型的核心设计。该策略允许每个ground truth对象匹配多个预测框，这有助于增强模型对复杂场景和小目标的检测能力。RTDETRv3的另一个重要创新是结合了DETR和YOLO的优势，引入了辅助的PPYOLOEHead来提供额外的监督信号。
若要了解更多细节，请参考论文[paper](https://arxiv.org/pdf/2409.08475).

## 模型

| Model | Epoch | Backbone  | Input shape | $AP^{val}$ | $AP^{val}_{50}$| Params(M) | FLOPs(G) |  T4 TensorRT FP16(FPS) | Weight | Config |
|:--------------:|:-----:|:----------:|:-------:|:--------------------------:|:---------------------------:|:---------:|:--------:|:---------------------:|:------------------------------------------------------------------------------------:|:-------------------------------------------:|
| RT-DETRv3-R18 | 6x |  ResNet-18 | 640 | 48.1 | 65.6 | 20 | 60 | 217 | [download](https://paddledet.bj.bcebos.com/models/rtdetrv3_r18vd_6x.pdparams)| [config](./rtdetrv3_r18vd_6x_coco.yml) |
| RT-DETRv3-R34 | 6x |  ResNet-34 | 640 | 49.9 | 67.7 | 31 | 92 | 161 | [download](https://paddledet.bj.bcebos.com/models/rtdetrv3_r34vd_6x.pdparams)| [config](./rtdetrv3_r34vd_6x_coco.yml) |
| RT-DETRv3-R50 | 6x |  ResNet-50 | 640 | 52.8 | 71.1 | 42 | 136 | 108 | [download](https://paddledet.bj.bcebos.com/models/rtdetrv3_r50vd_6x.pdparams)| [config](./rtdetrv3_r50vd_6x_coco.yml) |


**注意事项:**

- RT-DETRv3模型均使用4个GPU训练。
- RT-DETRv3在COCO train2017上训练，并在val2017上评估。

## 快速开始

<details open>
<summary>依赖包:</summary>

- PaddlePaddle >= 2.4.1

</details>

<details>
<summary>安装</summary>

- [安装指导文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/INSTALL.md)

</details>

<details>
<summary>训练&评估</summary>

- 单卡GPU上训练:

```shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml --eval
```

- 多卡GPU上训练:

```shell
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml --fleet --eval
```

- 评估:

```shell
python tools/eval.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
              -o weights=https://paddledet.bj.bcebos.com/models/rtdetrv3_r18vd_6x.pdparams
```

- 测试:

```shell
python tools/infer.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
              -o weights=https://paddledet.bj.bcebos.com/models/rtdetrv3_r18vd_6x.pdparams \
              --infer_img=./demo/000000570688.jpg
```

详情请参考[快速开始文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED.md).

</details>

## 部署

<details open>
<summary>1. 导出模型 </summary>

```shell
cd PaddleDetection
python tools/export_model.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
              -o weights=https://paddledet.bj.bcebos.com/models/rtdetrv3_r18vd_6x.pdparams trt=True \
              --output_dir=output_inference
```

</details>

<details>
<summary>2. 转换模型至ONNX </summary>

- 安装[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) 和 ONNX

```shell
pip install onnx==1.13.0
pip install paddle2onnx==1.0.5
```

- 转换模型:

```shell
paddle2onnx --model_dir=./output_inference/rtdetrv3_r18vd_6x_coco/ \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 16 \
            --save_file rtdetrv3_r18vd_6x_coco.onnx
```

</details>

<details>
<summary>3. 转换成TensorRT（可选） </summary>

- 基础模型请确保TensorRT的版本>=8.5.1，离散采样模型支持TensorRT的版本==8.4甚至一些更早的版本
- TRT推理可以参考[RT-DETR](https://github.com/lyuwenyu/RT-DETR)的部分代码或者其他网络资源

```shell
trtexec --onnx=./rtdetrv3_r18vd_6x_coco.onnx \
        --workspace=4096 \
        --shapes=image:1x3x640x640 \
        --saveEngine=rtdetrv3_r18vd_6x_coco.trt \
        --avgRuns=100 \
        --fp16
```

</details>

## 引用

```
@article{wang2024rt,
  title={RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Positive Supervision},
  author={Wang, Shuo and Xia, Chunlong and Lv, Feng and Shi, Yifeng},
  journal={arXiv preprint arXiv:2409.08475},
  year={2024}
}
