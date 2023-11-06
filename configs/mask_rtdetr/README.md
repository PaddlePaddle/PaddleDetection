# Mask-RT-DETR


## 简介
Mask-RT-DETR是一个实例分割模型。基于RT-DETR和MaskDINO。

## 模型库
|        Model        | Epoch | Backbone | Input shape | Box AP | Mask AP | Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS) | Pretrained Model |                      config                      |
|:-------------------:|:-----:|:--------:|:-----------:|:------:|:-------:|:---------:|:--------:|:---------------------:|:----------------:|:------------------------------------------------:|
|   Mask-RT-DETR-L    |  6x   | HGNetv2  |     640     |        |         |           |          |                       |                  |   [config](mask_rtdetr_hgnetv2_l_6x_coco.yml)    |


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
python tools/train.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml --eval
```

- 多卡GPU上训练:

```shell
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml --amp --eval
```

- 评估:

```shell
python tools/eval.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml \
              -o weights=${model_params_path}
```

- 测试:

```shell
python tools/infer.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml \
              -o weights=${model_params_path} \
              --infer_img=./demo/000000570688.jpg
```

详情请参考[快速开始文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED.md).

</details>

## 部署

<details open>
<summary>1. 导出模型 </summary>

```shell
cd PaddleDetection
python tools/export_model.py -c configs/mask_rtdetr/mask_rtdetr_hgnetv2_l_6x_coco.yml \
              -o weights=${model_params_path} trt=True exclude_post_process=True \
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
paddle2onnx --model_dir=./output_inference/mask_rtdetr_hgnetv2_l_6x_coco/ \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 16 \
            --save_file mask_rtdetr_hgnetv2_l_6x_coco.onnx
```
</details>

<details>
<summary>3. 转换成TensorRT（可选） </summary>

- 确保TensorRT的版本>=8.5.1
- TRT推理可以参考[RT-DETR](https://github.com/lyuwenyu/RT-DETR)的部分代码或者其他网络资源

```shell
trtexec --onnx=./mask_rtdetr_hgnetv2_l_6x_coco.onnx \
        --workspace=4096 \
        --shapes=image:1x3x640x640 \
        --saveEngine=mask_rtdetr_hgnetv2_l_6x_coco.trt \
        --avgRuns=100 \
        --fp16
```