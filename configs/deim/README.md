# DEIM: DETR with Improved Matching for Fast Convergence

## 简介

DEIM 是一种创新且高效的训练框架，专为加速基于 Transformer 架构的实时目标检测模型（如 DETR）收敛而设计。针对 DETR 中一对一（O2O）匹配所导致的监督稀疏问题，DEIM 引入了密集 O2O 匹配策略，通过标准数据增强手段引入更多正样本，从而提升训练效率。为缓解低质量匹配带来的负面影响，DEIM 提出了新颖的 Matchability-Aware Loss（MAL）函数，用于在不同质量水平下优化匹配效果。基于 COCO 数据集的大量实验证明，DEIM 能有效提升性能并显著缩短训练时间（可达 50%）。在与 RT-DETRv2 结合时，DEIM 在单张 NVIDIA 4090 显卡上一天内即可达到 53.2% 的 AP；在无需额外数据的前提下，DEIM-D-FINE-L 和 DEIM-D-FINE-X 更分别在 NVIDIA T4 上以 124 FPS 和 78 FPS 实现了 54.7% 与 56.5% 的 AP。

若要了解更多细节，请参考论文[paper](https://arxiv.org/abs/2412.04234).

## 基础模型

### DEIM-D-FINE
| Model | Epoch | $AP^{val}$ | $AP^{val}_{50}$ | Params | Latency | GFLOPs | config | Pretrained Model
| :---: | :---: | :--------: | :-------------: | :-----: | :-----: | :----: | :----: | :--------------:
| N | 160 | 43.0 | 60.2 |  4M |  2.12ms |   7 | [config](./deim_dfine/deim_hgnetv2_n_160e_coco.yml) | [download](https://paddledet.bj.bcebos.com/models/deim_hgnetv2_n_160e_coco.pdparams)
| S | 132 | 49.0 | 65.6 | 10M |  3.49ms |  25 | [config](./deim_dfine/deim_hgnetv2_s_132e_coco.yml) | [download](https://paddledet.bj.bcebos.com/models/deim_hgnetv2_s_132e_coco.pdparams)
| M | 102 | 52.7 | 69.6 | 19M |  5.62ms |  57 | [config](./deim_dfine/deim_hgnetv2_m_102e_coco.yml) | [download](https://paddledet.bj.bcebos.com/models/deim_hgnetv2_m_102e_coco.pdparams)
| L |  58 | 54.7 | 72.2 | 31M |  8.07ms |  91 | [config](./deim_dfine/deim_hgnetv2_l_58e_coco.yml)  | [download](https://paddledet.bj.bcebos.com/models/deim_hgnetv2_l_58e_coco.pdparams)
| X |  58 | 56.5 | 73.8 | 62M | 12.89ms | 202 | [config](./deim_dfine/deim_hgnetv2_x_58e_coco.yml)  | [download](https://paddledet.bj.bcebos.com/models/deim_hgnetv2_x_58e_coco.pdparams)


### DEIM-RT-DETRv2
| Model | Epoch | $AP^{val}$ | $AP^{val}_{50}$ | Params | Latency | GFLOPs | config | Pretrained Model
| :---: | :---: | :--------: | :-------------: | :-----: | :-----: | :----: | :----: | :--------------:
| S  | 120 | 49.0 | 66.1 | 20M |  4.59ms |  60 | [config](./deim_rtdetrv2/deim_r18vd_120e_coco.yml)  | [download](https://paddledet.bj.bcebos.com/models/deim_r18vd_120e_coco.pdparams)
| M  | 120 | 50.9 | 68.6 | 31M |  6.40ms |  92 | [config](./deim_rtdetrv2/deim_r34vd_120e_coco.yml)  | [download](https://paddledet.bj.bcebos.com/models/deim_r34vd_120e_coco.pdparams)
| M* |  60 | 53.2 | 71.2 | 33M |  6.90ms | 100 | [config](./deim_rtdetrv2/deim_r50vd_60e_coco.yml)   | [download](https://paddledet.bj.bcebos.com/models/deim_r50vd_60e_coco.pdparams)
| L  |  60 | 54.3 | 72.2 | 42M |  9.15ms | 136 | [config](./deim_rtdetrv2/deim_r50vd_m_60e_coco.yml) | [download](https://paddledet.bj.bcebos.com/models/deim_r50vd_m_60e_coco.pdparams)
| X  |  60 | 55.5 | 73.5 | 76M | 13.66ms | 259 | [config](./deim_rtdetrv2/deim_r101vd_60e_coco.yml)  | [download](https://paddledet.bj.bcebos.com/models/deim_r101vd_60e_coco.pdparams)


**注意事项:**

- DEIM 模型均使用4个GPU训练。
- DEIM 在COCO train2017上训练，并在val2017上评估。

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
python tools/train.py -c configs/deim/deim_rtdetrv2/deim_r18vd_120e_coco.yml --eval
```

- 多卡GPU上训练:

```shell
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/deim/deim_rtdetrv2/deim_r18vd_120e_coco.yml --fleet --eval
```

- 评估:

```shell
python tools/eval.py -c configs/deim/deim_rtdetrv2/deim_r18vd_120e_coco.yml \
              -o weights=https://paddledet.bj.bcebos.com/models/deim_r18vd_120e_coco.pdparams
```

- 测试:

```shell
python tools/infer.py -c configs/deim/deim_rtdetrv2/deim_r18vd_120e_coco.yml \
              -o weights=https://paddledet.bj.bcebos.com/models/deim_r18vd_120e_coco.pdparams \
              --infer_img=./demo/000000570688.jpg
```

详情请参考[快速开始文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED.md).

</details>

## 部署

<details open>
<summary>1. 导出模型 </summary>

```shell
cd PaddleDetection
python tools/export_model.py -c configs/deim/deim_rtdetrv2/deim_r18vd_120e_coco.yml \
              -o weights=https://paddledet.bj.bcebos.com/models/deim_r18vd_120e_coco.pdparams trt=True \
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
paddle2onnx --model_dir=./output_inference/deim_r18vd_120e_coco/ \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 16 \
            --save_file deim_r18vd_120e_coco.onnx
```

</details>

<details>
<summary>3. 转换成TensorRT（可选） </summary>

- 基础模型请确保TensorRT的版本>=8.5.1，离散采样模型支持TensorRT的版本==8.4甚至一些更早的版本
- TRT推理可以参考[RT-DETR](https://github.com/lyuwenyu/RT-DETR)的部分代码或者其他网络资源

```shell
trtexec --onnx=./deim_r18vd_120e_coco.onnx \
        --workspace=4096 \
        --shapes=image:1x3x640x640 \
        --saveEngine=deim_r18vd_120e_coco.trt \
        --avgRuns=100 \
        --fp16
```

</details>

## 引用

```
@misc{huang2024deim,
      title={DEIM: DETR with Improved Matching for Fast Convergence},
      author={Shihua Huang, Zhichao Lu, Xiaodong Cun, Yongjun Yu, Xiao Zhou, and Xi Shen},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2025},
}
```
