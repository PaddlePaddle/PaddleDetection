# Improved Baselines with Bag-of-freebies for Real-time Detection with Transformers

## 简介

RT-DETRv2是基于 Transformer 的实时端到端检测器。它在SOTA的 RT-DETR
的基础上，引入了灵活的解码器，并运用了一系列有效的训练策略。具体而言，我们为解码器的各种特征图建议了不同数量的采样点，在多个训练阶段采用动态数据增强策略，并为每个独特的模型确定特定的优化超参数。为适应各种部署方案，解码器现在提供了一个利用离散采样而非网格采样的选项。RT-DETRv2-R18
在相同速度下相比 RT-DETR-R18 实现了 1.4 的提升，在 T4 GPU 上以 FP16 模式达到了 47.9 mAP 和 217 FPS。若要了解更多细节，请参考论文[paper](https://arxiv.org/pdf/2407.17140).

## 基础模型

|      Model       | Epoch |  Backbone  | Input shape | $AP^{val}$ | $AP^{val}_{50}$ | Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS) |                                    Pretrained Model                                     |                  config                  |
|:----------------:|:-----:|:----------:|:-----------:|:----------:|:---------------:|:---------:|:--------:|:---------------------:|:---------------------------------------------------------------------------------------:|:----------------------------------------:|
|  *RT-DETRv2-R18  |  120  | ResNet-18  |     640     |    47.9    |      64.9       |    20     |    60    |          217          | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetrv2_r18vd_120e_coco.pdparams) | [config](./rtdetrv2_r18vd_120e_coco.yml) |
|  *RT-DETRv2-R34  |  120  | ResNet-34  |     640     |    49.9    |      67.6       |    31     |    92    |          161          | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetrv2_r34vd_120e_coco.pdparams) | [config](./rtdetrv2_r34vd_120e_coco.yml) |
| *RT-DETRv2-R50-m |  84   | ResNet-50  |     640     |    51.9    |      69.8       |    36     |   100    |          145          | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetrv2_r50vd_m_7x_coco.pdparams) | [config](./rtdetrv2_r50vd_m_7x_coco.yml) |
|  *RT-DETRv2-R50  |  72   | ResNet-50  |     640     |    53.4    |      71.6       |    42     |   136    |          108          |  [download](https://bj.bcebos.com/v1/paddledet/models/rtdetrv2_r50vd_6x_coco.pdparams)  |  [config](./rtdetrv2_r50vd_6x_coco.yml)  |
| *RT-DETRv2-R101  |  72   | ResNet-101 |     640     |    54.3    |      72.8       |    76     |   259    |          74           | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetrv2_r101vd_6x_coco.pdparams)  | [config](./rtdetrv2_r101vd_6x_coco.yml)  |

## 离散采样调优模型

|        Model         | Epoch | Backbone  | Input shape | $AP^{val}$ | $AP^{val}_{50}$ | Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS) |                                      Pretrained Model                                       |                    config                    |
|:--------------------:|:-----:|:---------:|:-----------:|:----------:|:---------------:|:---------:|:--------:|:---------------------:|:-------------------------------------------------------------------------------------------:|:--------------------------------------------:|
|  *RT-DETRv2-R18-dsp  |  36   | ResNet-18 |     640     |    47.4    |      64.8       |    20     |    60    |          217          |  [download](https://bj.bcebos.com/v1/paddledet/models/rtdetrv2_r18vd_dsp_3x_coco.pdparams)  |  [config](./rtdetrv2_r18vd_dsp_3x_coco.yml)  |
|  *RT-DETRv2-R34-dsp  |  12   | ResNet-34 |     640     |    49.2    |      67.2       |    31     |    92    |          161          |  [download](https://bj.bcebos.com/v1/paddledet/models/rtdetrv2_r34vd_dsp_1x_coco.pdparams)  |  [config](./rtdetrv2_r34vd_dsp_1x_coco.yml)  |
| *RT-DETRv2-R50-m-dsp |  36   | ResNet-50 |     640     |    51.3    |      69.7       |    36     |   100    |          145          | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetrv2_r50vd_m_dsp_3x_coco.pdparams) | [config](./rtdetrv2_r50vd_m_dsp_3x_coco.yml) |
|  *RT-DETRv2-R50-dsp  |  12   | ResNet-50 |     640     |    52.8    |      71.3       |    42     |   136    |          108          |  [download](https://bj.bcebos.com/v1/paddledet/models/rtdetrv2_r50vd_dsp_1x_coco.pdparams)  |  [config](./rtdetrv2_r50vd_dsp_1x_coco.yml)  |

**注意事项:**

- RT-DETRv2 基础模型均使用4个GPU训练。
- RT-DETRv2 在COCO train2017上训练，并在val2017上评估。
- 基础模型的采样方法默认为`grid_sample`,
  离散采样调优模型则使用`discrete_sample`, [详见](../../ppdet/modeling/transformers/utils.py)。
- 离散采样调优模型使用对应基础模型权重作为预训练权重。

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
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml --eval
```

- 多卡GPU上训练:

```shell
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml --fleet --eval
```

- 评估:

```shell
python tools/eval.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetrv2_r50vd_6x_coco.pdparams
```

- 测试:

```shell
python tools/infer.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetrv2_r50vd_6x_coco.pdparams \
              --infer_img=./demo/000000570688.jpg
```

详情请参考[快速开始文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED.md).

</details>

## 部署

<details open>
<summary>1. 导出模型 </summary>

```shell
cd PaddleDetection
python tools/export_model.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetrv2_r50vd_6x_coco.pdparams trt=True \
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
paddle2onnx --model_dir=./output_inference/rtdetrv2_r50vd_6x_coco/ \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 16 \
            --save_file rtdetrv2_r50vd_6x_coco.onnx
```

</details>

<details>
<summary>3. 转换成TensorRT（可选） </summary>

- 基础模型请确保TensorRT的版本>=8.5.1，离散采样模型支持TensorRT的版本==8.4甚至一些更早的版本
- TRT推理可以参考[RT-DETR](https://github.com/lyuwenyu/RT-DETR)的部分代码或者其他网络资源

```shell
trtexec --onnx=./rtdetrv2_r50vd_6x_coco.onnx \
        --workspace=4096 \
        --shapes=image:1x3x640x640 \
        --saveEngine=rtdetrv2_r50vd_6x_coco.trt \
        --avgRuns=100 \
        --fp16
```

</details>

## 引用

```
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
      title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer},
      author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
      year={2024},
      eprint={2407.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17140},
}
```
