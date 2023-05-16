# DETRs Beat YOLOs on Real-time Object Detection

## 最新动态

- 发布RT-DETR-R50和RT-DETR-R101的代码和预训练模型
- 发布RT-DETR-L和RT-DETR-X的代码和预训练模型
- 发布RT-DETR-R50-m模型（scale模型的范例）
- 发布RT-DETR-R34模型
- 发布RT-DETR-R18模型

## 简介
<!-- We propose a **R**eal-**T**ime **DE**tection **TR**ansformer (RT-DETR), the first real-time end-to-end object detector to our best knowledge. Specifically, we design an efficient hybrid encoder to efficiently process multi-scale features by decoupling the intra-scale interaction and cross-scale fusion, and propose IoU-aware query selection to improve the initialization of object queries. In addition, our proposed detector supports flexibly adjustment of the inference speed by using different decoder layers without the need for retraining, which facilitates the practical application of real-time object detectors. Our RT-DETR-L achieves 53.0% AP on COCO val2017 and 114 FPS on T4 GPU, while RT-DETR-X achieves 54.8% AP and 74 FPS, outperforming all YOLO detectors of the same scale in both speed and accuracy. Furthermore, our RT-DETR-R50 achieves 53.1% AP and 108 FPS, outperforming DINO-Deformable-DETR-R50 by 2.2% AP in accuracy and by about 21 times in FPS.  -->
RT-DETR是第一个实时端到端目标检测器。具体而言，我们设计了一个高效的混合编码器，通过解耦尺度内交互和跨尺度融合来高效处理多尺度特征，并提出了IoU感知的查询选择机制，以优化解码器查询的初始化。此外，RT-DETR支持通过使用不同的解码器层来灵活调整推理速度，而不需要重新训练，这有助于实时目标检测器的实际应用。RT-DETR-L在COCO val2017上实现了53.0%的AP，在T4 GPU上实现了114FPS，RT-DETR-X实现了54.8%的AP和74FPS，在速度和精度方面都优于相同规模的所有YOLO检测器。RT-DETR-R50实现了53.1%的AP和108FPS，RT-DETR-R101实现了54.3%的AP和74FPS，在精度上超过了全部使用相同骨干网络的DETR检测器。
若要了解更多细节，请参考我们的论文[paper](https://arxiv.org/abs/2304.08069).

<div align="center">
  <img src="https://github.com/PaddlePaddle/PaddleDetection/assets/17582080/3184a08e-aa4d-49cf-9079-f3695c4cc1c3" width=500 />
</div>

## 模型

| Model | Epoch | backbone  | input shape | $AP^{val}$ | $AP^{val}_{50}$| Params(M) | FLOPs(G) |  T4 TensorRT FP16(FPS) | Pretrained Model | config |
|:--------------:|:-----:|:----------:| :-------:|:--------------------------:|:---------------------------:|:---------:|:--------:| :---------------------: |:------------------------------------------------------------------------------------:|:-------------------------------------------:|
| RT-DETR-R18 | 6x |  ResNet-18 | 640 | 46.5 | 63.8 | 22 | 60 | 217 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r18vd_dec3_6x_coco.pdparams) | [config](./rtdetr_r18vd_6x_coco.yml)
| RT-DETR-R34 | 6x |  ResNet-34 | 640 | 48.9 | 66.8 | 32 | 92 | 161 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r34vd_dec4_6x_coco.pdparams) | [config](./rtdetr_r34vd_6x_coco.yml)
| RT-DETR-R50-m | 6x |  ResNet-50 | 640 | 51.3 | 69.6 | 35 | 100 | 145 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_m_6x_coco.pdparams) | [config](./rtdetr_r50vd_m_6x_coco.yml)
| RT-DETR-R50 | 6x |  ResNet-50 | 640 | 53.1 | 71.3 | 42 | 136 | 108 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams) | [config](./rtdetr_r50vd_6x_coco.yml)
| RT-DETR-R101 | 6x |  ResNet-101 | 640 | 54.3 | 72.7 | 76 | 259 | 74 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r101vd_6x_coco.pdparams) | [config](./rtdetr_r101vd_6x_coco.yml)
| RT-DETR-L | 6x |  HGNetv2 | 640 | 53.0 | 71.6 | 32 | 110 | 114 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_hgnetv2_l_6x_coco.pdparams) | [config](rtdetr_hgnetv2_l_6x_coco.yml)
| RT-DETR-X | 6x |  HGNetv2 | 640 | 54.8 | 73.1 | 67 | 234 | 74 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_hgnetv2_x_6x_coco.pdparams) | [config](rtdetr_hgnetv2_x_6x_coco.yml)



**注意事项:**
- RT-DETR 使用4个GPU训练。
- RT-DETR 在COCO train2017上训练，并在val2017上评估。

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
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --eval
```

- 多卡GPU上训练:

```shell
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --fleet --eval
```

- 评估:

```shell
python tools/eval.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams
```

- 测试:

```shell
python tools/infer.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams \
              --infer_img=./demo/000000570688.jpg
```

详情请参考[快速开始文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED.md).

</details>

## 部署

<details open>
<summary>1. 导出模型 </summary>

```shell
cd PaddleDetection
python tools/export_model.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams trt=True \
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
paddle2onnx --model_dir=./output_inference/rtdetr_r50vd_6x_coco/ \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 16 \
            --save_file rtdetr_r50vd_6x_coco.onnx
```
</details>

<details>
<summary>3. 转换成TensorRT（可选） </summary>

- 确保TensorRT的版本>=8.5.1
- TRT推理可以参考[RT-DETR](https://github.com/lyuwenyu/RT-DETR)的部分代码或者其他网络资源

```shell
trtexec --onnx=./rtdetr_r50vd_6x_coco.onnx \
        --workspace=4096 \
        --shapes=image:1x3x640x640 \
        --saveEngine=rtdetr_r50vd_6x_coco.trt \
        --avgRuns=100 \
        --fp16
```

-
</details>


## 其他

<details>
<summary>1. 参数量和计算量统计 </summary>
可以使用以下代码片段实现参数量和计算量的统计

```
import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.core.workspace import create

cfg_path = './configs/rtdetr/rtdetr_r50vd_6x_coco.yml'
cfg = load_config(cfg_path)
model = create(cfg.architecture)

blob = {
    'image': paddle.randn([1, 3, 640, 640]),
    'im_shape': paddle.to_tensor([[640], [640]]),
    'scale_factor': paddle.to_tensor([[1.], [1.]])
}
paddle.flops(model, None, blob, custom_ops=None, print_detail=False)
```
</details>


<details open>
<summary>2. YOLOs端到端速度测速 </summary>

- 可以参考[RT-DETR](https://github.com/lyuwenyu/RT-DETR) benchmark部分或者其他网络资源

</details>



## 引用RT-DETR
如果需要在你的研究中使用RT-DETR，请通过以下方式引用我们的论文：
```
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
