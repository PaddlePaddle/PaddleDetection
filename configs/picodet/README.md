# PicoDet

![](../../docs/images/picedet_demo.jpeg)
## Introduction

We developed a series of lightweight models, which named `PicoDet`. Because of its excellent performance, it is very suitable for deployment on mobile or CPU.

- 🌟 Higher mAP: the **first** object detectors that surpass mAP(0.5:0.95) **30+** within 1M parameters when the input size is 416.
- 🚀 Faster latency: 129FPS on mobile ARM CPU.
- 😊 Deploy friendly: support PaddleLite/MNN/NCNN/OpenVINO and provide C++/Python/Android implementation.
- 😍 Advanced algorithm: use the most advanced algorithms and innovate, such as ESNet, CSP-PAN, SimOTA with VFL, etc.

### Comming soon
- [ ] More series of model, such as smaller or larger model.
- [ ] Pretrained models for more scenarios.
- [ ] More features in need.

## Requirements
- PaddlePaddle >= 2.1.2
- PaddleSlim >= 2.1.1

## Benchmark

| Model                  | Input size | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params<br><sup>(M) | FLOPS<br><sup>(G)  | Latency<br><sup>(ms) |                           download                          | config |
| :------------------------ | :-------:  | :------: | :---: | :---: | :---: | :------------:  | :-------------------------------------------------: | :-----: |
| PicoDet-S    | 320*320    |   27.1     | 41.4 | 0.99 | 0.73 | 7.78 | [model](https://paddledet.bj.bcebos.com/models/picodet_s_320_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_s_320_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_s_320_coco.yml) |
| PicoDet-S    | 416*416    |   30.6     | 45.5 | 0.99 | 1.24 | 11.84 | [model](https://paddledet.bj.bcebos.com/models/picodet_s_416_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_s_416_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_s_416_coco.yml) |
| PicoDet-M    | 320*320   |  30.9     | 45.7 |  2.15 | 1.48 | 10.56 | [model](https://paddledet.bj.bcebos.com/models/picodet_m_320_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_m_320_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_m_320_coco.yml) |
| PicoDet-M    | 416*416    |   34.3     | 49.8 |  2.15 | 2.50 | 15.87 | [model](https://paddledet.bj.bcebos.com/models/picodet_m_416_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_m_416_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_m_416_coco.yml) |
| PicoDet-L    | 320*320   |  32.6     | 47.9 |  3.24 | 2.18 | 12.82 | [model](https://paddledet.bj.bcebos.com/models/picodet_l_320_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_l_320_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_l_320_coco.yml) |
| PicoDet-L    | 416*416   |  35.9    |   51.7  |  3.24 | 3.69 | 19.42 | [model](https://paddledet.bj.bcebos.com/models/picodet_l_416_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_l_416_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_l_416_coco.yml) |
| PicoDet-L    | 640*640   |  40.3    |  57.1   |  3.24 | 8.74 | 41.52 | [model](https://paddledet.bj.bcebos.com/models/picodet_l_640_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_l_640_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_l_640_coco.yml) |


<details>
<summary>Table Notes (click to expand)</summary>

- PicoDet inference speed is tested on SD 888(1*X1+3*A78+4*A55) with 4 threads by arm8 and with FP16.
- PicoDet is trained on COCO train2017 dataset and evaluated on COCO val2017.
- PicoDet used 4 or 8 GPUs for training and all checkpoints are trained to 300 epochs with default settings and hyperparameters.

</details>

## More config

| Model                  | Input size | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | Params<br><sup>(M) | FLOPS<br><sup>(G)  | Latency<br><sup>(ms) |                           download                          | config |
| :------------------------ | :-------:  | :------: | :---: | :---: | :---: | :------------:  | :-------------------------------------------------: | :-----: |
| PicoDet-Shufflenetv2 1x    | 416*416    |   30.0     | 44.6 | 1.17 | 1.53 | 14.76 | [model](https://paddledet.bj.bcebos.com/models/picodet_shufflenetv2_1x_416_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_shufflenetv2_1x_416_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/more_config/picodet_shufflenetv2_1x_416_coco.yml) |
| PicoDet-MobileNetv3-large 1x    | 416*416    |   35.6    | 52.0 | 3.55 | 2.80 | 18.87 | [model](https://paddledet.bj.bcebos.com/models/picodet_mobilenetv3_large_1x_416_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_mobilenetv3_large_1x_416_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/more_config/picodet_mobilenetv3_large_1x_416_coco.yml) |
| PicoDet-LCNet 1.5x    | 416*416    |   36.3    | 52.2 | 3.10 | 3.85 | 19.75 | [model](https://paddledet.bj.bcebos.com/models/picodet_lcnet_1_5x_416_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_lcnet_1_5x_416_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/more_config/picodet_lcnet_1_5x_416_coco.yml) |

## Deployment

### Export and Convert model

<details>
<summary>1. Export model (click to expand)</summary>

```shell
cd PaddleDetection
python tools/export_model.py -c configs/picodet/picodet_s_320_coco.yml \
              -o weights=https://paddledet.bj.bcebos.com/models/picodet_s_320_coco.pdparams --output_dir=inference_model
```

</details>

<details>
<summary>2. Convert to PaddleLite (click to expand)</summary>

- Install Paddlelite>=2.10.rc:

```shell
pip install paddlelite
```

- Convert model:

```shell
# FP32
paddle_lite_opt --model_dir=inference_model/picodet_s_320_coco --valid_targets=arm --optimize_out=picodet_s_320_coco_fp32
# FP16
paddle_lite_opt --model_dir=inference_model/picodet_s_320_coco --valid_targets=arm --optimize_out=picodet_s_320_coco_fp16 --enable_fp16=true
```

</details>

<details>
<summary>3. Convert to ONNX (click to expand)</summary>

- Install [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) >= 0.7 and ONNX > 1.10.1, for details, please refer to [Tutorials of Export ONNX Model](../../deploy/EXPORT_ONNX_MODEL.md)

```shell
pip install onnx
pip install paddle2onnx
```

- Convert model:

```shell
paddle2onnx --model_dir output_inference/picodet_s_320_coco/ \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 11 \
            --save_file picodet_s_320_coco.onnx
```

- Simplify ONNX model: use onnx-simplifier to simplify onnx model.

  - Install onnx-simplifier >= 0.3.6:
  ```shell
  pip install onnx-simplifier
  ```
  - simplify onnx model:
  ```shell
  python -m onnxsim picodet_s_320_coco.onnx picodet_s_processed.onnx
  ```

</details>

### Deploy

- PaddleInference demo [Python](../../deploy/python) & [C++](../../deploy/cpp)
- [PaddleLite C++ demo](../../deploy/lite)
- [NCNN C++/Python demo](../../deploy/third_engine/demo_ncnn)
- [MNN C++/Python demo](../../deploy/third_engine/demo_mnn)
- [OpenVINO C++/Python demo](../../deploy/third_engine/demo_openvino)
- [Android demo]()

## Slim

### quantization

<details>
<summary>Quant aware (click to expand)</summary>

Configure the quant config and start training:

```shell
python tools/train.py -c configs/picodet/picodet_s_320_coco.yml \
          --slim_config configs/slim/quant/picodet_s_quant.yml --eval
```

</details>

<details>
<summary>Post quant (click to expand)</summary>

Configure the post quant config and start calibrate model:

```shell
python tools/post_quant.py -c configs/picodet/picodet_s_320_coco.yml \
          --slim_config configs/slim/post_quant/picodet_s_ptq.yml
```

</details>

## Cite PiocDet
If you use PiocDet in your research, please cite our work by using the following BibTeX entry:
```
comming soon

```
