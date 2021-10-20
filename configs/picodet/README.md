# PicoDet

![](../../docs/images/picedet_demo.jpeg)
## Introduction

We developed a series of lightweight models, which named `PicoDet`. Because of its excellent performance, it is very suitable for deployment on mobile or CPU.

- 🌟 Higher mAP: the **first** object detectors that surpass mAP(0.5:0.95) **30+** within 1M parameters when the input size is 416.
- 🚀 Faster latency: 114FPS on mobile ARM CPU.
- 😊 Deploy friendly: support PaddleLite/MNN/NCNN/OpenVINO and provide C++/Python/Android implementation.
- 😍 Advanced algorithm: use the most advanced algorithms and innovate, such as ESNet, CSP-PAN, SimOTA with VFL, etc.

### Comming soon
- [ ] More series of model, such as smaller or larger model.
- [ ] Pretrained models for more scenarios.
- [ ] More features in need.

## Requirements
- PaddlePaddle >= 2.1.2
- PaddleSlim >= 2.1.1

## Model Zoo

| Model                  | Input size | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | FLOPS<br><sup>(G) | Params<br><sup>(M) | Latency<br><sup>(ms) |                           download                          | config |
| :------------------------ | :-------:  | :------: | :---: | :---: | :---: | :------------:  | :-------------------------------------------------: | :-----: |
| PicoDet-S    | 320*320    |   27.1     | 41.4 | -- | 3.9M | -- | [model](https://paddledet.bj.bcebos.com/models/picodet_s_320_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_s_320_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_s_320_coco.yml) |
| PicoDet-M    | 320*320   |  30.9     | 45.7 |  -- | 8.4M | -- | [model](https://paddledet.bj.bcebos.com/models/picodet_m_320_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_m_320_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_m_320_coco.yml) |
| PicoDet-L    | 320*320   |  32.6     | 47.9 |  -- | 13M | -- | [model](https://paddledet.bj.bcebos.com/models/picodet_l_320_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_l_320_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_l_320_coco.yml) |
| PicoDet-S    | 416*416    |   30.6     | 45.5 | -- | 3.9M | -- | [model](https://paddledet.bj.bcebos.com/models/picodet_s_416_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_s_416_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_s_416_coco.yml) |
| PicoDet-M    | 416*416    |   34.3     | 49.8 |  -- | 8.4M | -- | [model](https://paddledet.bj.bcebos.com/models/picodet_m_416_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_m_416_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_m_416_coco.yml) |
| PicoDet-L    | 416*416   |  -    |   -  |  -- | 13M | -- | [model](https://paddledet.bj.bcebos.com/models/picodet_l_416_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_l_416_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_l_416_coco.yml) |
| PicoDet-L    | 640*640   |  -    |  -   |  -- | 13M | -- | [model](https://paddledet.bj.bcebos.com/models/picodet_l_640_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_l_640_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_l_640_coco.yml) |

**Notes:**

- PicoDet inference speed is tested on Snapdragon 888(4xA78+4xA55) with 4 threads by arm8 and with FP16.
- PicoDet is trained on COCO train2017 dataset and evaluated on val2017.
- PicoDet used 4 or 8 GPUs for training.

## Deployment

### Export and Convert model

<details>
<summary>1. Export model</summary>

```shell
cd PaddleDetection
python tools/export_model.py -c configs/picodet/picodet_s_320_coco.yml \
              -o weights=https://paddledet.bj.bcebos.com/models/picodet_s_320_coco.pdparams --output_dir=inference_model
```

</details>

<details>
<summary>2. Convert to PaddleLite</summary>

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
<summary>3. Convert to ONNX</summary>

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
- [NCNN C++ demo]()
- [MNN C++ demo]()
- [OpenVINO C++ demo]()
- [Android demo]()

## Slim

### quantization

<details>
<summary>Quant aware</summary>

Configure the quant config and start training:

```shell
python tools/train.py -c configs/picodet/picodet_s_320_coco.yml \
          --slim_config configs/slim/quant/picodet_s_quant.yml --eval
```

</details>

<details>
<summary>Post quant</summary>

Configure the post quant config and start calibrate model:

```shell
python tools/posy_quant.py -c configs/picodet/picodet_s_320_coco.yml \
          --slim_config configs/slim/posy_quant/picodet_s_quant.yml
```

</details>

## Cite PiocDet
If you use PiocDet in your research, please cite our work by using the following BibTeX entry:
```
comming soon

```
