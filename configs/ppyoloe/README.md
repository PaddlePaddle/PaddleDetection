English | [简体中文](README_cn.md)

# PP-YOLOE

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model-Zoo)
- [Getting Start](#Getting-Start)
- [Appendix](#Appendix)

## Introduction
PP-YOLOE is an excellent single-stage anchor-free model based on PP-YOLOv2, surpassing a variety of popular yolo models. PP-YOLOE has a series of models, named s/m/l/x, which are configured through width multiplier and depth multiplier. PP-YOLOE avoids using special operators, such as deformable convolution or matrix nms, to be deployed friendly on various hardware. For more details, please refer to our report.

<div align="center">
  <img src="../../docs/images/ppyoloe_map_fps.png" width=500 />
</div>

PP-YOLOE-l achieves 51.4 mAP on COCO test-dev2017 dataset with 78.1 FPS on Tesla V100. While using TensorRT FP16, PP-YOLOE-l can be further accelerated to 149.2 FPS. PP-YOLOE-s/m/x also have excellent accuracy and speed performance, which can be found in [Model Zoo](#Model-Zoo)

PP-YOLOE is composed of following methods:
- Scalable backbone and neck
- [Task Alignment Learning](https://arxiv.org/abs/2108.07755)
- Efficient Task-aligned head with [DFL](https://arxiv.org/abs/2006.04388) and [VFL](https://arxiv.org/abs/2008.13367)
- [SiLU activation function](https://arxiv.org/abs/1710.05941)

## Model Zoo
|          Model           | GPU number | images/GPU |  backbone  | input shape | Box AP<sup>val</sup> | Box AP<sup>test</sup> | V100 FP32(FPS) | V100 TensorRT FP16(FPS) | download | config  |
|:------------------------:|:-------:|:-------------:|:----------:| :-------:| :------------------: | :-------------------: | :------------: | :---------------------: | :------: | :------: |
| PP-YOLOE-s                  |     8      |     32     | cspresnet-s |     640     |       42.7        |        43.1         |      208.3      |          333.3          | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml)                   |
| PP-YOLOE-m                  |     8      |     32     | cspresnet-m |     640     |       48.6        |        48.9         |      123.4      |          208.3          | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyoloe/ppyoloe_crn_m_300e_coco.yml)                   |
| PP-YOLOE-l                  |     8      |     24     | cspresnet-l |     640     |       50.9        |        51.4         |      78.1      |          149.2          | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml)                   |
| PP-YOLOE-x                  |     8      |     16     | cspresnet-x |     640     |       51.9        |        52.2         |      45.0      |          95.2          | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_x_300e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyoloe/ppyoloe_crn_x_300e_coco.yml)                   |

**Notes:**

- PP-YOLOE is trained on COCO train2017 dataset and evaluated on val2017 & test-dev2017 dataset，Box AP<sup>test</sup> is evaluation results of `mAP(IoU=0.5:0.95)`.
- PP-YOLOE used 8 GPUs for mixed precision training, if GPU number and mini-batch size is changed, learning rate and iteration times should be adjusted according [FAQ](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/FAQ).
- PP-YOLOE inference speed is tesed on single Tesla V100 with batch size as 1, CUDA 10.2, CUDNN 7.6.5, TensorRT 6.0.1.8 in TensorRT mode.
- PP-YOLOE inference speed testing uses inference model exported by `tools/export_model.py` with `-o exclude_nms=True` and benchmarked by running `depoly/python/infer.py` with `--run_benchmark`. All testing results do not contains the time cost of data reading and post-processing(NMS), which is same as [YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet) in testing method.
- If you set `--run_benchmark=True`，you should install these dependencies at first, `pip install pynvml psutil GPUtil`.

## Getting Start

### 1. Training

Training PP-YOLOE with mixed precision on 8 GPUs with following command

```bash
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml --amp
```

** Notes: ** use `--amp` to train with default config to avoid out of memeory.

### 2. Evaluation

Evaluating PP-YOLOE on COCO val2017 dataset in single GPU with following commands:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
```

For evaluation on COCO test-dev2017 dataset, please download COCO test-dev2017 dataset from [COCO dataset download](https://cocodataset.org/#download) and decompress to COCO dataset directory and configure `EvalDataset` like `configs/ppyolo/ppyolo_test.yml`.

### 3. Inference

Inference images in single GPU with following commands, use `--infer_img` to inference a single image and `--infer_dir` to inference all images in the directory.

```bash
# inference single image
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams --infer_img=demo/000000014439_640x640.jpg

# inference all images in the directory
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams --infer_dir=demo
```

### 4. Deployment

For deployment on GPU or benchmarked, model should be first exported to inference model using `tools/export_model.py`.

Exporting PP-YOLOE for Paddle Inference **without TensorRT**, use following command.

```bash
python tools/export_model.py configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
```

Exporting PP-YOLOE for Paddle Inference **with TensorRT** for better performance, use following command with extra `-o trt=True` setting.

```bash
python tools/export_model.py configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams -o trt=True
```

`deploy/python/infer.py` is used to load exported paddle inference model above for inference and benchmark through PaddleInference.

```bash
# inference single image
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyolo_r50vd_dcn_1x_coco --image_file=demo/000000014439_640x640.jpg --device=gpu

# inference all images in the directory
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyolo_r50vd_dcn_1x_coco --image_dir=demo/ --device=gpu

# benchmark
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyolo_r50vd_dcn_1x_coco --image_file=demo/000000014439_640x640.jpg --device=gpu --run_benchmark=True
```

If you want to export PP-YOLOE model to **ONNX format**, use following command refer to [PaddleDetection Model Export as ONNX Format Tutorial](../../deploy/EXPORT_ONNX_MODEL_en.md).

```bash
# export inference model
python tools/export_model.py configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml --output_dir=output_inference -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams

# install paddle2onnx
pip install paddle2onnx

# convert to onnx
paddle2onnx --model_dir output_inference/ppyoloe_crn_l_300e_coco --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 11 --save_file ppyoloe_crn_l_300e_coco.onnx

```

## Appendix

Ablation experiments of PP-YOLOE.

| NO.  |        Model                 | Box AP<sup>val</sup> | Params(M) | FLOPs(G) | V100 FP32 FPS |
| :--: | :---------------------------: | :------------------: | :-------: | :------: | :-----------: |
|  A   | PP-YOLOv2          |         49.1         |   54.58   |  115.77   |     68.9     |
|  B   | A + Anchor-free    |         48.8         |   54.27   |  114.78   |     69.8     |
|  C   | B + CSPRepResNet   |         49.5         |   47.42   |  101.87   |     85.5     |
|  D   | C + TAL            |         50.4         |   48.32   |  104.75   |     84.0     |
|  E   | D + ET-Head        |         50.9         |   52.20   |  110.07   |     78.1     |
