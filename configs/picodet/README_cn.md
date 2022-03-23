简体中文 | [English](README.md)

# PP-PicoDet

![](../../docs/images/picedet_demo.jpeg)

## 最新动态

- 发布全新系列PP-PicoDet模型，引入TAL及Task-aligned Head，优化PAN等结构，精度大幅提升，优化CPU端预测速度，同时训练速度大幅提升。**（2022.03.20）**

## 历史版本模型

- 详情请参考：[PicoDet 2021.10版本](./legacy_model/)

## 简介

PaddleDetection中提出了全新的轻量级系列模型`PP-PicoDet`，在移动端具有卓越的性能，成为全新SOTA轻量级模型。详细的技术细节可以参考我们的[arXiv技术报告](https://arxiv.org/abs/2111.00902)。

PP-PicoDet模型有如下特点：

- 🌟 更高的mAP: 第一个在1M参数量之内`mAP(0.5:0.95)`超越**30+**(输入416像素时)。
- 🚀 更快的预测速度: 网络预测在ARM CPU下可达150FPS。
- 😊 部署友好: 支持PaddleLite/MNN/NCNN/OpenVINO等预测库，支持转出ONNX，提供了C++/Python/Android的demo。
- 😍 先进的算法: 我们在现有SOTA算法中进行了创新, 包括：ESNet, CSP-PAN, SimOTA等等。


<div align="center">
  <img src="../../docs/images/picodet_map.png" width='600'/>
</div>

## 基线

| 模型     | 输入尺寸 | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | 参数量<br><sup>(M) | FLOPS<br><sup>(G) | 预测时延<sup><small>[NCNN](#latency)</small><sup><br><sup>(ms) | 预测时延<sup><small>[Lite](#latency)</small><sup><br><sup>(ms) |  下载  | 配置文件 |
| :-------- | :--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: | :-----------------------------: | :----------------------------------------: | :--------------------------------------- |
| PicoDet-XS |  320*320   |          23.5           |        36.1       |        -        |       -        |              -              |            -             | [model](https://paddledet.bj.bcebos.com/models/picodet_xs_320_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_xs_320_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_xs_320_coco_lcnet.yml) |
| PicoDet-XS |  416*416   |          26.2           |        39.3        |        -        |       -        |              -              |            -             | [model](https://paddledet.bj.bcebos.com/models/picodet_xs_416_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_xs_416_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_xs_416_coco_lcnet.yml) |
| PicoDet-S |  320*320   |          29.1           |        43.4        |        -       |       -       |             -              |            -             | [model](https://paddledet.bj.bcebos.com/models/picodet_s_320_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_s_320_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_s_320_coco_lcnet.yml) |
| PicoDet-S |  416*416   |          32.5           |        47.6        |        -        |       -       |              -              |            -             | [model](https://paddledet.bj.bcebos.com/models/picodet_s_416_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_s_416_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_s_416_coco_lcnet.yml) |
| PicoDet-M |  320*320   |          34.4           |        50.0        |        -        |       -       |              -              |            -             | [model](https://paddledet.bj.bcebos.com/models/picodet_m_320_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_m_320_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_m_320_coco_lcnet.yml) |
| PicoDet-M |  416*416   |          37.5           |        53.4       |        -        |       -        |              -              |            -            | [model](https://paddledet.bj.bcebos.com/models/picodet_m_416_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_m_416_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_m_416_coco_lcnet.yml) |
| PicoDet-L |  320*320   |          36.1           |        52.0        |        -       |       -        |              -             |            -           | [model](https://paddledet.bj.bcebos.com/models/picodet_l_320_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_l_320_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_l_320_coco_lcnet.yml) |
| PicoDet-L |  416*416   |          39.4           |        55.7        |        -        |       -       |              -              |            -            | [model](https://paddledet.bj.bcebos.com/models/picodet_l_416_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_l_416_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_l_416_coco_lcnet.yml) |
| PicoDet-L |  640*640   |          42.3           |        59.2        |        -        |       -        |              -              |            -           | [model](https://paddledet.bj.bcebos.com/models/picodet_l_640_coco_lcnet.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_l_640_coco_lcnet.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/picodet_l_640_coco_lcnet.yml) |


<details open>
<summary><b>注意事项:</b></summary>

- <a name="latency">时延测试：</a> 我们所有的模型都在`骁龙865(4xA77+4xA55)` 上测试(4线程，FP16预测)。上面表格中标有`NCNN`的是使用[NCNN](https://github.com/Tencent/ncnn)库测试，标有`Lite`的是使用[Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)进行测试。 测试的benchmark脚本来自: [MobileDetBenchmark](https://github.com/JiweiMaster/MobileDetBenchmark)。
- PicoDet在COCO train2017上训练，并且在COCO val2017上进行验证。
- PicoDet使用4卡GPU训练(PicoDet-L-640使用8卡训练)，并且所有的模型都是通过发布的默认配置训练得到。

</details>

#### 其他模型的基线

| 模型     | 输入尺寸 | mAP<sup>val<br>0.5:0.95 | mAP<sup>val<br>0.5 | 参数量<br><sup>(M) | FLOPS<br><sup>(G) | 预测时延<sup><small>[NCNN](#latency)</small><sup><br><sup>(ms) |
| :-------- | :--------: | :---------------------: | :----------------: | :----------------: | :---------------: | :-----------------------------: |
| YOLOv3-Tiny |  416*416   |          16.6           |        33.1      |        8.86        |       5.62        |             25.42               |
| YOLOv4-Tiny |  416*416   |          21.7           |        40.2        |        6.06           |       6.96           |             23.69               |
| PP-YOLO-Tiny |  320*320       |          20.6         |        -              |   1.08             |    0.58             |    6.75                           |  
| PP-YOLO-Tiny |  416*416   |          22.7          |    -               |    1.08               |    1.02             |    10.48                          |  
| Nanodet-M |  320*320      |          20.6            |    -               |    0.95               |    0.72             |    8.71                           |  
| Nanodet-M |  416*416   |          23.5             |    -               |    0.95               |    1.2              |  13.35                          |
| Nanodet-M 1.5x |  416*416   |          26.8        |    -                  | 2.08               |    2.42             |    15.83                          |
| YOLOX-Nano     |  416*416   |          25.8          |    -               |    0.91               |    1.08             |    19.23                          |
| YOLOX-Tiny     |  416*416   |          32.8          |    -               |    5.06               |    6.45             |    32.77                          |
| YOLOv5n |  640*640       |          28.4             |    46.0            |    1.9                |    4.5              |    40.35                          |
| YOLOv5s |  640*640       |          37.2             |    56.0            |    7.2                |    16.5             |    78.05                          |


## 快速开始

<details open>
<summary>依赖包:</summary>

- PaddlePaddle >= 2.2.1

</details>

<details>
<summary>安装</summary>

- [安装指导文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/INSTALL.md)
- [准备数据文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/PrepareDataSet_en.md)

</details>

<details>
<summary>训练&评估</summary>

- 单卡GPU上训练:

```shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/picodet/picodet_s_320_coco_lcnet.yml --eval
```

- 多卡GPU上训练:


```shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/picodet/picodet_s_320_coco_lcnet.yml --eval
```

- 评估:

```shell
python tools/eval.py -c configs/picodet/picodet_s_320_coco_lcnet.yml \
              -o weights=https://paddledet.bj.bcebos.com/models/picodet_s_320_coco_lcnet.pdparams
```

- 测试:

```shell
python tools/infer.py -c configs/picodet/picodet_s_320_coco_lcnet.yml \
              -o weights=https://paddledet.bj.bcebos.com/models/picodet_s_320_coco_lcnet.pdparams
```

详情请参考[快速开始文档](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/GETTING_STARTED.md).

</details>


## 部署

### 导出及转换模型

<details>
<summary>1. 导出模型 (点击展开)</summary>

```shell
cd PaddleDetection
python tools/export_model.py -c configs/picodet/picodet_s_320_coco_lcnet.yml \
              -o weights=https://paddledet.bj.bcebos.com/models/picodet_s_320_coco_lcnet.pdparams \
              --output_dir=inference_model
```

</details>

<details>
<summary>2. 转换模型至Paddle Lite (点击展开)</summary>

- 安装Paddlelite>=2.10:

```shell
pip install paddlelite
```

- 转换模型至Paddle Lite格式：

```shell
# FP32
paddle_lite_opt --model_dir=inference_model/picodet_s_320_coco_lcnet --valid_targets=arm --optimize_out=picodet_s_320_coco_fp32
# FP16
paddle_lite_opt --model_dir=inference_model/picodet_s_320_coco_lcnet --valid_targets=arm --optimize_out=picodet_s_320_coco_fp16 --enable_fp16=true
```

</details>

<details>
<summary>3. 转换模型至ONNX (点击展开)</summary>

- 安装[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) >= 0.7 并且 ONNX > 1.10.1, 细节请参考[导出ONNX模型教程](../../deploy/EXPORT_ONNX_MODEL.md)

```shell
pip install onnx
pip install paddle2onnx
```

- 转换模型:

```shell
paddle2onnx --model_dir output_inference/picodet_s_320_coco_lcnet/ \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 11 \
            --save_file picodet_s_320_coco.onnx
```

- 简化ONNX模型: 使用`onnx-simplifier`库来简化ONNX模型。

  - 安装 onnx-simplifier >= 0.3.6:
  ```shell
  pip install onnx-simplifier
  ```
  - 简化ONNX模型:
  ```shell
  python -m onnxsim picodet_s_320_coco.onnx picodet_s_processed.onnx
  ```

</details>

- 部署用的模型

| 模型     | 输入尺寸 | ONNX  | Paddle Lite(fp32) | Paddle Lite(fp16) |
| :-------- | :--------: | :---------------------: | :----------------: | :----------------: |
| PicoDet-S |  320*320   | [model](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_s_320_coco.onnx) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_s_320.tar) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_s_320_fp16.tar) |
| PicoDet-S |  416*416   |  [model](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_s_416_coco.onnx) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_s_416.tar) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_s_416_fp16.tar) |
| PicoDet-M |  320*320   | [model](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_m_320_coco.onnx) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_m_320.tar) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_m_320_fp16.tar) |
| PicoDet-M |  416*416   | [model](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_m_416_coco.onnx) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_m_416.tar) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_m_416_fp16.tar) |
| PicoDet-L |  320*320   | [model](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_l_320_coco.onnx) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_l_320.tar) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_l_320_fp16.tar) |
| PicoDet-L |  416*416   | [model](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_l_416_coco.onnx) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_l_416.tar) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_l_416_fp16.tar) |
| PicoDet-L |  640*640   | [model](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_l_640_coco.onnx) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_l_640.tar) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_l_640_fp16.tar) |
| PicoDet-Shufflenetv2 1x      |  416*416   | [model](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_shufflenetv2_1x_416_coco.onnx) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_shufflenetv2_1x.tar) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_shufflenetv2_1x_fp16.tar) |
| PicoDet-MobileNetv3-large 1x |  416*416   | [model](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_mobilenetv3_large_1x_416_coco.onnx) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_mobilenetv3_large_1x.tar) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_mobilenetv3_large_1x_fp16.tar) |
| PicoDet-LCNet 1.5x           |  416*416   | [model](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_lcnet_1_5x_416_coco.onnx) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_lcnet_1_5x.tar) | [model](https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_lcnet_1_5x_fp16.tar) |


### 部署

- PaddleInference demo [Python](../../deploy/python) & [C++](../../deploy/cpp)
- [PaddleLite C++ demo](../../deploy/lite)
- [NCNN C++/Python demo](../../deploy/third_engine/demo_ncnn)
- [MNN C++/Python demo](../../deploy/third_engine/demo_mnn)
- [OpenVINO C++ demo](../../deploy/third_engine/demo_openvino)
- [Android demo(Paddle Lite)](https://github.com/PaddlePaddle/Paddle-Lite-Demo/tree/develop/object_detection/android/app/cxx/picodet_detection_demo)


Android demo可视化：
<div align="center">
  <img src="../../docs/images/picodet_android_demo1.jpg" height="500px" ><img src="../../docs/images/picodet_android_demo2.jpg" height="500px" ><img src="../../docs/images/picodet_android_demo3.jpg" height="500px" ><img src="../../docs/images/picodet_android_demo4.jpg" height="500px" >
</div>


## 量化

<details open>
<summary>依赖包:</summary>

- PaddlePaddle >= 2.2.2
- PaddleSlim >= 2.2.1

**安装:**

```shell
pip install paddleslim==2.2.1
```

</details>

<details>
<summary>量化训练 (点击展开)</summary>

开始量化训练:

```shell
python tools/train.py -c configs/picodet/picodet_s_320_coco_lcnet.yml \
          --slim_config configs/slim/quant/picodet_s_quant.yml --eval
```

- 更多细节请参考[slim文档](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/slim)

</details>

<details>
<summary>离线量化 (点击展开)</summary>

校准及导出量化模型:

```shell
python tools/post_quant.py -c configs/picodet/picodet_s_320_coco_lcnet.yml \
          --slim_config configs/slim/post_quant/picodet_s_ptq.yml
```

- 注意: 离线量化模型精度问题正在解决中.

</details>

## 非结构化剪枝

<details open>
<summary>教程:</summary>

训练及部署细节请参考[非结构化剪枝文档](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet/pruner/README.md)。

</details>

## 应用

- **行人检测：** `PicoDet-S-Pedestrian`行人检测模型请参考[PP-TinyPose](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/keypoint/tiny_pose#%E8%A1%8C%E4%BA%BA%E6%A3%80%E6%B5%8B%E6%A8%A1%E5%9E%8B)

- **主体检测：** `PicoDet-L-Mainbody`主体检测模型请参考[主体检测文档](./application/mainbody_detection/README.md)

## FAQ

<details>
<summary>显存爆炸(Out of memory error)</summary>

请减小配置文件中`TrainReader`的`batch_size`。

</details>

<details>
<summary>如何迁移学习</summary>

请重新设置配置文件中的`pretrain_weights`字段，比如利用COCO上训好的模型在自己的数据上继续训练：
```yaml
pretrain_weights: https://paddledet.bj.bcebos.com/models/picodet_l_640_coco_lcnet.pdparams
```

</details>

<details>
<summary>`transpose`算子在某些硬件上耗时验证</summary>

请使用`PicoDet-LCNet`模型，`transpose`较少。

</details>


<details>
<summary>如何计算模型参数量。</summary>

可以将以下代码插入：[trainer.py](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/engine/trainer.py#L141) 来计算参数量。

```python
params = sum([
    p.numel() for n, p in self.model. named_parameters()
    if all([x not in n for x in ['_mean', '_variance']])
]) # exclude BatchNorm running status
print('params: ', params)
```

</details>

## 引用PP-PicoDet
如果需要在你的研究中使用PP-PicoDet，请通过一下方式引用我们的技术报告：
```
@misc{yu2021pppicodet,
      title={PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices},
      author={Guanghua Yu and Qinyao Chang and Wenyu Lv and Chang Xu and Cheng Cui and Wei Ji and Qingqing Dang and Kaipeng Deng and Guanzhong Wang and Yuning Du and Baohua Lai and Qiwen Liu and Xiaoguang Hu and Dianhai Yu and Yanjun Ma},
      year={2021},
      eprint={2111.00902},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
