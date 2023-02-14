简体中文 | [English](README_en.md)

# PP-YOLOE-R

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [使用说明](#使用说明)
- [预测部署](#预测部署)
- [附录](#附录)
- [引用](#引用)

## 简介
PP-YOLOE-R是一个高效的单阶段Anchor-free旋转框检测模型。基于PP-YOLOE, PP-YOLOE-R以极少的参数量和计算量为代价，引入了一系列有用的设计来提升检测精度。在DOTA 1.0数据集上，PP-YOLOE-R-l和PP-YOLOE-R-x在单尺度训练和测试的情况下分别达到了78.14和78.27 mAP，这超越了几乎所有的旋转框检测模型。通过多尺度训练和测试，PP-YOLOE-R-l和PP-YOLOE-R-x的检测精度进一步提升至80.02和80.73 mAP。在这种情况下，PP-YOLOE-R-x超越了所有的anchor-free方法并且和最先进的anchor-based的两阶段模型精度几乎相当。此外，PP-YOLOE-R-s和PP-YOLOE-R-m通过多尺度训练和测试可以达到79.42和79.71 mAP。考虑到这两个模型的参数量和计算量，其性能也非常卓越。在保持高精度的同时，PP-YOLOE-R避免使用特殊的算子，例如Deformable Convolution或Rotated RoI Align，以使其能轻松地部署在多种多样的硬件上。在1024x1024的输入分辨率下，PP-YOLOE-R-s/m/l/x在RTX 2080 Ti上使用TensorRT FP16分别能达到69.8/55.1/48.3/37.1 FPS，在Tesla V100上分别能达到114.5/86.8/69.7/50.7 FPS。更多细节可以参考我们的[**技术报告**](https://arxiv.org/abs/2211.02386)。

<div align="center">
  <img src="../../../docs/images/ppyoloe_r_map_fps.png" width=500 />
</div>

PP-YOLOE-R相较于PP-YOLOE做了以下几点改动：
- Rotated Task Alignment Learning
- 解耦的角度预测头
- 使用DFL进行角度预测
- 可学习的门控单元
- [ProbIoU损失函数](https://arxiv.org/abs/2106.06072)

## 模型库

| 模型 | Backbone | mAP | V100 TRT FP16 (FPS) | RTX 2080 Ti TRT FP16 (FPS) | Params (M) | FLOPs (G) | 学习率策略 | 角度表示 | 数据增广 | GPU数目 | 每GPU图片数目 | 模型下载 | 配置文件 |
|:---:|:--------:|:----:|:--------------------:|:------------------------:|:----------:|:---------:|:--------:|:----------:|:-------:|:------:|:-----------:|:--------:|:------:|
| PP-YOLOE-R-s | CRN-s | 73.82 | 114.5 | 69.8 | 8.09 | 43.46 | 3x | oc | RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_s_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_s_3x_dota.yml) |
| PP-YOLOE-R-s | CRN-s | 79.42 | 114.5 | 69.8 | 8.09 | 43.46 | 3x | oc | MS+RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_s_3x_dota_ms.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_s_3x_dota_ms.yml) |
| PP-YOLOE-R-m | CRN-m | 77.64 | 86.8  | 55.1 | 23.96 |127.00 | 3x | oc | RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_m_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_m_3x_dota.yml) |
| PP-YOLOE-R-m | CRN-m | 79.71 | 86.8  | 55.1 | 23.96 |127.00 | 3x | oc | MS+RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_m_3x_dota_ms.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_m_3x_dota_ms.yml) |
| PP-YOLOE-R-l | CRN-l | 78.14 | 69.7  | 48.3 | 53.29 |281.65 | 3x | oc | RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota.yml) |
| PP-YOLOE-R-l | CRN-l | 80.02 | 69.7  | 48.3 | 53.29 |281.65 | 3x | oc | MS+RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota_ms.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota_ms.yml) |
| PP-YOLOE-R-x | CRN-x | 78.28 | 50.7  | 37.1 | 100.27|529.82 | 3x | oc | RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_x_3x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_x_3x_dota.yml) |
| PP-YOLOE-R-x | CRN-x | 80.73 | 50.7  | 37.1 | 100.27|529.82 | 3x | oc | MS+RR | 4 | 2 | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_x_3x_dota_ms.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/rotate/ppyoloe_r/ppyoloe_r_crn_x_3x_dota_ms.yml) |

**注意:**

- 如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- 模型库中的模型默认使用单尺度训练单尺度测试。如果数据增广一栏标明MS，意味着使用多尺度训练和多尺度测试。如果数据增广一栏标明RR，意味着使用RandomRotate数据增广进行训练。
- CRN表示在PP-YOLOE中提出的CSPRepResNet
- PP-YOLOE-R的参数量和计算量是在重参数化之后计算得到，输入图像的分辨率为1024x1024
- 速度测试使用TensorRT 8.2.3在DOTA测试集中测试2000张图片计算平均值得到。参考速度测试以复现[速度测试](#速度测试)

## 使用说明

参考[数据准备](../README.md#数据准备)准备数据。

### 训练

GPU单卡训练
``` bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota.yml
```

GPU多卡训练
``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota.yml
```

### 预测

执行以下命令预测单张图片，图片预测结果会默认保存在`output`文件夹下面
``` bash
python tools/infer.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams --infer_img=demo/P0861__1.0__1154___824.png --draw_threshold=0.5
```

### DOTA数据集评估

参考[DOTA Task](https://captain-whu.github.io/DOTA/tasks.html), 评估DOTA数据集需要生成一个包含所有检测结果的zip文件，每一类的检测结果储存在一个txt文件中，txt文件中每行格式为：`image_name score x1 y1 x2 y2 x3 y3 x4 y4`。将生成的zip文件提交到[DOTA Evaluation](https://captain-whu.github.io/DOTA/evaluation.html)的Task1进行评估。你可以执行以下命令得到test数据集的预测结果：
``` bash
python tools/infer.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams --infer_dir=/path/to/test/images --output_dir=output_ppyoloe_r --visualize=False --save_results=True
```
将预测结果处理成官网评估所需要的格式：
``` bash
python configs/rotate/tools/generate_result.py --pred_txt_dir=output_ppyoloe_r/ --output_dir=submit/ --data_type=dota10

zip -r submit.zip submit
```

### 速度测试
可以使用Paddle模式或者Paddle-TRT模式进行测速。当使用Paddle-TRT模式测速时，需要确保**TensorRT版本大于8.2, PaddlePaddle版本为develop版本**。使用Paddle-TRT进行测速，可以执行以下命令：

``` bash
# 导出模型
python tools/export_model.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams trt=True

# 速度测试
CUDA_VISIBLE_DEVICES=0 python configs/rotate/tools/inference_benchmark.py --model_dir output_inference/ppyoloe_r_crn_l_3x_dota/ --image_dir /path/to/dota/test/dir --run_mode trt_fp16
```
当只使用Paddle进行测速，可以执行以下命令：
``` bash
# 导出模型
python tools/export_model.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams

# 速度测试
CUDA_VISIBLE_DEVICES=0 python configs/rotate/tools/inference_benchmark.py --model_dir output_inference/ppyoloe_r_crn_l_3x_dota/ --image_dir /path/to/dota/test/dir --run_mode paddle
```

## 预测部署

**使用Paddle**进行部署，执行以下命令：
``` bash
# 导出模型
python tools/export_model.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams

# 预测图片
python deploy/python/infer.py --image_file demo/P0072__1.0__0___0.png --model_dir=output_inference/ppyoloe_r_crn_l_3x_dota --run_mode=paddle --device=gpu
```

**使用Paddle-TRT进行部署**，执行以下命令：
```
# 导出模型
python tools/export_model.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams trt=True

# 预测图片
python deploy/python/infer.py --image_file demo/P0072__1.0__0___0.png --model_dir=output_inference/ppyoloe_r_crn_l_3x_dota --run_mode=trt_fp16 --device=gpu
```

**注意：**
- 使用Paddle-TRT使用确保**PaddlePaddle版本为develop版本且TensorRT版本大于8.2**.

**使用ONNX Runtime进行部署**，执行以下命令：
```
# 导出模型
python tools/export_model.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota.pdparams export_onnx=True

# 安装paddle2onnx
pip install paddle2onnx

# 转换成onnx模型
paddle2onnx --model_dir output_inference/ppyoloe_r_crn_l_3x_dota --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 11 --save_file ppyoloe_r_crn_l_3x_dota.onnx

# 预测图片
python configs/rotate/tools/onnx_infer.py --infer_cfg output_inference/ppyoloe_r_crn_l_3x_dota/infer_cfg.yml --onnx_file ppyoloe_r_crn_l_3x_dota.onnx --image_file demo/P0072__1.0__0___0.png

```

## 附录

PP-YOLOE-R消融实验

| 模型 | mAP | 参数量(M) | FLOPs(G) |
| :-: | :-: | :------: | :------: |
| Baseline | 75.61 | 50.65 | 269.09 |
| +Rotated Task Alignment Learning | 77.24 | 50.65 | 269.09 |
| +Decoupled Angle Prediction Head | 77.78 | 52.20 | 272.72 |
| +Angle Prediction with DFL | 78.01 | 53.29 | 281.65 |
| +Learnable Gating Unit for RepVGG | 78.14 | 53.29 | 281.65 |


## 引用

```
@article{wang2022pp,
  title={PP-YOLOE-R: An Efficient Anchor-Free Rotated Object Detector},
  author={Wang, Xinxin and Wang, Guanzhong and Dang, Qingqing and Liu, Yi and Hu, Xiaoguang and Yu, Dianhai},
  journal={arXiv preprint arXiv:2211.02386},
  year={2022}
}

@article{xu2022pp,
  title={PP-YOLOE: An evolved version of YOLO},
  author={Xu, Shangliang and Wang, Xinxin and Lv, Wenyu and Chang, Qinyao and Cui, Cheng and Deng, Kaipeng and Wang, Guanzhong and Dang, Qingqing and Wei, Shengyu and Du, Yuning and others},
  journal={arXiv preprint arXiv:2203.16250},
  year={2022}
}

@article{llerena2021gaussian,
  title={Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection},
  author={Llerena, Jeffri M and Zeni, Luis Felipe and Kristen, Lucas N and Jung, Claudio},
  journal={arXiv preprint arXiv:2106.06072},
  year={2021}
}
```
