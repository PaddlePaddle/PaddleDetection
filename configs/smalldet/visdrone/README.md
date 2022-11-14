# VisDrone-DET 小目标检测模型

PaddleDetection团队提供了针对VisDrone-DET小目标数航拍场景的基于PP-YOLOE的检测模型，用户可以下载模型进行使用。整理后的COCO格式VisDrone-DET数据集[下载链接](https://bj.bcebos.com/v1/paddledet/data/smalldet/visdrone.zip)，检测其中的10类，包括 `pedestrian(1), people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10)`，原始数据集[下载链接](https://github.com/VisDrone/VisDrone-Dataset)。

**注意:**
- VisDrone-DET数据集包括**train集6471张，val集548张，test_dev集1610张**，test-challenge集1580张(未开放检测框标注)，前三者均有开放检测框标注。
- 模型均**只使用train集训练**，在val集和test_dev集上分别验证精度，test_dev集图片数较多，精度参考性较高。


## 原图训练，原图评估：

|    模型   | COCOAPI mAP<sup>val<br>0.5:0.95 | COCOAPI mAP<sup>val<br>0.5 | COCOAPI mAP<sup>test_dev<br>0.5:0.95 | COCOAPI mAP<sup>test_dev<br>0.5 | MatlabAPI mAP<sup>test_dev<br>0.5:0.95 | MatlabAPI mAP<sup>test_dev<br>0.5 | 下载  | 配置文件 |
|:---------|:------:|:------:| :----: | :------:| :------: | :------:| :----: | :------:|
|PP-YOLOE-s|  23.5  |  39.9  |  19.4  |  33.6   |  23.68   |  40.66  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_80e_visdrone.pdparams) | [配置文件](./ppyoloe_crn_s_80e_visdrone.yml) |
|PP-YOLOE-P2-Alpha-s|    24.4  |  41.6  |  20.1  |  34.7  |  24.55   |  42.19  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_p2_alpha_80e_visdrone.pdparams) | [配置文件](./ppyoloe_crn_s_p2_alpha_80e_visdrone.yml) |
|PP_YOLOE_plus_sod_s|  25.1  |  42.8  |  20.7  |  36.2   |  25.16  |  43.86   | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_sod_crn_s_80e_visdrone.pdparams) | [配置文件](./ppyoloe_plus_sod_crn_s_80e_visdrone.yml) |
|PP-YOLOE-l|  29.2  |  47.3  |  23.5  |  39.1   |  28.00   |  46.20  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_visdrone.pdparams) | [配置文件](./ppyoloe_crn_l_80e_visdrone.yml) |
|PP-YOLOE-P2-Alpha-l|  30.1  |  48.9  |  24.3  |  40.8   |  28.47   |  48.16  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_p2_alpha_80e_visdrone.pdparams) | [配置文件](./ppyoloe_crn_l_p2_alpha_80e_visdrone.yml) |
|PP_YOLOE_plus_sod_l|  31.9  |  52.1  |  25.6  |  43.5   |  30.25  |  51.18   | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_sod_crn_l_80e_visdrone.pdparams) | [配置文件](./ppyoloe_plus_sod_crn_l_80e_visdrone.yml) |
|PP-YOLOE-Alpha-largesize-l|  41.9  |  65.0 |  32.3  |  53.0   |  37.13   |  61.15  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_alpha_largesize_80e_visdrone.pdparams) | [配置文件](./ppyoloe_crn_l_alpha_largesize_80e_visdrone.yml) |
|PP-YOLOE-P2-Alpha-largesize-l|  41.3  |  64.5  |  32.4  |  53.1   |  37.49   |  51.54  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_p2_alpha_largesize_80e_visdrone.pdparams) | [配置文件](./ppyoloe_crn_l_p2_alpha_largesize_80e_visdrone.yml) |
|PP-YOLOE-plus-largesize-l |  43.3  |  66.7 |  33.5  |  54.7   |  38.24   |  62.76  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_largesize_80e_visdrone.pdparams) | [配置文件](./ppyoloe_plus_crn_l_largesize_80e_visdrone.yml) |
|PP-YOLOE-plus_sod-largesize_l |  42.7  |  65.9 |  33.6  |  55.1   |  38.4   |  63.07  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone.pdparams) | [配置文件](./ppyoloe_plus_sod_crn_l_largesize_80e_visdrone.yml) |

**注意:**
  - 上表中的模型均为**使用原图训练**，也使用**原图评估预测**。
  - **sod**表示使用**基于向量的DFL算法**和针对小目标的**中心先验优化策略**，并**在模型的Neck结构中加入transformer**。
  - **P2**表示增加P2层(1/4下采样层)的特征，共输出4个PPYOLOEHead。
  - **Alpha**表示对CSPResNet骨干网络增加可一个学习权重参数Alpha参与训练。
  - **largesize**表示使用**以1600尺度为基础的多尺度训练**和**1920尺度预测**，相应的训练batch_size也减小，以速度来换取高精度。


## 子图训练，原图评估和拼图评估：

|    模型   |       数据集     |  SLICE_SIZE  |  OVERLAP_RATIO  | 类别数  | mAP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | 下载链接  | 配置文件 |
|:---------|:---------------:|:---------------:|:---------------:|:------:|:-----------------------:|:-------------------:|:---------:| :-----: |
|PP-YOLOE-l(原图评估)| VisDrone-DET|  640 | 0.25 | 10 |  29.7 |  48.5 | [下载链接](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams) | [配置文件](../ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml) |
|PP-YOLOE-l (拼图评估)| VisDrone-DET|  640 | 0.25 | 10 | 37.2 | 59.4 | [下载链接](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams) | [配置文件](../ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml) |

**注意:**
  - 上表中的模型均为使用**切图后的子图**训练，评估预测时分为两种，使用原图评估预测，和使用子图拼图评估预测。
  - **SLICE_SIZE**表示使用SAHI工具切图后子图的边长大小，**OVERLAP_RATIO**表示切图的子图之间的重叠率。


## 注意事项：
  - PP-YOLOE模型训练过程中使用8 GPUs进行混合精度训练，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
  - 具体使用教程请参考[ppyoloe](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/ppyoloe#getting-start)。
  - MatlabAPI测试是使用官网评测工具[VisDrone2018-DET-toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit)。
  - 切图训练模型的配置文件及训练相关流程请参照[README](../README.cn)。


## 部署模型

| 网络模型     | 输入尺寸 | 导出后的权重(w/o NMS) | ONNX(w/o NMS)  |
| :-------- | :--------: | :---------------------: | :----------------: |
| PP_YOLOE_plus_sod_s |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/smalldet/ppyoloe_plus_sod_crn_s_80e_visdrone_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/smalldet/ppyoloe_plus_sod_crn_s_80e_visdrone_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/smalldet/ppyoloe_plus_sod_crn_s_80e_visdrone_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/smalldet/ppyoloe_plus_sod_crn_s_80e_visdrone_wo_nms.onnx) |
| PP_YOLOE_plus_sod_l |  640   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/smalldet/ppyoloe_plus_sod_crn_l_80e_visdrone_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/smalldet/ppyoloe_plus_sod_crn_l_80e_visdrone_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/smalldet/ppyoloe_plus_sod_crn_l_80e_visdrone_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/smalldet/ppyoloe_plus_sod_crn_l_80e_visdrone_wo_nms.onnx) |
| PP-YOLOE-plus_sod-largesize_l     |  1920   | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/smalldet/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone_w_nms.zip) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/smalldet/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone_wo_nms.zip) | [( w/ nms)](https://paddledet.bj.bcebos.com/deploy/smalldet/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone_w_nms.onnx) &#124; [( w/o nms)](https://paddledet.bj.bcebos.com/deploy/smalldet/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone_wo_nms.onnx) |


## 测速

1.参考[Paddle Inference文档](https://www.paddlepaddle.org.cn/inference/master/user_guides/download_lib.html#python)，下载并安装与你的CUDA, CUDNN和TensorRT相应的wheel包。
测速需要设置`--run_benchmark=True`, 你需要安装以下依赖`pip install pynvml psutil GPUtil`。
导出ONNX，你需要安装以下依赖`pip install paddle2onnx`。

2.运行以下命令导出**带NMS的模型和ONNX**，并使用TensorRT FP16进行推理和测速

### 注意：

- 由于NMS参数设置对速度影响极大，部署测速时可调整`keep_top_k`和`nms_top_k`，在只低约0.1 mAP精度的情况下加快预测速度，导出模型的时候也可这样设置：
  ```
  nms:
    name: MultiClassNMS
    nms_top_k: 1000 # 10000
    keep_top_k: 100 # 500
    score_threshold: 0.01
    nms_threshold: 0.6
  ```

```bash
# 导出带NMS的模型
python tools/export_model.py -c configs/smalldet/visdrone/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone.pdparams trt=True

# 导出带NMS的ONNX
paddle2onnx --model_dir output_inference/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ppyoloe_plus_sod_crn_l_largesize_80e_visdrone.onnx

# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone --image_file=demo/0000315_01601_d_0000509.jpg --device=gpu --run_mode=trt_fp16

# 推理文件夹下的所有图片
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone --image_dir=demo/ --device=gpu --run_mode=trt_fp16

# 单张图片普通测速
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone --image_file=demo/0000315_01601_d_0000509.jpg --device=gpu --run_benchmark=True

# 单张图片TensorRT FP16测速
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone --image_file=demo/0000315_01601_d_0000509.jpg --device=gpu --run_benchmark=True --run_mode=trt_fp16
```

3.运行以下命令导出**不带NMS的模型和ONNX**，并使用TensorRT FP16进行推理和测速，以及**ONNX下FP16测速**

```bash
# 导出带NMS的模型
python tools/export_model.py -c configs/smalldet/visdrone/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone.pdparams trt=True exclude_nms=True

# 导出带NMS的ONNX
paddle2onnx --model_dir output_inference/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ppyoloe_plus_sod_crn_l_largesize_80e_visdrone.onnx

# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone --image_file=demo/0000315_01601_d_0000509.jpg --device=gpu --run_mode=trt_fp16

# 推理文件夹下的所有图片
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone --image_dir=demo/ --device=gpu --run_mode=trt_fp16

# 单张图片普通测速
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone --image_file=demo/0000315_01601_d_0000509.jpg --device=gpu --run_benchmark=True

# 单张图片TensorRT FP16测速
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone --image_file=demo/0000315_01601_d_0000509.jpg --device=gpu --run_benchmark=True --run_mode=trt_fp16

# 单张图片ONNX TensorRT FP16测速
/usr/local/TensorRT-8.0.3.4/bin/trtexec --onnx=ppyoloe_plus_sod_crn_l_largesize_80e_visdrone.onnx --workspace=4096 --avgRuns=10 --shapes=input:1x3x1920x1920 --fp16
```

**注意：**
- TensorRT会根据网络的定义，执行针对当前硬件平台的优化，生成推理引擎并序列化为文件。该推理引擎只适用于当前软硬件平台。如果你的软硬件平台没有发生变化，你可以设置[enable_tensorrt_engine](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/deploy/python/infer.py#L857)的参数`use_static=True`，这样生成的序列化文件将会保存在`output_inference`文件夹下，下次执行TensorRT时将加载保存的序列化文件。
- PaddleDetection release/2.4及其之后的版本将支持NMS调用TensorRT，需要依赖PaddlePaddle release/2.3及其之后的版本


# 引用
```
@ARTICLE{9573394,
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Detection and Tracking Meet Drones Challenge},
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3119563}
}
```
