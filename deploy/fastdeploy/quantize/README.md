[English](README.md) | 简体中文  

# PaddleDetection 量化模型部署-FastDeploy

FastDeploy已支持部署量化模型,并提供一键模型自动化压缩的工具.
用户可以使用一键模型自动化压缩工具,自行对模型量化后部署, 也可以直接下载FastDeploy提供的量化模型进行部署.

## 1. FastDeploy一键模型自动化压缩工具  

FastDeploy 提供了一键模型自动化压缩工具, 能够简单地通过输入一个配置文件, 对模型进行量化.
详细教程请见: [一键模型自动化压缩工具](https://github.com/PaddlePaddle/FastDeploy/tree/develop/tools/common_tools/auto_compression)。**注意**: 推理量化后的分类模型仍然需要FP32模型文件夹下的deploy.yaml文件, 自行量化的模型文件夹内不包含此yaml文件, 用户从FP32模型文件夹下复制此yaml文件到量化后的模型文件夹内即可。

## 2. 量化完成的PaddleDetection模型  

用户也可以直接下载下表中的量化模型进行部署.(点击模型名字即可下载)

Benchmark表格说明:
- Runtime时延为模型在各种Runtime上的推理时延,包含CPU->GPU数据拷贝,GPU推理,GPU->CPU数据拷贝时间. 不包含模型各自的前后处理时间.
- 端到端时延为模型在实际推理场景中的时延, 包含模型的前后处理.
- 所测时延均为推理1000次后求得的平均值, 单位是毫秒.
- INT8 + FP16 为在推理INT8量化模型的同时, 给Runtime 开启FP16推理选项
- INT8 + FP16 + PM, 为在推理INT8量化模型和开启FP16的同时, 开启使用Pinned Memory的选项,可加速GPU->CPU数据拷贝的速度
- 最大加速比, 为FP32时延除以INT8推理的最快时延,得到最大加速比.
- 策略为量化蒸馏训练时, 采用少量无标签数据集训练得到量化模型, 并在全量验证集上验证精度, INT8精度并不代表最高的INT8精度.
- CPU为Intel(R) Xeon(R) Gold 6271C, 所有测试中固定CPU线程数为1.  GPU为Tesla T4, TensorRT版本8.4.15.


- Runtime Benchmark
| 模型                 |推理后端            |部署硬件    | FP32 Runtime时延   | INT8 Runtime时延 | INT8 + FP16 Runtime时延  | INT8+FP16+PM Runtime时延  | 最大加速比    | FP32 mAP | INT8 mAP | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ppyoloe_crn_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_qat.tar )  | TensorRT         |    GPU    |  27.90 | 6.39 |6.44|5.95    |      4.67       | 51.4  | 50.7 | 量化蒸馏训练 |
| [ppyoloe_crn_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_qat.tar )  | Paddle-TensorRT |    GPU    |  30.89     |None  |  13.78 |14.01    |      2.24       | 51.4  | 50.5 | 量化蒸馏训练 |
| [ppyoloe_crn_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_qat.tar)  | ONNX Runtime |    CPU    |     1057.82 |   449.52 |None|None    |      2.35        |51.4 | 50.0 |量化蒸馏训练 |

NOTE:
- TensorRT比Paddle-TensorRT快的原因是在runtime移除了multiclass_nms3算子

- 端到端 Benchmark
| 模型                 |推理后端            |部署硬件    | FP32 End2End时延   | INT8 End2End时延 | INT8 + FP16 End2End时延  | INT8+FP16+PM End2End时延  | 最大加速比    | FP32 mAP | INT8 mAP | 量化方式   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ppyoloe_crn_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_qat.tar )  | TensorRT         |    GPU    |  35.75 | 15.42 |20.70|20.85  |      2.32      | 51.4  | 50.7 | 量化蒸馏训练 |
| [ppyoloe_crn_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_qat.tar )  | Paddle-TensorRT |    GPU    | 33.48    |None  |  18.47 |18.03   |     1.81       | 51.4  | 50.5 | 量化蒸馏训练 |
| [ppyoloe_crn_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_qat.tar)  | ONNX Runtime |    CPU    |     1067.17 |   461.037 |None|None    |      2.31        |51.4 | 50.0 |量化蒸馏训练 |


量化后模型的Benchmark比较，请参考[量化模型 Benchmark](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/quantize.md)

## 3. 支持部署量化模型的硬件  

FastDeploy 量化模型部署的过程大致都与FP32模型类似，只是模型量化与非量化的区别，如果硬件在量化模型部署过程有特殊处理，也会在文档中特别标明，因此量化模型部署可以参考如下硬件的链接

|硬件类型|该硬件是否支持|使用指南|Python|C++|
|:---:|:---:|:---:|:---:|:---:|
|X86 CPU|✅|[链接](cpu-gpu)|✅|✅|
|NVIDIA GPU|✅|[链接](cpu-gpu)|✅|✅|
|飞腾CPU|✅|[链接](cpu-gpu)|✅|✅|
|ARM CPU|✅|[链接](cpu-gpu)|✅|✅|
|Intel GPU(集成显卡)|✅|[链接](cpu-gpu)|✅|✅|  
|Intel GPU(独立显卡)|✅|[链接](cpu-gpu)|✅|✅|  
|昆仑|✅|[链接](kunlun)|✅|✅|
|昇腾|✅|[链接](ascend)|✅|✅|
|瑞芯微|✅|[链接](rockchip)|✅|✅|  
|晶晨|✅|[链接](amlogic)|--|✅|  
|算能|✅|[链接](sophgo)|✅|✅|  
