# Python端预测部署

在PaddlePaddle中预测引擎和训练引擎底层有着不同的优化方法, 预测引擎使用了AnalysisPredictor，专门针对推理进行了优化，是基于[C++预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/native_infer.html)的Python接口，该引擎可以对模型进行多项图优化，减少不必要的内存拷贝。如果用户在部署已训练模型的过程中对性能有较高的要求，我们提供了独立于PaddleDetection的预测脚本，方便用户直接集成部署。


主要包含两个步骤：

- 导出预测模型
- 基于Python进行预测

## 1. 导出预测模型

PaddleDetection在训练过程包括网络的前向和优化器相关参数，而在部署过程中，我们只需要前向参数，具体参考:[导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/deploy/EXPORT_MODEL.md)

导出后目录下，包括`infer_cfg.yml`, `model.pdiparams`,  `model.pdiparams.info`, `model.pdmodel`四个文件。

## 2. 基于Python的预测



在终端输入以下命令进行预测：

```bash
python deploy/python/infer.py --model_dir=./output_inference/yolov3_mobilenet_v1_roadsign --image_file=./demo/road554.png --device=GPU
```

参数说明如下:

| 参数 | 是否必须|含义 |
|-------|-------|----------|
| --model_dir | Yes| 上述导出的模型路径 |
| --image_file | Option | 需要预测的图片 |
| --image_dir  | Option |  要预测的图片文件夹路径   |
| --video_file | Option | 需要预测的视频 |
| --camera_id | Option | 用来预测的摄像头ID，默认为-1(表示不使用摄像头预测，可设置为：0 - (摄像头数目-1) )，预测过程中在可视化界面按`q`退出输出预测结果到：output/output.mp4|
| --device | Option | 运行时的设备，可选择`CPU/GPU/XPU`，默认为`CPU`|
| --run_mode | Option |使用GPU时，默认为fluid, 可选（fluid/trt_fp32/trt_fp16/trt_int8）|
| --batch_size | Option |预测时的batch size，在指定`image_dir`时有效，默认为1 |
| --threshold | Option|预测得分的阈值，默认为0.5|
| --output_dir | Option|可视化结果保存的根目录，默认为output/|
| --run_benchmark | Option| 是否运行benchmark，同时需指定`--image_file`或`--image_dir`，默认为False |
| --enable_mkldnn | Option | CPU预测中是否开启MKLDNN加速，默认为False |
| --cpu_threads | Option| 设置cpu线程数，默认为1 |
| --trt_calib_mode | Option| TensorRT是否使用校准功能，默认为False。使用TensorRT的int8功能时，需设置为True，使用PaddleSlim量化后的模型时需要设置为False |

说明：

- 参数优先级顺序：`camera_id` > `video_file` > `image_dir` > `image_file`。
- run_mode：fluid代表使用AnalysisPredictor，精度float32来推理，其他参数指用AnalysisPredictor，TensorRT不同精度来推理。
- 如果安装的PaddlePaddle不支持基于TensorRT进行预测，需要自行编译，详细可参考[预测库编译教程](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html)。
- --run_benchmark如果设置为True，则需要安装依赖`pip install pynvml psutil GPUtil`。
