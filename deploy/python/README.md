# Python端预测部署

在PaddlePaddle中预测引擎和训练引擎底层有着不同的优化方法, 预测引擎使用了AnalysisPredictor，专门针对推理进行了优化，是基于[C++预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/native_infer.html)的Python接口，该引擎可以对模型进行多项图优化，减少不必要的内存拷贝。如果用户在部署已训练模型的过程中对性能有较高的要求，我们提供了独立于PaddleDetection的预测脚本，方便用户直接集成部署。


Python端预测部署主要包含两个步骤：
- 导出预测模型
- 基于Python进行预测

## 1. 导出预测模型

PaddleDetection在训练过程包括网络的前向和优化器相关参数，而在部署过程中，我们只需要前向参数，具体参考:[导出模型](../deploy/EXPORT_MODEL.md)，例如

```bash
# 导出YOLOv3检测模型
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --output_dir=./inference_model \
 -o weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams

# 导出HigherHRNet(bottom-up)关键点检测模型
python tools/export_model.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml -o weights=https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_512.pdparams

# 导出HRNet(top-down)关键点检测模型
python tools/export_model.py -c configs/keypoint/hrnet/hrnet_w32_384x288.yml -o weights=https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_384x288.pdparams

# 导出FairMOT多目标跟踪模型
python tools/export_model.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams

# 导出ByteTrack多目标跟踪模型(相当于只导出检测器)
python tools/export_model.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/ppyoloe_crn_l_36e_640x640_mot17half.pdparams
```

导出后目录下，包括`infer_cfg.yml`, `model.pdiparams`,  `model.pdiparams.info`, `model.pdmodel`四个文件。


## 2. 基于Python的预测

### 2.1 通用检测
在终端输入以下命令进行预测：
```bash
python deploy/python/infer.py --model_dir=./output_inference/yolov3_darknet53_270e_coco --image_file=./demo/000000014439.jpg --device=GPU
```

### 2.2 关键点检测
在终端输入以下命令进行预测：
```bash
# keypoint top-down(HRNet)/bottom-up(HigherHRNet)单独推理，该模式下top-down模型HRNet只支持单人截图预测
python deploy/python/keypoint_infer.py --model_dir=output_inference/hrnet_w32_384x288/ --image_file=./demo/hrnet_demo.jpg --device=GPU --threshold=0.5
python deploy/python/keypoint_infer.py --model_dir=output_inference/higherhrnet_hrnet_w32_512/ --image_file=./demo/000000014439_640x640.jpg --device=GPU --threshold=0.5

# detector 检测 + keypoint top-down模型联合部署（联合推理只支持top-down关键点模型）
python deploy/python/det_keypoint_unite_infer.py --det_model_dir=output_inference/yolov3_darknet53_270e_coco/ --keypoint_model_dir=output_inference/hrnet_w32_384x288/ --video_file={your video name}.mp4  --device=GPU
```
**注意:**
 - 关键点检测模型导出和预测具体可参照[keypoint](../../configs/keypoint/README.md)，可分别在各个模型的文档中查找具体用法；
 - 此目录下的关键点检测部署为基础前向功能，更多关键点检测功能可使用PP-Human项目，参照[pphuman](../pphuman/README.md)；


### 2.3 多目标跟踪
在终端输入以下命令进行预测：
```bash
# FairMOT跟踪
python deploy/python/mot_jde_infer.py --model_dir=output_inference/fairmot_dla34_30e_1088x608 --video_file={your video name}.mp4 --device=GPU

# ByteTrack跟踪
python deploy/python/mot_sde_infer.py --model_dir=output_inference/ppyoloe_crn_l_36e_640x640_mot17half/ --tracker_config=deploy/python/tracker_config.yml --video_file={your video name}.mp4 --device=GPU --scaled=True

# FairMOT多目标跟踪联合HRNet关键点检测（联合推理只支持top-down关键点模型）
python deploy/python/mot_keypoint_unite_infer.py --mot_model_dir=output_inference/fairmot_dla34_30e_1088x608/ --keypoint_model_dir=output_inference/hrnet_w32_384x288/ --video_file={your video name}.mp4 --device=GPU
```

**注意:**
 - 多目标跟踪模型导出和预测具体可参照[mot]](../../configs/mot/README.md)，可分别在各个模型的文档中查找具体用法；
 - 此目录下的跟踪部署为基础前向功能以及联合关键点部署，更多跟踪功能可使用PP-Human项目，参照[pphuman](../pphuman/README.md)，或PP-Tracking项目(绘制轨迹、出入口流量计数)，参照[pptracking](../pptracking/README.md)；


参数说明如下:

| 参数 | 是否必须|含义 |
|-------|-------|----------|
| --model_dir | Yes| 上述导出的模型路径 |
| --image_file | Option | 需要预测的图片 |
| --image_dir  | Option |  要预测的图片文件夹路径   |
| --video_file | Option | 需要预测的视频 |
| --camera_id | Option | 用来预测的摄像头ID，默认为-1(表示不使用摄像头预测，可设置为：0 - (摄像头数目-1) )，预测过程中在可视化界面按`q`退出输出预测结果到：output/output.mp4|
| --device | Option | 运行时的设备，可选择`CPU/GPU/XPU`，默认为`CPU`|
| --run_mode | Option |使用GPU时，默认为paddle, 可选（paddle/trt_fp32/trt_fp16/trt_int8）|
| --batch_size | Option |预测时的batch size，在指定`image_dir`时有效，默认为1 |
| --threshold | Option|预测得分的阈值，默认为0.5|
| --output_dir | Option|可视化结果保存的根目录，默认为output/|
| --run_benchmark | Option| 是否运行benchmark，同时需指定`--image_file`或`--image_dir`，默认为False |
| --enable_mkldnn | Option | CPU预测中是否开启MKLDNN加速，默认为False |
| --cpu_threads | Option| 设置cpu线程数，默认为1 |
| --trt_calib_mode | Option| TensorRT是否使用校准功能，默认为False。使用TensorRT的int8功能时，需设置为True，使用PaddleSlim量化后的模型时需要设置为False |
| --save_results | Option| 是否在文件夹下将图片的预测结果以JSON的形式保存 |


说明：

- 参数优先级顺序：`camera_id` > `video_file` > `image_dir` > `image_file`。
- run_mode：paddle代表使用AnalysisPredictor，精度float32来推理，其他参数指用AnalysisPredictor，TensorRT不同精度来推理。
- 如果安装的PaddlePaddle不支持基于TensorRT进行预测，需要自行编译，详细可参考[预测库编译教程](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html)。
- --run_benchmark如果设置为True，则需要安装依赖`pip install pynvml psutil GPUtil`。
