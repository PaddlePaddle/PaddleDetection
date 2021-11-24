# Python端预测部署

在PaddlePaddle中预测引擎和训练引擎底层有着不同的优化方法, 预测引擎使用了AnalysisPredictor，专门针对推理进行了优化，是基于[C++预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/native_infer.html)的Python接口，该引擎可以对模型进行多项图优化，减少不必要的内存拷贝。如果用户在部署已训练模型的过程中对性能有较高的要求，我们提供了独立于PaddleDetection的预测脚本，方便用户直接集成部署。

主要包含两个步骤：

- 导出预测模型
- 基于Python进行预测

PaddleDetection在训练过程包括网络的前向和优化器相关参数，而在部署过程中，我们只需要前向参数，具体参考:[导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/deploy/EXPORT_MODEL.md)
导出后目录下，包括`infer_cfg.yml`, `model.pdiparams`,  `model.pdiparams.info`, `model.pdmodel`四个文件。

## 1. 对FairMOT模型的导出和预测

### 1.1 导出预测模型
```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/fairmot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.pdparams
```

### 1.2 用导出的模型基于Python去预测
```bash
python deploy/pptracking/python/mot_jde_infer.py --model_dir=output_inference/fairmot_hrnetv2_w18_dlafpn_30e_576x320 --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**注意:**
 - 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。
 - 跟踪结果txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`。
 - 对于多类别或车辆的FairMOT模型的导出和Python预测只需更改相应的config和模型权重即可。如：
 ```
 job_name=mcfairmot_hrnetv2_w18_dlafpn_30e_576x320_visdrone
 model_type=mot/mcfairmot
 config=configs/${model_type}/${job_name}.yml

 CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/mot/${job_name}.pdparams
 python deploy/pptracking/python/mot_jde_infer.py --model_dir=output_inference/${job_name} --video_file={your video name}.mp4 --device=GPU --save_mot_txts
 ```
 - 多类别跟踪结果txt文件每行信息是`frame,id,x1,y1,w,h,score,cls_id,-1,-1`。


## 2. 对DeepSORT模型的导出和预测
### 2.1 导出预测模型

Step 1：导出检测模型
```bash
# 导出JDE YOLOv3行人检测模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/deepsort/detector/jde_yolov3_darknet53_30e_1088x608_mix.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/jde_yolov3_darknet53_30e_1088x608_mix.pdparams

# 或导出PPYOLOv2行人检测模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/deepsort/detector/ppyolov2_r50vd_dcn_365e_640x640_mot17half.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/ppyolov2_r50vd_dcn_365e_640x640_mot17half.pdparams
```

Step 2：导出ReID模型
```bash
# 导出PCB Pyramid ReID模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/deepsort/reid/deepsort_pcb_pyramid_r101.yml -o reid_weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pcb_pyramid_r101.pdparams
# 或者导出PPLCNet ReID模型
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/deepsort/reid/deepsort_pplcnet.yml -o reid_weights=https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet.pdparams
```

### 2.2 用导出的模型基于Python去预测

```bash
# 用导出JDE YOLOv3行人检测模型和PCB Pyramid ReID模型
python deploy/pptracking/python/mot_sde_infer.py --model_dir=output_inference/jde_yolov3_darknet53_30e_1088x608_mix/ --reid_model_dir=output_inference/deepsort_pcb_pyramid_r101/ --video_file={your video name}.mp4 --device=GPU --save_mot_txts

# 或用导出的PPYOLOv2行人检测模型和PPLCNet ReID模型
python deploy/pptracking/python/mot_sde_infer.py --model_dir=output_inference/ppyolov2_r50vd_dcn_365e_640x640_mot17half/ --reid_model_dir=output_inference/deepsort_pplcnet/ --video_file={your video name}.mp4 --device=GPU --scaled=True --save_mot_txts
```
**注意:**
 - 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`(对每个视频保存一个txt)或`--save_images`表示保存跟踪结果可视化图片。
 - 跟踪结果txt文件每行信息是`frame,id,x1,y1,w,h,score,-1,-1,-1`。
 - `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE的YOLOv3则为False，如果使用通用检测模型则为True。


## 3. 跨境跟踪模型的导出和预测
### 3.1 导出预测模型
Step 1：下载导出的检测模型
```bash
wget https://paddledet.bj.bcebos.com/models/mot/deepsort/picodet_l_640_aic21mtmct_vehicle.tar
tar -xvf picodet_l_640_aic21mtmct_vehicle.tar
```
Step 2：下载导出的ReID模型
```bash
wget https://paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet_vehicle.tar
tar -xvf deepsort_pplcnet_vehicle.tar
```

### 3.2 用导出的模型基于Python去预测
```bash
# 用导出PicoDet车辆检测模型和PPLCNet车辆ReID模型
python deploy/pptracking/python/mot_sde_infer.py --model_dir=picodet_l_640_aic21mtmct_vehicle/ --reid_model_dir=deepsort_pplcnet_vehicle/ --mtmct_dir={your mtmct scene video folder} --mtmct_cfg=mtmct_cfg --device=GPU --scaled=True --save_mot_txts --save_images
```
**注意:**
  跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`(对每个视频保存一个txt)，或`--save_images`表示保存跟踪结果可视化图片。
  跨镜头跟踪结果txt文件每行信息是`carame_id,frame,id,x1,y1,w,h,-1,-1`。
  `--scaled`表示在模型输出结果的坐标是否已经是缩放回原图的，如果使用的检测模型是JDE的YOLOv3则为False，如果使用通用检测模型则为True。
  `--mtmct_dir`是MTMCT预测的某个场景的文件夹名字，里面包含该场景不同摄像头拍摄视频的图片文件夹，其数量至少为两个。
  `--mtmct_cfg`是MTMCT预测的某个场景的配置文件，里面包含该一些trick操作的开关和该场景摄像头相关设置的文件路径，用户可以自行更改相关路径以及设置某些操作是否启用。


## 参数说明:

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
| --do_entrance_counting | Option | 是否统计出入口流量，默认为False |
| --draw_center_traj | Option | 是否绘制跟踪轨迹，默认为False |
| --mtmct_dir | Option | 需要进行MTMCT跨境头跟踪预测的图片文件夹路径，默认为None |
| --mtmct_cfg | Option | 需要进行MTMCT跨境头跟踪预测的配置文件路径，默认为None |

说明：

- 参数优先级顺序：`camera_id` > `video_file` > `image_dir` > `image_file`。
- run_mode：fluid代表使用AnalysisPredictor，精度float32来推理，其他参数指用AnalysisPredictor，TensorRT不同精度来推理。
- 如果安装的PaddlePaddle不支持基于TensorRT进行预测，需要自行编译，详细可参考[预测库编译教程](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html)。
- --run_benchmark如果设置为True，则需要安装依赖`pip install pynvml psutil GPUtil`。
