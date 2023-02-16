English | [简体中文](PPVehicle_QUICK_STARTED.md)

# Quick Start for PP-Vehicle

## Content

- [Environment Preparation](#Environment-Preparation)
- [Model Download](#Model-Download)
- [Configuration](#Configuration)
- [Inference Deployment](#Inference-Deployment)
  - [rtsp_stream](#rtsp_stream)
  - [Nvidia_Jetson](#Nvidia_Jetson)
  - [Parameters](#Parameters)
- [Solutions](#Solutions)
  - [Vehicle Detection](#Vehicle-Detection)
  - [Vehicle Tracking](#Vehicle-Tracking)
  - [License Plate Recognition](#License-Plate-Recognition)
  - [Attribute Recognition](#Attribute-Recognition)
  - [Illegal Parking Detection](#Illegal-Parking-Detection)

## Environment Preparation

Environment Preparation: PaddleDetection version >= release/2.5 or develop

Installation of PaddlePaddle and PaddleDetection

```
# PaddlePaddle CUDA10.1
python -m pip install paddlepaddle-gpu==2.2.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# PaddlePaddle CPU
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

# Clone PaddleDetectionrepositories
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git

# Installing dependencies
cd PaddleDetection
pip install -r requirements.txt
```

1. For installation details, please refer to [Installation Tutorials](../../../../docs/tutorials/INSTALL.md)
2. If you need TensorRT inference acceleration (speed measurement), please install PaddlePaddle with `TensorRT version`. You can download and install it from the [PaddlePaddle Installation Package](https://paddleinference.paddlepaddle.org.cn/v2.2/user_guides/download_lib.html#python) or follow the [Instructions]([https://www](https://www). paddlepaddle.org.cn/inference/master/optimize/paddle_trt.html) or use docker, or self-compiling to prepare the environment.

## Model Download

PP-Vehicle provides object detection, attribute recognition, behaviour recognition and ReID pre-trained models for different applications. Developers can download them directly.

| Task                              | End-to（ms） | Model Solution                                                                                                                                                                                                                           | Model Size                                                            |
|:---------------------------------:|:----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------:|
| Vehicle Detection（high precision） | 25.7ms     | [Multi-Object Tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip)                                                                                                                              | 182M                                                                  |
| Vehicle Detection（Lightweight）    | 13.2ms     | [Multi-Object Tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_ppvehicle.zip)                                                                                                                              | 27M                                                                   |
| Vehicle detection (super lightweight)     | 10ms(Jetson AGX)     | [object detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppvehicle/ppyoloe_plus_crn_t_auxhead_320_60e_ppvehicle.tar.gz)  | 17M     |
| Vehicle Tracking（high precision）  | 40ms       | [Multi-Object Tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip)                                                                                                                              | 182M                                                                  |
| Vehicle Tracking（Lightweight）     | 25ms       | [Multi-Object Tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_ppvehicle.zip)                                                                                                                              | 27M                                                                   |
| Vehicle tracking (super lightweight)     | 13.2ms(Jetson AGX)               | [multi-object tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppvehicle/ppyoloe_plus_crn_t_auxhead_320_60e_ppvehicle.tar.gz)                                                                                                                           | 17M                              |
| License plate recognition         | 4.68ms     | [License plate recognition](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_det_infer.tar.gz) <br> [License plate character recognition](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_rec_infer.tar.gz) | Vehicle Detection：3.9M  <br> License plate character recognition： 12M |
| Vehicle Attribute Recognition     | 7.31ms     | [Vehicle Attribute](https://bj.bcebos.com/v1/paddledet/models/pipeline/vehicle_attribute_model.zip)                                                                                                                                      | 7.2M                                                                  |
| Lane line Segmentation      |   47ms | [Lane line Segmentation](https://bj.bcebos.com/v1/paddledet/models/pipeline/pp_lite_stdc2_bdd100k.zip) | 47M |

Download the model and unzip it into the `. /output_inference` folder.

In the configuration file, the model path defaults to the download path of the model. If the user does not change it, the corresponding model will be downloaded automatically upon inference.

**Notes：**

- The accuracy of detection tracking model is obtained from the joint dataset PPVehicle (integration of the public dataset BDD100K-MOT and UA-DETRAC). For more details, please refer to [PP-Vehicle](../../../../configs/ppvehicle)
- Inference speed is obtained at T4 with TensorRT FP16 enabled, which includes data pre-processing, model inference and post-processing.

## Configuration

PP-Vehicle related configuration locates in ``deploy/pipeline/config/infer_cfg_ppvehicle.yml``. Developers need to set specific task types to use different features.

The features and corresponding task types are as follows.

| Input               | Feature               | Task                                        | Config           |
| ------------------- | --------------------- | ------------------------------------------- | ---------------- |
| Image               | Attribute Recognition | Object Detection Attribute Recognition      | DET ATTR         |
| Single-camera video | Attribute Recognition | Multi-Object Tracking Attribute Recognition | MOT ATTR         |
| Single-camera video | License-plate Recognition | Multi-Object Tracking License-plate Recognition | MOT VEHICLEPLATE |

Take attribute recognition based on video input as an example: Its task type includes multi-object tracking and attributes recognition. The specific configuration is as follows.

```
crop_thresh: 0.5
visual: True
warmup_frame: 50

MOT:
  model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip
  tracker_config: deploy/pipeline/config/tracker_config.yml
  batch_size: 1
  enable: True

VEHICLE_ATTR:
  model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/vehicle_attribute_model.zip
  batch_size: 8
  color_threshold: 0.5
  type_threshold: 0.5
  enable: True
```

**Notes：**

- If the developer needs to carry out different tasks, set the corresponding enables option to be True in the configuration file.
- If the developer only needs to modify the model file path, run the command line with `-o MOT.model_dir=ppyoloe/` after --config, or manually modify the corresponding model path in the configuration file. For more details, please refer to the following parameter descriptions

## Inference Deployment

1. Use the default configuration directly or the configuration file in examples, or modify the configuration in `infer_cfg_ppvehicle.yml`

```
# Example：In vehicle detection，specify configuration file path and test image
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml --image_file=test_image.jpg --device=gpu

# Example：In license plate recognition，directly configure the examples
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_vehicle_plate.yml --video_file=test_video.mp4 --device=gpu
```

2. Use the command line to enable functions or change the model path.

   ```
   #  Example：In vehicle tracking，specify configuration file path and test video, Turn on the MOT model and modify the model path on the command line, the model path specified on the command line has higher priority than the configuration file
   python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_ppvehicle.yml -o MOT.enable=True MOT.model_dir=ppyoloe_infer/ --video_file=test_video.mp4 --device=gpu

   # Example：In vehicle illegal action analysis，specify configuration file path and test video, 命令行中指定违停区域设置、违停时间判断。
   python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_illegal_parking.yml \
                                                      --video_file=../car_test.mov \
                                                      --device=gpu \
                                                      --draw_center_traj \
                                                      --illegal_parking_time=3 \
                                                      --region_type=custom \
                                                      --region_polygon 600 300 1300 300 1300 800 600 800
   ```

### rtsp_stream

The online stream decode based on opencv Capture function, normally support rtsp and rtmp.

- rtsp pull stream

For rtsp pull stream, use --rtsp RTSP [RTSP ...] parameter to specify one or more rtsp streams. Separate the multiple addresses with a space, or replace the video address directly after the video_file with the rtsp stream address), examples as follows

   ```
   # Example: Single video stream for pedestrian attribute recognition
   python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_human_attr.yml -o visual=False --rtsp rtsp://[YOUR_RTSP_SITE]  --device=gpu
   # Example: Multiple-video stream for pedestrian attribute recognition
   python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_human_attr.yml -o visual=False --rtsp rtsp://[YOUR_RTSP_SITE1] rtsp://[YOUR_RTSP_SITE2] --device=gpu                                                                      |
   ```

- rtsp push stream

For rtsp push stream, use --pushurl rtsp:[IP] parameter to push stream to a IP set, and you can visualize the output video by [VLC Player](https://vlc.onl/) with the `open network` funciton. the whole url path is `rtsp:[IP]/videoname`, the videoname here is the basename of the video file to infer, and the default of videoname is `output` when the video come from local camera and there is no video name.

```
# Example：license plate recognition，in this example the whole url path is: rtsp://[YOUR_SERVER_IP]:8554/test_video
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_vehicle_plate.yml --video_file=test_video.mp4  --device=gpu --pushurl rtsp://[YOUR_SERVER_IP]:8554
```
Note:
1. rtsp push stream is based on [rtsp-simple-server](https://github.com/aler9/rtsp-simple-server), please enable this serving first.
It's very easy to use: 1) download the [release package](https://github.com/aler9/rtsp-simple-server/releases) which is compatible with your workspace. 2) run command './rtsp-simple-server', which works as a rtsp server.
2. the output visualize will be frozen frequently if the model cost too much time, we suggest to use faster model like ppyoloe_s in tracking, this is simply replace mot_ppyoloe_l_36e_pipeline.zip with mot_ppyoloe_s_36e_pipeline.zip in model config yaml file.

### Nvidia_Jetson

Due to the large gap in computing power of the Jetson platform compared to the server, we suggest:

1. choose a lightweight model, we provide a new model named [PP-YOLOE-Plus Tiny](../../../../configs/ppvehicle/README.md)，which achieve 20fps with four rtsp streams work togather on Jetson AGX.
2. For further speedup, you can set frame skipping of tracking; we recommend 2 or 3: `skip_frame_num: 3`

PP-YOLOE-Plus Tiny module speed test data on AGX：（a single car in the test video）

| module  | time cost per frame(ms)  | speed(fps)  |
|:----------|:----------|:----------|
| tracking    | 13    | 77    |
| Attribute    | 20.2    | 49.4    |
| Plate    | -    | -    |


### Parameters

#

| Parameters             | Necessity | Implications                                                                                                                                                                                                                                      |
| ---------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| --config               | Yes       | Path to configuration file                                                                                                                                                                                                                        |
| -o                     | Option    | Overwrite the corresponding configuration in the configuration file                                                                                                                                                                               |
| --image_file           | Option    | Images to be predicted                                                                                                                                                                                                                            |
| --image_dir            | Option    | Path to the images folder to be predicted                                                                                                                                                                                                         |
| --video_file           | Option    | Video to be predicted, or rtsp stream address (rtsp parameter recommended)                                                                                                                                                                        |
| --rtsp                 | Option    | rtsp video stream address, supports one or more simultaneous streams input                                                                                                                                                                        |
| --camera_id            | Option    | The camera ID for prediction, default is -1 ( for no camera prediction, can be set to 0 - (number of cameras - 1) ), press `q` in the visualization interface during the prediction process to output the prediction result to: output/output.mp4 |
| --device               | Option    | Running device, options include `CPU/GPU/XPU`, and the default is `CPU`.                                                                                                                                                                          |
| --pushurl              | Option    | push the output video to rtsp stream, normaly start with `rtsp://`; this has higher priority than local video save, while this is set, pipeline will not save local visualize video, the default is "", means this will not work now.|
| --output_dir           | Option    | The root directory for the visualization results, and the default is output/                                                                                                                                                                      |
| --run_mode             | Option    | For GPU, the default is paddle, with (paddle/trt_fp32/trt_fp16/trt_int8) as optional                                                                                                                                                              |
| --enable_mkldnn        | Option    | Whether to enable MKLDNN acceleration in CPU prediction, the default is False                                                                                                                                                                     |
| --cpu_threads          | Option    | Set the number of cpu threads, and the default is 1                                                                                                                                                                                               |
| --trt_calib_mode       | Option    | Whether TensorRT uses the calibration function, and the default is False; set to True when using TensorRT's int8 function and False when using the PaddleSlim quantized model                                                                     |
| --do_entrance_counting | Option    | Whether to count entrance/exit traffic flows, the default is False                                                                                                                                                                                |
| --draw_center_traj     | Option    | Whether to draw center trajectory, the default is False                                                                                                                                                                                           |
| --region_type          | Option    | 'horizontal' (default), 'vertical': traffic count direction; 'custom': set illegal parking area                                                                                                                                                   |
| --region_polygon       | Option    | Set the coordinates of the polygon multipoint in the illegal parking area. No default.                                                                                                                                                            |
| --illegal_parking_time | Option    | Set the time threshold for illegal parking in seconds (s), -1 (default) indicates no check                                                                                                                                                        |

## Solutions

The overall solution for PP-Vehicle v2 is shown in the graph below:

<div width="1000" align="center">
  <img src="https://user-images.githubusercontent.com/31800336/218659932-31f4298c-042d-436d-9845-18879f5d31e3.png"/>
</div>

###

### Vehicle detection

- Take PP-YOLOE L as the object detection model
- For detailed documentation, please refer to [PP-YOLOE](../../../../configs/ppyoloe/) and [Multiple-Object-Tracking](ppvehicle_mot_en.md)

### Vehicle tracking

- Vehicle tracking by SDE solution
- Adopt PP-YOLOE L (high precision) and S (lightweight) for detection models
- Adopt the OC-SORT solution for racking module
- Refer to [OC-SORT](../../../../configs/mot/ocsort) and [Multi-Object Tracking](ppvehicle_mot_en.md) for details

### Attribute Recognition

- Use PP-LCNet provided by PaddleClas to recognize vehicle colours and model attributes.
- For details, please refer to [Attribute Recognition](ppvehicle_attribute_en.md)

### License plate recognition

- Use ch_PP-OCRv3_det+ch_PP-OCRv3_rec model to recognize license plate number
- For details, please refer to [Plate Recognition](ppvehicle_plate_en.md)

### Illegal Parking Detection

- Use vehicle tracking model (high precision) PP-YOLOE L to determine whether the parking is illegal based on the vehicle's trajectory and the designated illegal parking area. If it is illegal parking, display the illegal parking plate number.

- For details, please refer to [Illegal Parking Detection](ppvehicle_illegal_parking_en.md)

#### Vehicle Press Line

- Use segmentation model PP-LiteSeg to get the lane line in frame, combine it with vehicle route to find out the vehicle against traffic.
- For details, please refer to [Vehicle Press Line](ppvehicle_press_en.md)

#### Vehicle Retrograde

- Use segmentation model PP-LiteSeg to get the lane line in frame, combine it with vehicle detection box to juege if the car is pressing on lines.
- For details, please refer to [Vehicle Retrograde](ppvehicle_retrograde_en.md)

