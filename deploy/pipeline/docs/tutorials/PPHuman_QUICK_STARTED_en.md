English | [简体中文](PPHuman_QUICK_STARTED.md)

# Quick Start for PP-Human

## Contents

- [Environment Preparation](#Environment-Preparation)
- [Model Download](#Model-Download)
- [Configuration](#Configuration)
- [Inference Deployment](#Inference-Deployment)
  - [Parameters](#Parameters)
- [Solutions](#Solutions)
  - [Pedestrian Detection](#edestrian-Detection)
  - [Pedestrian Tracking](#Pedestrian-Tracking)
  - [Multi-camera & multi-pedestrain tracking](#Multi-camera-&-multi-pedestrain-tracking)
  - [Attribute Recognition](#Attribute-Recognition)
  - [Behavior Recognition](#Behavior-Recognition)

## Environment Preparation

Environment requirements： PaddleDetection>= release/2.4  or develop version

Installation of PaddlePaddle and PaddleDetection

```
# PaddlePaddle CUDA10.1
python -m pip install paddlepaddle-gpu==2.2.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# PaddlePaddle CPU
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

#Clone PaddleDetection repositories
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git

# Install dependencies
cd PaddleDetection
pip install -r requirements.txt
```

1. For installation details, please refer to [Installation Tutorials](../../../../docs/tutorials/INSTALL.md)
2. If you need TensorRT inference acceleration (speed measurement), please install PaddlePaddle with `TensorRT version`. You can download and install it from the [PaddlePaddle Installation Package](https://paddleinference.paddlepaddle.org.cn/v2.2/user_guides/download_lib.html#python) or follow the [Instructions](https://www. paddlepaddle.org.cn/inference/master/optimize/paddle_trt.html) or use docker, or self-compiling to prepare the environment.

## Model Download

PP-Human provides object detection, attribute recognition, behaviour recognition and ReID pre-trained models for different applications. Developers can download them directly.

| Task                                   | End-to（ms）           | Model Solution                                                                                                                                                                                                                                                                                                             | Model Size                                                                                       |
|:--------------------------------------:|:--------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|
| Pedestrian Detection (high precision)  | 25.1ms               | [Multi-Object Tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                                                                                                                                                 | 182M                                                                                             |
| Pedestrian Detection (Lightweight)     | 16.2ms               | [Multi-Object Tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip)                                                                                                                                                                                                                 | 27M                                                                                              |
| Pedestrian Tracking (high precision)   | 31.8ms               | [Multi-Object Tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                                                                                                                                                 | 182M                                                                                             |
| Pedestrian Tracking (Lightweight)      | 21.0ms               | [Multi-Object Tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip)                                                                                                                                                                                                                 | 27M                                                                                              |
| Attribute Recognition (high precision) | Single Person 8.5ms  | [Object Detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [Attribute Recognition](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_small_person_attribute_954_infer.zip)                                                                                         | Object Detection：182M<br>Attribute Recogniton：86M                                                |
| Attribute Recognition (Lightweight)    | Single Person 7.1ms  | [Object Detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [Attribute Recogniton](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPLCNet_x1_0_person_attribute_945_infer.zip)                                                                                           | Object Detection：182M<br>Attribute Recogniton：86M                                                |
| Falling Detection                      | Single Person 10ms   | [Multi-Object Tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) <br> [Keypoint Detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip) <br> [Skeleton Based Action Recognition](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip) | Multi-Object Tracking：182M<br>Keypoint Detection：101M<br>Skeleton Based Action Recognition：21.8M |
| Breaking-In Detection                  | 31.8ms               | [Multi-Object Tracking](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                                                                                                                                                 | Multi-Object Tracking：182M                                                                       |
| Fighting Detection                     | 19.7ms               | [Video Classification](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                                                                                                                                                  | 90M                                                                                              |
| Smoking Detection                      | Single Person 15.1ms | [Object Detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[Object Detection Based On Body ID](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppyoloe_crn_s_80e_smoking_visdrone.zip)                                                                                    | Object Detection：182M<br>Object Detection Based On Body ID：27M                                   |
| Phone-calling Detection                | Single Person 6.0ms  | [Object Detection](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[Image Classification Based On Body ID](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_calling_halfbody.zip)                                                                                     | Object Detection：182M<br>Image Classification Based On Body ID：45M                               |

Download the model and unzip it into the `. /output_inference` folder.

In the configuration file, the model path defaults to the download path of the model. If the user does not change it, the corresponding model will be downloaded automatically upon inference.

**Note:**

- Model accuracy is tested on fused datasets, which contain both open source and enterprise datasets.
- ReID model accuracy is tested on the Market1501 dataset
- Prediction speed is obtained at T4 with TensorRT FP16 enabled, which includes data pre-processing, model inference and post-processing.

## Configuration

The PP-Human-related configuration is located in ``deploy/pipeline/config/infer_cfg_pphuman.yml``, and this configuration file contains all the features currently supported by PP-Human. If you want to see the configuration for a specific feature, please refer to the relevant configuration in ``deploy/pipeline/config/examples/``. In addition, the contents of the configuration file can be modified with the `-o`command line parameter. E.g. to modify the model directory of an attribute, developers can run ```-o ATTR.model_dir="DIR_PATH"``.

The features and corresponding task types are as follows.

| Input               | Feature               | Task                                                       | Config                  |
| ------------------- | --------------------- | ---------------------------------------------------------- | ----------------------- |
| Image               | Attribute Recognition | Object Detection Attribute Recognition                     | DET ATTR                |
| Single-camera video | Attribute Recognition | Multi-Object Tracking Attribute Recognition                | MOT ATTR                |
| Single-camera video | Behaviour Recognition | Multi-Object Tracking Keypoint Detection Falling detection | MOT KPT SKELETON_ACTION |

Take attribute recognition based on video input as an example: Its task type includes multi-object tracking and attributes recognition. The specific configuration is as follows.

```
crop_thresh: 0.5
attr_thresh: 0.5
visual: True

MOT:
  model_dir: https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip
  tracker_config: deploy/pipeline/config/tracker_config.yml
  batch_size: 1
  enable: True

ATTR:
  model_dir:  https://bj.bcebos.com/v1/paddledet/models/pipeline/PPLCNet_x1_0_person_attribute_945_infer.zip
  batch_size: 8
  enable: True
```

**Note:**

- If developer needs to carry out different tasks, set the corresponding enables option to be True in the configuration file.

## Inference Deployment

1. Use the default configuration directly or the configuration file in examples, or modify the configuration in `infer_cfg_pphuman.yml`

   ```
   # Example: In pedestrian detection model, specify configuration file path and test image, and image input opens detection model by default
   python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml --image_file=test_image.jpg --device=gpu
   # Example: In pedestrian attribute recognition, directly configure the examples
   python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_human_attr.yml --video_file=test_video.mp4 --device=gpu
   ```

2. Use the command line to enable functions or change the model path.

```
# Example: Pedestrian tracking, specify config file path, model path and test video. The specified model path on the command line has a higher priority than the config file.
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml -o MOT.enable=True MOT.model_dir=ppyoloe_infer/ --video_file=test_video.mp4 --device=gpu

# Example: In behaviour recognition, with fall recognition as an example, enable the SKELETON_ACTION model on the command line
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/infer_cfg_pphuman.yml -o SKELETON_ACTION.enbale=True --video_file=test_video.mp4 --device=gpu
```

3. For rtsp stream, use --rtsp RTSP [RTSP ...] parameter to specify one or more rtsp streams. Separate the multiple addresses with a space, or replace the video address directly after the video_file with the rtsp stream address), examples as follows

```
# Example: Single video stream for pedestrian attribute recognition
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_human_attr.yml -o visual=False --rtsp rtsp://[YOUR_RTSP_SITE]  --device=gpu
# Example: Multiple-video stream for pedestrian attribute recognition
python deploy/pipeline/pipeline.py --config deploy/pipeline/config/examples/infer_cfg_human_attr.yml -o visual=False --rtsp rtsp://[YOUR_RTSP_SITE1] rtsp://[YOUR_RTSP_SITE2] --device=gpu                                                                      |
```

### Jetson Deployment

Due to the large gap in computing power of the Jetson platform compared to the server, we suggest:

1. choose a lightweight model, especially for tracking model, `ppyoloe_s: https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_pipeline.zip` is recommended
2. For frame skipping of tracking; we recommend 2 or 3: `skip_frame_num: 3`

With this recommended configuration, it is possible to achieve higher speeds on the TX2 platform. It has been tested with attribute case, with speeds up to 20fps. The configuration file can be modified directly (recommended) or from the command line (not recommended due to its long fields).

### Parameters

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
| --output_dir           | Option    | The root directory for the visualization results, and the default is output/                                                                                                                                                                      |
| --run_mode             | Option    | For GPU, the default is paddle, with (paddle/trt_fp32/trt_fp16/trt_int8) as optional                                                                                                                                                              |
| --enable_mkldnn        | Option    | Whether to enable MKLDNN acceleration in CPU prediction, the default is False                                                                                                                                                                     |
| --cpu_threads          | Option    | Set the number of cpu threads, and the default is 1                                                                                                                                                                                               |
| --trt_calib_mode       | Option    | Whether TensorRT uses the calibration function, and the default is False; set to True when using TensorRT's int8 function and False when using the PaddleSlim quantized model                                                                     |
| --do_entrance_counting | Option    | Whether to count entrance/exit traffic flows, the default is False                                                                                                                                                                                |
| --draw_center_traj     | Option    | Whether to map the trace, the default is False                                                                                                                                                                                                    |
| --region_type          | Option    | 'horizontal' (default), 'vertical': traffic count direction; 'custom': set break-in area                                                                                                                                                          |
| --region_polygon       | Option    | Set the coordinates of the polygon multipoint in the break-in area. No default.                                                                                                                                                                   |
| --do_break_in_counting | Option    | Area break-in checks                                                                                                                                                                                                                              |

## Solutions

The overall solution for PP-Human v2 is shown in the graph below:

### Pedestrian detection

- Take PP-YOLOE L as the object detection model
- For detailed documentation, please refer to [PP-YOLOE](... /... /... /... /configs/ppyoloe/) and [Multiple-Object-Tracking](pphuman_mot_en.md)

### Pedestrian tracking

- Vehicle tracking by SDE solution
- Adopt PP-YOLOE L (high precision) and S (lightweight) for detection models
- Adopt the OC-SORT solution for racking module
- Refer to [OC-SORT](... /... /... /... /configs/mot/ocsort) and [Multi-Object Tracking](pphuman_mot_en.md) for details

### Multi-camera & multi-pedestrain tracking

- Use PP-YOLOE & OC-SORT to acquire single-camera multi-object tracking trajectory
- Extract features for each frame using ReID (StrongBaseline network).
- Match multi-camera trajectory features to obtain multi-camera tracking results.
- Refer to [Multi-camera & multi-pedestrain tracking](pphuman_mtmct_en.md) for details.

### Attribute Recognition

- Use PP-YOLOE + OC-SORT to track the human body.
- Use PP-HGNet, PP-LCNet (multi-classification model) to complete the attribute recognition. Main attributes include age, gender, hat, eyes, top and bottom dressing style, backpack.
- Refer to [attribute recognition](pphuman_attribute_en.md) for details.

### Behaviour Recognition:

- Four behaviour recognition solutions are provided:

- 1. Behaviour recognition based on skeletal points, e.g. falling recognition

- 2. Behaviour recognition based on image classification, e.g. phone call recognition

- 3. Behaviour recognition based on detection, e.g. smoking recognition

- 4. Behaviour recognition based on Video classification, e.g. fighting recognition

- For details, please refer to [Behaviour Recognition](pphuman_action_en.md)
