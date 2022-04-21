English | [简体中文](README.md)

# PP-Human— a Real-Time Pedestrian Analysis Tool

PP-Human serves as the first open-source tool of real-time pedestrian anaylsis relying on the PaddlePaddle deep learning framework. Versatile and efficient in deployment, it has been used in various senarios. PP-Human
offers many input options, including image/single-camera video/multi-camera video, and covers multi-object tracking, attribute recognition, and action recognition. PP-Human can be applied to intelligent traffic, the intelligent community, industiral patrol, and so on. It supports server-side deployment and TensorRT acceleration，and achieves real-time analysis on the T4 server.

Community intelligent management supportted by PP-Human, please refer to this [AI Studio project](https://aistudio.baidu.com/aistudio/projectdetail/3679564) for quick start tutorial.

Full-process operation tutorial of PP-Human, covering training, deployment, action expansion, please refer to this [AI Studio project](https://aistudio.baidu.com/aistudio/projectdetail/3842982).

## I. Environment Preparation

Requirement: PaddleDetection version >= release/2.4 or develop


The installation of PaddlePaddle and PaddleDetection

```
# PaddlePaddle CUDA10.1
python -m pip install paddlepaddle-gpu==2.2.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# PaddlePaddle CPU
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

# Clone the PaddleDetection repository
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git

# Install other dependencies
cd PaddleDetection
pip install -r requirements.txt
```

1. For details of the installation, please refer to this [document](../../docs/tutorials/INSTALL.md)
2. Please install `Paddle-TensorRT` if your want speedup inference by TensorRT. You can download the whl package from [Paddle-whl-list](https://paddleinference.paddlepaddle.org.cn/v2.2/user_guides/download_lib.html#python), or prepare the envs by yourself follows the [Install-Guide](https://www.paddlepaddle.org.cn/inference/master/optimize/paddle_trt.html).

## II. Quick Start

### 1. Model Download

To make users have access to models of different scenarios, PP-Human provides pre-trained models of object detection, attribute recognition， behavior recognition, and ReID.

| Task            | Scenario | Precision | Inference Speed（FPS） | Model Inference and Deployment |
| :---------:     |:---------:     |:---------------     | :-------:  | :------:      |
| Object Detection        | Image/Video Input | mAP: 56.3  | 28.0ms           | [Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) |
| Attribute Recognition    | Image/Video Input  Attribute Recognition | MOTA: 72.0 |  33.1ms       | [Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.zip) |
| Keypoint Detection    | Video Input  Action Recognition | mA: 94.86 | 2ms per person        | [Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip)
| Behavior Recognition   |  Video Input  Bheavior Recognition  | Precision 96.43 |  2.7ms per person          | [Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip) |
| ReID         | Multi-Target Multi-Camera Tracking   | mAP: 98.8 | 1.5ms per person    | [Link](https://bj.bcebos.com/v1/paddledet/models/pipeline/reid_model.zip) |

Then, unzip the downloaded model to the folder `./output_inference`.

**Note: **

- The model precision is decided by the fusion of datasets which include open-source datasets and enterprise ones.
- The precision on ReID model is evaluated on Market1501.
- The inference speed is tested on T4, using TensorRT FP16. The pipeline of preprocess, prediction and postprocess is included.

### 2. Preparation of Configuration Files

Configuration files of PP-Human are stored in ```deploy/pphuman/config/infer_cfg.yml```. Different tasks are for different functions, so you need to set the task type beforhand.

Their correspondence is as follows:

| Input | Function | Task Type | Config |
|-------|-------|----------|-----|
| Image | Attribute Recognition | Object Detection  Attribute Recognition | DET ATTR |
| Single-Camera Video | Attribute Recognition | Multi-Object Tracking  Attribute Recognition | MOT ATTR |
| Single-Camera Video | Behavior Recognition | Multi-Object Tracking  Keypoint Detection  Action Recognition | MOT KPT ACTION |

For example, for the attribute recognition with the video input, its task types contain multi-object tracking and attribute recognition, and the config is:

```
crop_thresh: 0.5
attr_thresh: 0.5
visual: True

MOT:
  model_dir: output_inference/mot_ppyoloe_l_36e_pipeline/
  tracker_config: deploy/pphuman/config/tracker_config.yml
  batch_size: 1

ATTR:
  model_dir: output_inference/strongbaseline_r50_30e_pa100k/
  batch_size: 8
```

**Note: **

- For different tasks, users could add `--enable_attr=True` or `--enable_action=True` in command line and do not need to set config file.
- if only need to change the model path, users could add `--model_dir det=ppyoloe/` in command line and do not need to set config file. For details info please refer to doc below.


### 3. Inference and Deployment

```
# Pedestrian detection. Specify the config file path and test images
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml --image_file=test_image.jpg --device=gpu [--run_mode trt_fp16]

# Pedestrian tracking. Specify the config file path and test videos
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml --video_file=test_video.mp4 --device=gpu [--run_mode trt_fp16]

# Pedestrian tracking. Specify the config file path, the model path and test videos
# The model path specified on the command line prioritizes over the config file
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml --video_file=test_video.mp4 --device=gpu --model_dir det=ppyoloe/ [--run_mode trt_fp16]

# Attribute recognition. Specify the config file path and test videos
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml --video_file=test_video.mp4 --device=gpu --enable_attr=True [--run_mode trt_fp16]

# Action Recognition. Specify the config file path and test videos
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml --video_file=test_video.mp4 --device=gpu --enable_action=True [--run_mode trt_fp16]

# Pedestrian Multi-Target Multi-Camera tracking. Specify the config file path and the directory of test videos
python deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml --video_dir=mtmct_dir/ --device=gpu [--run_mode trt_fp16]

```

Other usage please refer to [sub-task docs](./docs)

### 3.1 Description of Parameters

| Parameter | Optional or not| Meaning |
|-------|-------|----------|
| --config | Yes | Config file path |
| --model_dir | Option | the model paths of different tasks in PP-Human, with a priority higher than config files. For example, `--model_dir det=better_det/ attr=better_attr/` |
| --image_file | Option | Images to-be-predicted  |
| --image_dir  | Option |  The path of folders of to-be-predicted images  |
| --video_file | Option | Videos to-be-predicted |
| --camera_id | Option | ID of the inference camera is -1 by default (means inference without cameras，and it can be set to 0 - (number of cameras-1)), and during the inference, click `q` on the visual interface to exit and output the inference result to output/output.mp4|
| --enable_attr| Option | Enable attribute recognition or not |
| --enable_action| Option | Enable action recognition or not |
| --device | Option | During the operation，available devices are `CPU/GPU/XPU`，and the default is `CPU`|
| --output_dir | Option| The default root directory which stores the visualization result is output/|
| --run_mode | Option | When using GPU，the default one is paddle, and all these are available（paddle/trt_fp32/trt_fp16/trt_int8）.|
| --enable_mkldnn | Option |Enable the MKLDNN acceleration or not in the CPU inference, and the default value is false |
| --cpu_threads | Option| The default CPU thread is 1 |
| --trt_calib_mode | Option| Enable calibration on TensorRT or not, and the default is False. When using the int8 of TensorRT，it should be set to True; When using the model quantized by PaddleSlim, it should be set to False. |


## III. Introduction to the Solution

The overall solution of PP-Human is as follows:

<div width="1000" align="center">
  <img src="https://user-images.githubusercontent.com/48054808/160078395-e7b8f2db-1d1c-439a-91f4-2692fac25511.png"/>
</div>


### 1. Object Detection
- Use PP-YOLOE L as the model of object detection
- For details, please refer to [PP-YOLOE](../../configs/ppyoloe/) and [Detection and Tracking](docs/mot_en.md)

### 2. Multi-Object Tracking
- Conduct multi-object tracking with the SDE solution
- Use PP-YOLOE L as the detection model
- Use the Bytetrack solution to track modules
- For details, refer to [Bytetrack](configs/mot/bytetrack) and [Detection and Tracking](docs/mot_en.md)

### 3. Multi-Camera Tracking
- Use PP-YOLOE + Bytetrack to obtain the tracks of single-camera multi-object tracking
- Use ReID（centroid network）to extract features of the detection result of each frame
- Match the features of multi-camera tracks to get the cross-camera tracking result
- For details, please refer to [Multi-Camera Tracking](docs/mtmct_en.md)

### 4. Attribute Recognition
- Use PP-YOLOE + Bytetrack to track humans
- Use StrongBaseline（a multi-class model）to conduct attribute recognition, and the main attributes include age, gender, hats, eyes, clothing, and backpacks.
- For details, please refer to [Attribute Recognition](docs/attribute_en.md)

### 5. Action Recognition
- Use PP-YOLOE + Bytetrack to track humans
- Use HRNet for keypoint detection and get the information of the 17 key points in the human body
- According to the changes of the key points of the same person within 50 frames, judge whether the action made by the person within 50 frames is a fall with the help of ST-GCN
- For details, please refer to [Action Recognition](docs/action_en.md)
