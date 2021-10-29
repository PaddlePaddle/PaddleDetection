# Inference Benchmark

## 一、Prepare the Environment
- 1、Test Environment:
  - CUDA 10.1
  - CUDNN 7.6
  - TensorRT-6.0.1
  - PaddlePaddle v2.0.1
  - The GPUS are Tesla V100 and GTX 1080 Ti and Jetson AGX Xavier
- 2、Test Method:
  - In order to compare the inference speed of different models, the input shape is 3x640x640, use `demo/000000014439_640x640.jpg`.
  - Batch_size=1
  - Delete the warmup time of the first 100 rounds and test the average time of 100 rounds in ms/image, including network calculation time and data copy time to CPU.
  - Using Fluid C++ prediction engine: including Fluid C++ prediction, Fluid TensorRT prediction, the following test Float32 (FP32) and Float16 (FP16) inference speed.

**Attention:**  For TensorRT, please refer to the [TENSOR tutorial](TENSOR_RT.md) for the difference between fixed and dynamic dimensions. Due to the imperfect support for the two-stage model under fixed size, dynamic size test was adopted for the Faster RCNN model. Fixed size and dynamic size do not support exactly the same OP for fusion, so the performance of the same model tested at fixed size and dynamic size may differ slightly.


## 二、Inferring Speed

### 1、Linux System
#### （1）Tesla V100

| Model           | backbone      | Fixed size or not | The net size | paddle_inference | trt_fp32 | trt_fp16 |
| --------------- | ------------- | ----------------- | ------------ | ---------------- | -------- | -------- |
| Faster RCNN FPN | ResNet50      | no                | 640x640      | 27.99            | 26.15    | 21.92    |
| Faster RCNN FPN | ResNet50      | no                | 800x1312     | 32.49            | 25.54    | 21.70    |
| YOLOv3          | Mobilenet\_v1 | yes               | 608x608      | 9.74             | 8.61     | 6.28     |
| YOLOv3          | Darknet53     | yes               | 608x608      | 17.84            | 15.43    | 9.86     |
| PPYOLO          | ResNet50      | yes               | 608x608      | 20.77            | 18.40    | 13.53    |
| SSD             | Mobilenet\_v1 | yes               | 300x300      | 5.17             | 4.43     | 4.29     |
| TTFNet          | Darknet53     | yes               | 512x512      | 10.14            | 8.71     | 5.55     |
| FCOS            | ResNet50      | yes               | 640x640      | 35.47            | 35.02    | 34.24    |


#### （2）Jetson AGX Xavier

| Model           | backbone      | Fixed size or not | The net size | paddle_inference | trt_fp32 | trt_fp16 |
| --------------- | ------------- | ----------------- | ------------ | ---------------- | -------- | -------- |
| Faster RCNN FPN | ResNet50      | no                | 640x640      | 169.45           | 158.92   | 119.25   |
| Faster RCNN FPN | ResNet50      | no                | 800x1312     | 228.07           | 156.39   | 117.03   |
| YOLOv3          | Mobilenet\_v1 | yes               | 608x608      | 48.76            | 43.83    | 18.41    |
| YOLOv3          | Darknet53     | yes               | 608x608      | 121.61           | 110.30   | 42.38    |
| PPYOLO          | ResNet50      | yes               | 608x608      | 111.80           | 99.40    | 48.05    |
| SSD             | Mobilenet\_v1 | yes               | 300x300      | 10.52            | 8.84     | 8.77     |
| TTFNet          | Darknet53     | yes               | 512x512      | 73.77            | 64.03    | 31.46    |
| FCOS            | ResNet50      | yes               | 640x640      | 217.11           | 214.38   | 205.78   |

### 2、Windows System
#### （1）GTX 1080Ti

| Model           | backbone      | Fixed size or not | The net size | paddle_inference | trt_fp32 | trt_fp16 |
| --------------- | ------------- | ----------------- | ------------ | ---------------- | -------- | -------- |
| Faster RCNN FPN | ResNet50      | no                | 640x640      | 50.74            | 57.17    | 62.08    |
| Faster RCNN FPN | ResNet50      | no                | 800x1312     | 50.31            | 57.61    | 62.05    |
| YOLOv3          | Mobilenet\_v1 | yes               | 608x608      | 14.51            | 11.23    | 11.13    |
| YOLOv3          | Darknet53     | yes               | 608x608      | 30.26            | 23.92    | 24.02    |
| PPYOLO          | ResNet50      | yes               | 608x608      | 38.06            | 31.40    | 31.94    |
| SSD             | Mobilenet\_v1 | yes               | 300x300      | 16.47            | 13.87    | 13.76    |
| TTFNet          | Darknet53     | yes               | 512x512      | 21.83            | 17.14    | 17.09    |
| FCOS            | ResNet50      | yes               | 640x640      | 71.88            | 69.93    | 69.52    |
