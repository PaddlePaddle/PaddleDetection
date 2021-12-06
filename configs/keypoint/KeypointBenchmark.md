# Keypoint Inference Benchmark

## Benchmark on Server
We tested benchmarks in different runtime environments。 See the table below for details.

| Model | CPU + MKLDNN | GPU | TensorRT (FP32) | TensorRT (FP16) |
| :------------------------ | :------: | :---: | :---: | :---: |
| LiteHRNet-18-256x192 | 134.5 ms | 24.3 ms | 8.6 ms | 8.7 ms |
| LiteHRNet-18-384x288 | 235.4 ms | 24.9 ms | 10.8 ms | 9.2 ms |
| LiteHRNet-30-256x192 | 228.8 ms | 43.3 ms | 11.7 ms | 11.6 ms |
| LiteHRNet-30-384x288 | 390.2 ms | 45.1 ms | 11.8 ms | 12.6 ms |
| PP-TinyPose-128x96 | 51.7 ms | 15.9 ms | 4.4 ms | 4.4 ms |
| PP-TinyPose-256x192 | 115.7 ms | 16.0 ms | 6.2 ms | 5.9 ms |


**Notes:**
- These tests above are based Python deployment.
- The environment is NVIDIA T4 / PaddlePaddle(commit: 7df301f2fc0602745e40fa3a7c43ccedd41786ca) / CUDA10.1 / CUDNN7 / Python3.7 / TensorRT6.
- In test "CPU + MKLDNN", the cpu thread is set to 1 by default.
- The time only includes inference time.


| Model | CPU + MKLDNN | TensorRT (FP32) | TensorRT (FP16) |
| :------------------------ | :------: | :---: | :---: |
| DARK_HRNet_w32-256x192 | 363.93 ms | 3.74 ms | 1.75 ms |
| DARK_HRNet_w32-384x288 | 823.71 ms | 8.91 ms | 2.96 ms |
| HRNet_w32-256x192 | 363.67 ms | 3.71 ms | 1.72 ms |
| HRNet_w32-256x256_mpii | 485.56 ms | 4.26 ms | 2.00 ms |
| HRNet_w32-384x288 | 822.73 ms | 8.81 ms | 2.97 ms |

**Notes:**
- These tests above are based C++ deployment.
- The environment is NVIDIA T4 / PaddlePaddle(commit: 7df301f2fc0602745e40fa3a7c43ccedd41786ca) / CUDA10.1 / CUDNN7 / Python3.7 / TensorRT6.
- In test "CPU + MKLDNN", the cpu thread is set to 1 by default.
- The time only includes inference time.

## Benchmark on Mobile
We tested benchmarks on Kirin and Qualcomm Snapdragon devices. See the table below for details.

| Model | Kirin 980 (1-thread) | Kirin 980 (4-threads)  | Qualcomm Snapdragon 660 (1-thread) | Qualcomm Snapdragon 660 (4-threads) |
| :------------------------ | :---: | :---: | :---: | :---: |
| PicoDet-s-192x192 (det) | 14.85 ms | 5.45 ms | 80.08 ms | 27.36 ms |
| PicoDet-s-320x320 (det) | 38.09 ms | 12.00 ms | 232.81 ms | 58.68 ms |
| PP-TinyPose-128x96 (pose) | 12.03 ms | 5.09 ms | 71.87 ms | 20.04 ms |

**Notes:**
- These tests above are based Paddle Lite deployment.
- The time only includes inference time.
