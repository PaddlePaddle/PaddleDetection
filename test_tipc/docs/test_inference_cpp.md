# C++预测功能测试

C++预测功能测试的主程序为`test_inference_cpp.sh`，可以测试基于C++预测库的模型推理功能。

## 1. 测试结论汇总

基于训练是否使用量化，进行本测试的模型可以分为`正常模型`和`量化模型`，这两类模型对应的C++预测功能汇总如下：

| 模型类型 |device | batchsize | tensorrt | mkldnn | cpu多线程 |
|  ----   |  ---- |   ----   |  :----:  |   :----:   |  :----:  |
| 正常模型 | GPU | 1/8 | fp32/fp16 | - | - |
| 正常模型 | CPU | 1/8 | - | fp32 | 支持 |
| 量化模型 | GPU | 1/8 | int8 | - | - |
| 量化模型 | CPU | 1/8 | - | int8 | 支持 |

## 2. 测试流程
运行环境配置请参考[文档](./install.md)的内容配置TIPC的运行环境。
```
# 请设置paddle_inference环境变量，如：
export PADDLE_INFER_DIR=/path/to/paddle_inference
# 若不设置paddle_inference环境变量，也可通过指定参数的方式使脚本自动下载paddle_inference.tgz，如：
bash test_tipc/test_inference_cpp.sh test_tipc/configs/yolov3/yolov3_darknet53_270e_coco_model_linux_gpu_normal_normal_infer_cpp_linux_gpu_cpu.txt 'https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference.tgz'

# 若未使用docker镜像: paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7-gcc82-dev
# 请设置TensorRT环境变量，如：
export TENSORRT_ROOT=/usr/local/TensorRT6-cuda10.1-cudnn7
```

### 2.1 功能测试
先运行`prepare.sh`准备数据和模型，然后运行`test_inference_cpp.sh`进行测试，最终在```test_tipc/output```目录下生成`cpp_infer_*.log`后缀的日志文件。

```shell
bash test_tipc/prepare.sh ./test_tipc/configs/yolov3/yolov3_darknet53_270e_coco_model_linux_gpu_normal_normal_infer_cpp_linux_gpu_cpu.txt "cpp_infer"
# 用法1:
bash test_tipc/test_inference_cpp.sh test_tipc/configs/yolov3/yolov3_darknet53_270e_coco_model_linux_gpu_normal_normal_infer_cpp_linux_gpu_cpu.txt
# 用法2: 指定下载paddle_inference.tgz链接，第二个传入参数为下载链接
bash test_tipc/test_inference_cpp.sh test_tipc/configs/yolov3/yolov3_darknet53_270e_coco_model_linux_gpu_normal_normal_infer_cpp_linux_gpu_cpu.txt 'https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference.tgz'
# 用法3: 同时指定下载paddle_inference.tgz链接和指定GPU卡预测，第三个传入参数为GPU卡号
bash test_tipc/test_inference_cpp.sh test_tipc/configs/yolov3/yolov3_darknet53_270e_coco_model_linux_gpu_normal_normal_infer_cpp_linux_gpu_cpu.txt 'https://paddle-inference-lib.bj.bcebos.com/2.3.0/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddle_inference.tgz' '1'
```  

运行预测指令后，在`test_tipc/output`文件夹下自动会保存运行日志，包括以下文件：

```shell
test_tipc/output/
|- results_cpp.log    # 运行指令状态的日志
|- cpp_infer_cpu_usemkldnn_False_threads_1_precision_fluid_batchsize_1.log  # CPU上不开启Mkldnn，线程数设置为1，测试batch_size=1条件下的预测运行日志
|- cpp_infer_cpu_usemkldnn_False_threads_6_precision_fluid_batchsize_1.log  # CPU上不开启Mkldnn，线程数设置为6，测试batch_size=1条件下的预测运行日志
|- cpp_infer_gpu_precision_fluid_batchsize_1.log # GPU上不开启TensorRT，测试batch_size=1的fp32精度预测日志
|- cpp_infer_gpu_precision_trt_fp16_batchsize_1.log  # GPU上开启TensorRT，测试batch_size=1的fp16精度预测日志
......
```
其中results_cpp.log中包含了每条指令的运行状态，如果运行成功会输出：

```
Run successfully with command - python3.7 tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams filename=yolov3_darknet53_270e_coco --output_dir=./output_inference !
Run successfully with command - ./deploy/cpp/build/main --device=gpu --run_mode=fluid --model_dir=./output_inference/yolov3_darknet53_270e_coco --batch_size=8 --image_dir=./dataset/coco/test2017/ --run_benchmark=True   > ./test_tipc/output/cpp_infer_gpu_precision_fluid_batchsize_8.log 2>&1 !
......
```
如果运行失败，会输出：
```
Run failed with command - python3.7 tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams filename=yolov3_darknet53_270e_coco --output_dir=./output_inference !
Run failed with command - ./deploy/cpp/build/main --device=gpu --run_mode=fluid --model_dir=./output_inference/yolov3_darknet53_270e_coco --batch_size=8 --image_dir=./dataset/coco/test2017/ --run_benchmark=True   > ./test_tipc/output/cpp_infer_gpu_precision_fluid_batchsize_8.log 2>&1 !
......
```
可以很方便的根据results_cpp.log中的内容判定哪一个指令运行错误。


### 2.2 精度测试

使用compare_results.py脚本比较模型预测的结果是否符合预期，主要步骤包括：
- 提取日志中的预测坐标；
- 从本地文件中提取保存好的坐标结果；
- 比较上述两个结果是否符合精度预期，误差大于设置阈值时会报错。

#### 使用方式
运行命令：
```shell
python3.7 test_tipc/compare_results.py --gt_file=./test_tipc/results/cpp_*.txt  --log_file=./test_tipc/output/cpp_*.log --atol=1e-3 --rtol=1e-3
```

参数介绍：  
- gt_file： 指向事先保存好的预测结果路径，支持*.txt 结尾，会自动索引*.txt格式的文件，文件默认保存在test_tipc/result/ 文件夹下
- log_file: 指向运行test_tipc/test_inference_cpp.sh 脚本的infer模式保存的预测日志，预测日志中打印的有预测结果，比如：文本框，预测文本，类别等等，同样支持cpp_infer_*.log格式传入
- atol: 设置的绝对误差
- rtol: 设置的相对误差

#### 运行结果

正常运行效果如下图：
<img src="compare_cpp_right.png" width="1000">

出现不一致结果时的运行输出：
<img src="compare_cpp_wrong.png" width="1000">


## 3. 更多教程

本文档为功能测试用，更详细的c++预测使用教程请参考：[C++预测](../../deploy/cpp/README.md)
