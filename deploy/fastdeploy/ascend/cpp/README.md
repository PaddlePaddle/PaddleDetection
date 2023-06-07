[English](README.md) | 简体中文
# PaddleDetection Ascend C++部署示例

本目录下提供`infer.cc`快速完成PPYOLOE在华为昇腾上部署的示例。

## 1. 部署环境准备
在部署前，需自行编译基于华为昇腾NPU的预测库，参考文档[华为昇腾NPU部署环境编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#自行编译安装)

## 2. 部署模型准备  
在部署前，请准备好您所需要运行的推理模型，你可以选择使用[预导出的推理模型](../README.md)或者[自行导出PaddleDetection部署模型](../README.md)。  

## 3. 运行部署示例  
以Linux上推理为例，在本目录执行如下命令即可完成编译测试。
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection/deploy/fastdeploy/cpu-gpu/cpp/ascend/cpp
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
# git checkout develop

mkdir build
cd build
# 使用编译完成的FastDeploy库编译infer_demo
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-ascend
make -j

# 下载模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz

# 华为昇腾推理
./infer_demo ./ppyoloe_crn_l_300e_coco 000000014439.jpg
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/19339784/184326520-7075e907-10ed-4fad-93f8-52d0e35d4964.jpg", width=480px, height=320px />
</div> 


## 4. 更多指南
- [PaddleDetection C++ API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1detection.html)
- [FastDeploy部署PaddleDetection模型概览](../../)
- [Python部署](../python)

## 5. 常见问题
- [如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)
- [Intel GPU(独立显卡/集成显卡)的使用](https://github.com/PaddlePaddle/FastDeploy/blob/develop/tutorials/intel_gpu/README.md)
- [编译CPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/cpu.md)
- [编译GPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/gpu.md)
- [编译Jetson部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/jetson.md)