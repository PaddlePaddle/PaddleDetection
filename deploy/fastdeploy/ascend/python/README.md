[English](README.md) | 简体中文
# PaddleDetection Ascend Python部署示例

本目录下提供`infer.py`快速完成PPYOLOE在华为昇腾上部署的示例。

## 1. 部署环境准备  
在部署前，需自行编译基于华为昇腾NPU的FastDeploy python wheel包并安装，参考文档[华为昇腾NPU部署环境编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#自行编译安装)

## 2. 部署模型准备  
在部署前，请准备好您所需要运行的推理模型，你可以选择使用[预导出的推理模型](../README.md)或者[自行导出PaddleDetection部署模型](../README.md)。  

## 3. 运行部署示例
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection/deploy/fastdeploy/ascend/python
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
# git checkout develop

# 下载模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz

# 华为昇腾推理
python infer.py --model_dir ppyoloe_crn_l_300e_coco --image_file 000000014439.jpg
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/191712880-91ae128d-247a-43e0-b1e3-cafae78431e0.jpg", width=512px, height=256px />
</div>

## 4. 更多指南
- [PaddleDetection Python API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/object_detection.html)
- [FastDeploy部署PaddleDetection模型概览](../../)
- [C++部署](../cpp)

## 5. 常见问题
- [如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)
- [Intel GPU(独立显卡/集成显卡)的使用](https://github.com/PaddlePaddle/FastDeploy/blob/develop/tutorials/intel_gpu/README.md)
- [编译CPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/cpu.md)
- [编译GPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/gpu.md)
- [编译Jetson部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/jetson.md)