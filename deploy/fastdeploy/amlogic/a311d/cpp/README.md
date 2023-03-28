[English](README.md) | 简体中文
# PaddleDetection  A311D 量化模型 C++ 部署示例

本目录下提供的 `infer.cc`，可以帮助用户快速完成 PP-YOLOE 量化模型在 A311D 上的部署推理加速。

## 1. 部署环境准备
软硬件环境满足要求，以及交叉编译环境的准备，请参考：[FastDeploy 晶晨 A311d 编译文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#自行编译安装)  

## 2. 部署模型准备
1. 用户可以直接使用由 FastDeploy 提供的量化模型进行部署。
2. 用户可以先使用 PaddleDetection 自行导出 Float32 模型，注意导出模型模型时设置参数：use_shared_conv=False，更多细节请参考：[PP-YOLOE](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe)
3. 用户可以使用 FastDeploy 提供的[一键模型自动化压缩工具](https://github.com/PaddlePaddle/FastDeploy/blob/develop/tools/common_tools/auto_compression/)，自行进行模型量化, 并使用产出的量化模型进行部署。（注意: 推理量化后的检测模型仍然需要FP32模型文件夹下的 infer_cfg.yml 文件，自行量化的模型文件夹内不包含此 yaml 文件，用户从 FP32 模型文件夹下复制此yaml文件到量化后的模型文件夹内即可。）
4. 模型需要异构计算，异构计算文件可以参考：[异构计算](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/heterogeneous_computing_on_timvx_npu.md)，由于 FastDeploy 已经提供了模型，可以先测试我们提供的异构文件，验证精度是否符合要求。

更多量化相关相关信息可查阅[模型量化](../../../quantize/README.md)

## 3. 在 A311D 上部署量化后的 PP-YOLOE  检测模型
请按照以下步骤完成在 A311D 上部署 PP-YOLOE  量化模型：

1. 交叉编译编译 FastDeploy 库，具体请参考：[交叉编译 FastDeploy](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/a311d.md)

2. 将编译后的库拷贝到当前目录，可使用如下命令：
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection/deploy/fastdeploy/amlogic/a311d/cpp
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到develop分支
# git checkout develop
cp -r FastDeploy/build/fastdeploy-timvx/ PaddleDetection/deploy/fastdeploy/amlogic/a311d/cpp
```

3. 在当前路径下载部署所需的模型和示例图片：
```bash
cd PaddleDetection/deploy/fastdeploy/amlogic/a311d/cpp
mkdir models && mkdir images
wget https://bj.bcebos.com/fastdeploy/models/ppyoloe_noshare_qat.tar.gz
tar -xvf ppyoloe_noshare_qat.tar.gz
cp -r ppyoloe_noshare_qat models
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
cp -r 000000014439.jpg images
```

4. 编译部署示例，可使入如下命令：
```bash
cd PaddleDetection/deploy/fastdeploy/amlogic/a311d/cpp
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${PWD}/../fastdeploy-timvx/toolchain.cmake -DFASTDEPLOY_INSTALL_DIR=${PWD}/../fastdeploy-timvx -DTARGET_ABI=arm64 ..
make -j8
make install
# 成功编译之后，会生成 install 文件夹，里面有一个运行 demo 和部署所需的库
```

5. 基于 adb 工具部署 PP-YOLOE  检测模型到晶晨 A311D
```bash
# 进入 install 目录
cd PaddleDetection/deploy/fastdeploy/amlogic/a311d/cpp/build/install/
# 如下命令表示：bash run_with_adb.sh 需要运行的demo 模型路径 图片路径 设备的DEVICE_ID
bash run_with_adb.sh infer_demo ppyoloe_noshare_qat 000000014439.jpg $DEVICE_ID
```

部署成功后运行结果如下：

<img width="640" src="https://user-images.githubusercontent.com/30516196/203708564-43c49485-9b48-4eb2-8fe7-0fa517979fff.png">

需要特别注意的是，在 A311D 上部署的模型需要是量化后的模型，模型的量化请参考：[模型量化](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/quantize.md)

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