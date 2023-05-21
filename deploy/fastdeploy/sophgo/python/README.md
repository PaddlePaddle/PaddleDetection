# PaddleDetection Python部署示例

## 1. 部署环境准备

在部署前，需自行编译基于算能硬件的FastDeploy python wheel包并安装，参考文档[算能硬件部署环境](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#算能硬件部署环境)

本目录下提供`infer.py`, 快速完成 PP-YOLOE ,在SOPHGO TPU上部署的示例，执行如下脚本即可完成。PP-YOLOV8和 PicoDet的部署逻辑类似，只需要切换模型即可。 

## 2. 部署模型准备  
在部署前，请准备好您所需要运行的推理模型，你可以选择使用[预导出的推理模型](../README.md)或者[自行导出PaddleDetection部署模型](../README.md)。

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection/deploy/fastdeploy/sophgo/python

# 下载图片
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# 推理
#ppyoloe推理示例
python3 infer.py --model_file model/ppyoloe_crn_s_300e_coco_1684x_f32.bmodel --config_file model/infer_cfg.yml --image_file ./000000014439.jpg

# 运行完成后返回结果如下所示
可视化结果存储在sophgo_result.jpg中
```

## 3. 更多指南
- [C++部署](../cpp)
- [转换PP-YOLOE SOPHGO模型文档](../README.md)
