# PaddleDetection

动态图版本的PaddleDetection, 此版本为试用版本，还在持续优化设计、性能、新增模型、文档等。


支持的模型:

- Faster-RCNN (FPN)
- Mask-RCNN (FPN)
- Cascade RCNN
- YOLOv3
- SSD
- SOLOv2

扩展特性：

- [x] **Synchronized Batch Norm**
- [x] **Group Norm**
- [x] **Modulated Deformable Convolution**
- [x] **Deformable PSRoI Pooling**

## 文档教程

### 教程

- [安装说明](docs/tutorials/INSTALL_cn.md)
- [训练/评估/预测流程](docs/tutorials/GETTING_STARTED_cn.md)
- [常见问题汇总](docs/FAQ.md)
- [推理部署](deploy)
    - [模型导出教程](docs/advanced_tutorials/deploy/EXPORT_MODEL.md)
    - [Python端推理部署](deploy/python)
    - [C++端推理部署](deploy/cpp)
    - [推理Benchmark](docs/advanced_tutorials/deploy/BENCHMARK_INFER_cn.md)

## 模型库
- [模型库](docs/MODEL_ZOO_cn.md)
