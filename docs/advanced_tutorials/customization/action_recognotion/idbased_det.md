# 基于人体id的检测开发

## 数据准备

基于检测的行为识别方案中，数据准备的流程与一般的检测模型一致，详情可参考[目标检测数据准备](../../tutorials/data/PrepareDetDataSet.md)。将图像和标注数据组织成PaddleDetection中支持的格式之一即可。

## 模型优化

### 更大的分辨率
烟头的检测在监控视角下是一个典型的小目标检测问题，使用更大的分辨率有助于提升模型整体的识别率

### 预训练模型
加入小目标场景数据集VisDrone下的预训练模型进行训练，模型mAP由38.1提升到39.7。

## 新增行为

#### 模型训练及测试
- 按照`数据准备`部分，完成训练/验证集图像的裁剪及标注文件准备。
- 模型训练: 参考[PP-YOLOE](../../../configs/ppyoloe/README_cn.md)，执行下列步骤实现

```bash
python -m paddle.distributed.launch --gpus 0,1,2,3  tools/train.py -c ppyoloe_smoking/ppyoloe_crn_s_80e_smoking_visdrone.yml --eval
```

#### 模型导出
注意：如果在Tensor-RT环境下预测, 请开启`-o trt=True`以获得更好的性能
```bash
python tools/export_model.py -c ppyoloe_smoking/ppyoloe_crn_s_80e_smoking_visdrone.yml -o weights=output/ppyoloe_crn_s_80e_smoking_visdrone/best_model trt=True
```
至此，即可使用PP-Human进行实际预测了。
