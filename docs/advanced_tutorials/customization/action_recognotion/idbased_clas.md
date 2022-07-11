# 基于人体id的分类

## 数据准备

基于图像分类的行为识别方案直接对视频中的图像帧结果进行识别，因此模型训练流程与通常的图像分类模型一致。

#### 数据集下载
打电话的行为识别是基于公开数据集[UAV-Human](https://github.com/SUTDCV/UAV-Human)进行训练的。请通过该链接填写相关数据集申请材料后获取下载链接。

在`UAVHuman/ActionRecognition/RGBVideos`路径下包含了该数据集中RGB视频数据集，每个视频的文件名即为其标注信息。

#### 训练及测试图像处理
根据视频文件名，其中与行为识别相关的为`A`相关的字段（即action），我们可以找到期望识别的动作类型数据。
- 正样本视频：以打电话为例，我们只需找到包含`A024`的文件。
- 负样本视频：除目标动作以外所有的视频。

鉴于视频数据转化为图像会有较多冗余，对于正样本视频，我们间隔8帧进行采样，并使用行人检测模型处理为半身图像（取检测框的上半部分，即`img = img[:H/2, :, :]`)。正样本视频中的采样得到的图像即视为正样本，负样本视频中采样得到的图像即为负样本。

**注意**: 正样本视频中并不完全符合打电话这一动作，在视频开头结尾部分会出现部分冗余动作，需要移除。

#### 标注文件准备

基于图像分类的行为识别方案是借助[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)进行模型训练的。使用该方案训练的模型，需要准备期望识别的图像数据及对应标注文件。根据[PaddleClas数据集格式说明](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/data_preparation/classification_dataset.md#1-%E6%95%B0%E6%8D%AE%E9%9B%86%E6%A0%BC%E5%BC%8F%E8%AF%B4%E6%98%8E)准备对应的数据即可。标注文件样例如下，其中`0`,`1`分别是图片对应所属的类别：
```
    # 每一行采用"空格"分隔图像路径与标注
    train/000001.jpg 0
    train/000002.jpg 0
    train/000003.jpg 1
    ...
```

## 模型优化

### 半身图预测
在打电话这一动作中，实际是通过上半身就能实现动作的区分的，因此在训练和预测过程中，将图像由行人全身图换为半身图

## 新增行为

### 模型训练及测试
- 首先根据[Install PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/en/installation/install_paddleclas_en.md)完成PaddleClas的环境配置。
- 按照`数据准备`部分，完成训练/验证集图像的裁剪及标注文件准备。
- 模型训练: 参考[使用预训练模型进行训练](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/quick_start/quick_start_classification_new_user.md#422-%E4%BD%BF%E7%94%A8%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E8%BF%9B%E8%A1%8C%E8%AE%AD%E7%BB%83)完成模型的训练及精度验证

### 模型导出
模型导出的详细介绍请参考[这里](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/en/inference_deployment/export_model_en.md#2-export-classification-model)
可以参考以下步骤实现：
```python
python tools/export_model.py
    -c ./PPHGNet_tiny_resize_halfbody.yaml \
    -o Global.pretrained_model=./output/PPHGNet_tiny/best_model \
    -o Global.save_inference_dir=./output_inference/PPHGNet_tiny_resize_halfbody
```
然后将导出的模型重命名，并加入配置文件，以适配PP-Human的使用
```bash
cd ./output_inference/PPHGNet_tiny_resize_halfbody

mv inference.pdiparams model.pdiparams
mv inference.pdiparams.info model.pdiparams.info
mv inference.pdmodel model.pdmodel

cp infer_cfg.yml .
```
至此，即可使用PP-Human进行实际预测了。
