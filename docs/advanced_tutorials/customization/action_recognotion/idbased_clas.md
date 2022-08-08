简体中文 | [English](./idbased_clas_en.md)

# 基于人体id的分类模型开发

## 环境准备

基于人体id的分类方案是使用[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)的功能进行模型训练的。请按照[安装说明](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/installation/install_paddleclas.md)完成环境安装，以进行后续的模型训练及使用流程。

## 数据准备

基于图像分类的行为识别方案直接对视频中的图像帧结果进行识别，因此模型训练流程与通常的图像分类模型一致。

### 数据集下载
打电话的行为识别是基于公开数据集[UAV-Human](https://github.com/SUTDCV/UAV-Human)进行训练的。请通过该链接填写相关数据集申请材料后获取下载链接。

在`UAVHuman/ActionRecognition/RGBVideos`路径下包含了该数据集中RGB视频数据集，每个视频的文件名即为其标注信息。

### 训练及测试图像处理
根据视频文件名，其中与行为识别相关的为`A`相关的字段（即action），我们可以找到期望识别的动作类型数据。
- 正样本视频：以打电话为例，我们只需找到包含`A024`的文件。
- 负样本视频：除目标动作以外所有的视频。

鉴于视频数据转化为图像会有较多冗余，对于正样本视频，我们间隔8帧进行采样，并使用行人检测模型处理为半身图像（取检测框的上半部分，即`img = img[:H/2, :, :]`)。正样本视频中的采样得到的图像即视为正样本，负样本视频中采样得到的图像即为负样本。

**注意**: 正样本视频中并不完全符合打电话这一动作，在视频开头结尾部分会出现部分冗余动作，需要移除。

### 标注文件准备

基于图像分类的行为识别方案是借助[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)进行模型训练的。使用该方案训练的模型，需要准备期望识别的图像数据及对应标注文件。根据[PaddleClas数据集格式说明](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/data_preparation/classification_dataset.md#1-%E6%95%B0%E6%8D%AE%E9%9B%86%E6%A0%BC%E5%BC%8F%E8%AF%B4%E6%98%8E)准备对应的数据即可。标注文件样例如下，其中`0`,`1`分别是图片对应所属的类别：
```
    # 每一行采用"空格"分隔图像路径与标注
    train/000001.jpg 0
    train/000002.jpg 0
    train/000003.jpg 1
    ...
```

此外，标签文件`phone_label_list.txt`，帮助将分类序号映射到具体的类型名称：
```
0 make_a_phone_call  # 类型0
1 normal             # 类型1
```

完成上述内容后，放置于`dataset`目录下，文件结构如下：
```
data/
├── images  # 放置所有图片
├── phone_label_list.txt # 标签文件
├── phone_train_list.txt # 训练列表，包含图片及其对应类型
└── phone_val_list.txt   # 测试列表，包含图片及其对应类型
```

## 模型优化

### 检测-跟踪模型优化
基于分类的行为识别模型效果依赖于前序的检测和跟踪效果，如果实际场景中不能准确检测到行人位置，或是难以正确在不同帧之间正确分配人物ID，都会使行为识别部分表现受限。如果在实际使用中遇到了上述问题，请参考[目标检测任务二次开发](../detection.md)以及[多目标跟踪任务二次开发](../pphuman_mot.md)对检测/跟踪模型进行优化。


### 半身图预测
在打电话这一动作中，实际是通过上半身就能实现动作的区分的，因此在训练和预测过程中，将图像由行人全身图换为半身图

## 新增行为

### 数据准备
参考前述介绍的内容，完成数据准备的部分，放置于`{root of PaddleClas}/dataset`下：
```
data/
├── images  # 放置所有图片
├── label_list.txt # 标签文件
├── train_list.txt # 训练列表，包含图片及其对应类型
└── val_list.txt   # 测试列表，包含图片及其对应类型
```
其中，训练及测试列表如下：
```
    # 每一行采用"空格"分隔图像路径与标注
    train/000001.jpg 0
    train/000002.jpg 0
    train/000003.jpg 1
    train/000004.jpg 2   # 新增的类别直接填写对应类别号即可
    ...
```
`label_list.txt`中需要同样对应扩展类型的名称:
```
0 make_a_phone_call  # 类型0
1 Your New Action    # 类型1
 ...
n normal             # 类型n
```

### 配置文件设置
在PaddleClas中已经集成了[训练配置文件](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/configs/practical_models/PPHGNet_tiny_calling_halfbody.yaml)，需要重点关注的设置项如下：

```yaml
# model architecture
Arch:
  name: PPHGNet_tiny
  class_num: 2       # 对应新增后的数量

  ...

# 正确设置image_root与cls_label_path，保证image_root + cls_label_path中的图片路径能够正确访问图片路径
DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      image_root: ./dataset/
      cls_label_path: ./dataset/phone_train_list_halfbody.txt

      ...

Infer:
  infer_imgs: docs/images/inference_deployment/whl_demo.jpg
  batch_size: 1
  transforms:
    - DecodeImage:
        to_rgb: True
        channel_first: False
    - ResizeImage:
        size: 224
    - NormalizeImage:
        scale: 1.0/255.0
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        order: ''
    - ToCHWImage:
  PostProcess:
    name: Topk
    topk: 2                                           # 显示topk的数量，不要超过类别总数
    class_id_map_file: dataset/phone_label_list.txt   # 修改后的label_list.txt路径
```

### 模型训练及评估
#### 模型训练
通过如下命令启动训练：
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/practical_models/PPHGNet_tiny_calling_halfbody.yaml \
        -o Arch.pretrained=True
```
其中 `Arch.pretrained` 为 `True`表示使用预训练权重帮助训练。

#### 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/practical_models/PPHGNet_tiny_calling_halfbody.yaml \
    -o Global.pretrained_model=output/PPHGNet_tiny/best_model
```

其中 `-o Global.pretrained_model="output/PPHGNet_tiny/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。

### 模型导出
模型导出的详细介绍请参考[这里](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/en/inference_deployment/export_model_en.md#2-export-classification-model)
可以参考以下步骤实现：
```python
python tools/export_model.py
    -c ./PPHGNet_tiny_calling_halfbody.yaml \
    -o Global.pretrained_model=./output/PPHGNet_tiny/best_model \
    -o Global.save_inference_dir=./output_inference/PPHGNet_tiny_calling_halfbody
```
然后将导出的模型重命名，并加入配置文件，以适配PP-Human的使用。
```bash
cd ./output_inference/PPHGNet_tiny_calling_halfbody

mv inference.pdiparams model.pdiparams
mv inference.pdiparams.info model.pdiparams.info
mv inference.pdmodel model.pdmodel

# 下载预测配置文件
wget https://bj.bcebos.com/v1/paddledet/models/pipeline/infer_configs/PPHGNet_tiny_calling_halfbody/infer_cfg.yml
```

至此，即可使用PP-Human进行实际预测了。


### 自定义行为输出
基于人体id的分类的行为识别方案中，将任务转化为对应人物的图像进行图片级别的分类。对应分类的类型最终即视为当前阶段的行为。因此在完成自定义模型的训练及部署的基础上，还需要将分类模型结果转化为最终的行为识别结果作为输出，并修改可视化的显示结果。

#### 转换为行为识别结果
请对应修改[后处理函数](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/deploy/pipeline/pphuman/action_infer.py#L509)。

核心代码为：
```python
# 确定分类模型的最高分数输出结果
cls_id_res = 1
cls_score_res = -1.0
for cls_id in range(len(cls_result[idx])):
    score = cls_result[idx][cls_id]
    if score > cls_score_res:
        cls_id_res = cls_id
        cls_score_res = score

# Current now,  class 0 is positive, class 1 is negative.
if cls_id_res == 1 or (cls_id_res == 0 and
                       cls_score_res < self.threshold):
    # 如果分类结果不是目标行为或是置信度未达到阈值，则根据历史结果确定当前帧的行为
    history_cls, life_remain, history_score = self.result_history.get(
        tracker_id, [1, self.frame_life, -1.0])
    cls_id_res = history_cls
    cls_score_res = 1 - cls_score_res
    life_remain -= 1
    if life_remain <= 0 and tracker_id in self.result_history:
        del (self.result_history[tracker_id])
    elif tracker_id in self.result_history:
        self.result_history[tracker_id][1] = life_remain
    else:
        self.result_history[
            tracker_id] = [cls_id_res, life_remain, cls_score_res]
else:
    # 分类结果属于目标行为，则使用将该结果，并记录到历史结果中
    self.result_history[
        tracker_id] = [cls_id_res, self.frame_life, cls_score_res]

    ...
```

#### 修改可视化输出
目前基于ID的行为识别，是根据行为识别的结果及预定义的类别名称进行展示的。详细逻辑请见[此处](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/deploy/pipeline/pipeline.py#L1024-L1043)。如果自定义的行为需要修改为其他的展示名称，请对应修改此处，以正确输出对应结果。
