简体中文 | [English](./skeletonbased_rec_en.md)

# 基于人体骨骼点的行为识别

## 环境准备

基于骨骼点的行为识别方案是借助[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo)进行模型训练的。请按照[安装说明](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/install.md)完成PaddleVideo的环境安装，以进行后续的模型训练及使用流程。

## 数据准备
使用该方案训练的模型，可以参考[此文档](https://github.com/PaddlePaddle/PaddleVideo/tree/develop/applications/PPHuman#%E5%87%86%E5%A4%87%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE)准备训练数据，以适配PaddleVideo进行训练，其主要流程包含以下步骤：


### 数据格式说明
STGCN是一个基于骨骼点坐标序列进行预测的模型。在PaddleVideo中，训练数据为采用`.npy`格式存储的`Numpy`数据，标签则可以是`.npy`或`.pkl`格式存储的文件。对于序列数据的维度要求为`(N,C,T,V,M)`，当前方案仅支持单人构成的行为（但视频中可以存在多人，每个人独自进行行为识别判断），即`M=1`。

| 维度 | 大小 | 说明 |
| ---- | ---- | ---------- |
| N | 不定 | 数据集序列个数 |
| C | 2 | 关键点坐标维度，即(x, y) |
| T | 50 | 动作序列的时序维度（即持续帧数）|
| V | 17 | 每个人物关键点的个数，这里我们使用了`COCO`数据集的定义，具体可见[这里](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareKeypointDataSet_cn.md#COCO%E6%95%B0%E6%8D%AE%E9%9B%86) |
| M | 1 | 人物个数，这里我们每个动作序列只针对单人预测 |

### 获取序列的骨骼点坐标
对于一个待标注的序列（这里序列指一个动作片段，可以是视频或有顺序的图片集合）。可以通过模型预测或人工标注的方式获取骨骼点（也称为关键点）坐标。
- 模型预测：可以直接选用[PaddleDetection KeyPoint模型系列](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/keypoint) 模型库中的模型，并根据`3、训练与测试 - 部署预测 - 检测+keypoint top-down模型联合部署`中的步骤获取目标序列的17个关键点坐标。
- 人工标注：若对关键点的数量或是定义有其他需求，也可以直接人工标注各个关键点的坐标位置，注意对于被遮挡或较难标注的点，仍需要标注一个大致坐标，否则后续网络学习过程会受到影响。


当使用模型预测获取时，可以参考如下步骤进行，请注意此时在PaddleDetection中进行操作。

```bash
# current path is under root of PaddleDetection

# Step 1: download pretrained inference models.
wget https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip
wget https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip
unzip -d output_inference/ mot_ppyoloe_l_36e_pipeline.zip
unzip -d output_inference/ dark_hrnet_w32_256x192.zip

# Step 2: Get the keypoint coordinarys

# if your data is image sequence
python deploy/python/det_keypoint_unite_infer.py --det_model_dir=output_inference/mot_ppyoloe_l_36e_pipeline/ --keypoint_model_dir=output_inference/dark_hrnet_w32_256x192 --image_dir={your image directory path} --device=GPU --save_res=True

# if your data is video
python deploy/python/det_keypoint_unite_infer.py --det_model_dir=output_inference/mot_ppyoloe_l_36e_pipeline/ --keypoint_model_dir=output_inference/dark_hrnet_w32_256x192 --video_file={your video file path} --device=GPU --save_res=True
```
这样我们会得到一个`det_keypoint_unite_image_results.json`的检测结果文件。内容的具体含义请见[这里](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/python/det_keypoint_unite_infer.py#L108)。


### 统一序列的时序长度
由于实际数据中每个动作的长度不一，首先需要根据您的数据和实际场景预定时序长度（在PP-Human中我们采用50帧为一个动作序列），并对数据做以下处理：
- 实际长度超过预定长度的数据，随机截取一个50帧的片段
- 实际长度不足预定长度的数据：补0，直到满足50帧
- 恰好等于预定长度的数据： 无需处理

注意：在这一步完成后，请严格确认处理后的数据仍然包含了一个完整的行为动作，不会产生预测上的歧义，建议通过可视化数据的方式进行确认。

### 保存为PaddleVideo可用的文件格式
在经过前两步处理后，我们得到了每个人物动作片段的标注，此时我们已有一个列表`all_kpts`，这个列表中包含多个关键点序列片段，其中每一个片段形状为(T, V, C) （在我们的例子中即(50, 17, 2)), 下面进一步将其转化为PaddleVideo可用的格式。
- 调整维度顺序： 可通过`np.transpose`和`np.expand_dims`将每一个片段的维度转化为(C, T, V, M)的格式。
- 将所有片段组合并保存为一个文件

注意：这里的`class_id`是`int`类型，与其他分类任务类似。例如`0：摔倒， 1：其他`。


我们提供了执行该步骤的[脚本文件](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/applications/PPHuman/datasets/prepare_dataset.py),可以直接处理生成的`det_keypoint_unite_image_results.json`文件，该脚本执行的内容包括解析json文件内容、前述步骤中介绍的整理训练数据及保存数据文件。

```bash
mkdir {root of PaddleVideo}/applications/PPHuman/datasets/annotations

mv det_keypoint_unite_image_results.json {root of PaddleVideo}/applications/PPHuman/datasets/annotations/det_keypoint_unite_image_results_{video_id}_{camera_id}.json

cd {root of PaddleVideo}/applications/PPHuman/datasets/

python prepare_dataset.py
```

至此，我们得到了可用的训练数据（`.npy`）和对应的标注文件（`.pkl`）。


## 模型优化

### 检测-跟踪模型优化
基于骨骼点的行为识别模型效果依赖于前序的检测和跟踪效果，如果实际场景中不能准确检测到行人位置，或是难以正确在不同帧之间正确分配人物ID，都会使行为识别部分表现受限。如果在实际使用中遇到了上述问题，请参考[目标检测任务二次开发](../detection.md)以及[多目标跟踪任务二次开发](../pphuman_mot.md)对检测/跟踪模型进行优化。

### 关键点模型优化
骨骼点作为该方案的核心特征，对行人的骨骼点定位效果也决定了行为识别的整体效果。若发现在实际场景中对关键点坐标的识别结果有明显错误，从关键点组成的骨架图像看，已经难以辨别具体动作，可以参考[关键点检测任务二次开发](../keypoint_detection.md)对关键点模型进行优化。

### 坐标归一化处理
在完成骨骼点坐标的获取后，建议根据各人物的检测框进行归一化处理，以消除人物位置、尺度的差异给网络带来的收敛难度。


## 新增行为

基于关键点的行为识别方案中，行为识别模型使用的是[ST-GCN](https://arxiv.org/abs/1801.07455)，并在[PaddleVideo训练流程](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/stgcn.md)的基础上修改适配，完成模型训练及导出使用流程。


### 数据准备与配置文件修改
- 按照`数据准备`, 准备训练数据（`.npy`）和对应的标注文件（`.pkl`）。对应放置在`{root of PaddleVideo}/applications/PPHuman/datasets/`下。

- 参考[配置文件](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/applications/PPHuman/configs/stgcn_pphuman.yaml), 需要重点关注的内容如下：

```yaml
MODEL: #MODEL field
    framework:
        backbone:
        name: "STGCN"
        in_channels: 2  # 此处对应数据说明中的C维，表示二维坐标。
        dropout: 0.5
        layout: 'coco_keypoint'
        data_bn: True
    head:
        name: "STGCNHead"
        num_classes: 2  # 如果数据中有多种行为类型，需要修改此处使其与预测类型数目一致。
    if_top5: False # 行为类型数量不足5时请设置为False，否则会报错

...


# 请根据数据路径正确设置train/valid/test部分的数据及label路径
DATASET: #DATASET field
    batch_size: 64
    num_workers: 4
    test_batch_size: 1
    test_num_workers: 0
    train:
        format: "SkeletonDataset" #Mandatory, indicate the type of dataset, associate to the 'paddle
        file_path: "./applications/PPHuman/datasets/train_data.npy" #mandatory, train data index file path
        label_path: "./applications/PPHuman/datasets/train_label.pkl"

    valid:
        format: "SkeletonDataset" #Mandatory, indicate the type of dataset, associate to the 'paddlevideo/loader/dateset'
        file_path: "./applications/PPHuman/datasets/val_data.npy" #Mandatory, valid data index file path
        label_path: "./applications/PPHuman/datasets/val_label.pkl"

        test_mode: True
    test:
        format: "SkeletonDataset" #Mandatory, indicate the type of dataset, associate to the 'paddlevideo/loader/dateset'
        file_path: "./applications/PPHuman/datasets/val_data.npy" #Mandatory, valid data index file path
        label_path: "./applications/PPHuman/datasets/val_label.pkl"

        test_mode: True
```

### 模型训练与测试
- 在PaddleVideo中，使用以下命令即可开始训练：

```bash
# current path is under root of PaddleVideo
python main.py -c applications/PPHuman/configs/stgcn_pphuman.yaml

# 由于整个任务可能过拟合,建议同时开启验证以保存最佳模型
python main.py --validate -c applications/PPHuman/configs/stgcn_pphuman.yaml
```

- 在训练完成后，采用以下命令进行预测：
```bash
python main.py --test -c applications/PPHuman/configs/stgcn_pphuman.yaml  -w output/STGCN/STGCN_best.pdparams
```

### 模型导出
- 在PaddleVideo中，通过以下命令实现模型的导出，得到模型结构文件`STGCN.pdmodel`和模型权重文件`STGCN.pdiparams`，并增加配置文件：
```bash
# current path is under root of PaddleVideo
python tools/export_model.py -c applications/PPHuman/configs/stgcn_pphuman.yaml \
                                -p output/STGCN/STGCN_best.pdparams \
                                -o output_inference/STGCN

cp applications/PPHuman/configs/infer_cfg.yml output_inference/STGCN

# 重命名模型文件，适配PP-Human的调用
cd output_inference/STGCN
mv STGCN.pdiparams model.pdiparams
mv STGCN.pdiparams.info model.pdiparams.info
mv STGCN.pdmodel model.pdmodel
```
完成后的导出模型目录结构如下：
```
STGCN
├── infer_cfg.yml
├── model.pdiparams
├── model.pdiparams.info
├── model.pdmodel
```

至此，就可以使用PP-Human进行行为识别的推理了。

**注意**：如果在训练时调整了视频序列的长度或关键点的数量，在此处需要对应修改配置文件中`INFERENCE`字段内容，以实现正确预测。
```yaml
# 序列数据的维度为(N,C,T,V,M)
INFERENCE:
    name: 'STGCN_Inference_helper'
    num_channels: 2 # 对应C维
    window_size: 50 # 对应T维，请对应调整为数据长度
    vertex_nums: 17 # 对应V维，请对应调整为关键点数目
    person_nums: 1 # 对应M维
```

### 自定义行为输出
基于人体骨骼点的行为识别方案中，模型输出的分类结果即代表了该人物在一定时间段内行为类型。对应分类的类型最终即视为当前阶段的行为。因此在完成自定义模型的训练及部署的基础上，使用模型输出作为最终结果，修改可视化的显示结果即可。

#### 修改可视化输出
目前基于ID的行为识别，是根据行为识别的结果及预定义的类别名称进行展示的。详细逻辑请见[此处](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/pipeline/pipeline.py#L1024-L1043)。如果自定义的行为需要修改为其他的展示名称，请对应修改此处，以正确输出对应结果。
