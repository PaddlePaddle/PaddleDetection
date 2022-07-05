# 行为识别任务二次开发

在产业落地过程中应用行为识别算法，不可避免地会出现希望自定义类型的行为识别的需求，或是对已有行为识别模型的优化，以提升在特定场景下模型的效果。鉴于行为的多样性，我们在本文档通过案例来介绍如何根据期望识别的行为来进行行为识别方案的选择，以及使用PaddleDetection进行行为识别算法二次开发工作，包括：数据准备、模型优化思路和新增行为的开发流程。


## 方案选择
在PaddleDetection中，我们为行为识别提供了多种方案：基于视频分类、基于图像分类、基于检测、以及基于骨骼点的行为识别方案，以期望满足不同场景、不同目标行为的需求。对于二次开发，首先我们需要确定要采用何种方案来实现行为识别的需求，其核心是要通过对场景和具体行为的分析、并考虑数据采集成本等因素，综合选择一个合适的识别方案。我们在这里简要列举了当前PaddleDetection中所支持的方案的优劣势和适用场景，供大家参考。

| 技术方案 | 方案说明 | 方案优势 | 方案劣势 | 适用场景 |
| :--: | :--: | :--: | :--: | :--: |
| 基于人体骨骼点的行为识别 | 1. 通过目标检测技术识别出图像中的人；<br> 2. 针对每个人，基于关键点检测技术识别出关键点；<br>3. 基于关键点序列变化识别出具体行为。 | 1. 可识别出每个人的行为；<br>2. 聚焦动作本身，鲁棒性和泛化性好； | 1. 对关键点检测依赖较强，人员较密集或存在遮挡等情况效果不佳；<br>2. 无法准确识别多人交互动作；<br>3. 难以处理需要外观及场景信息的动作；<br>4. 数据收集和标注困难； | 适用于根据人体结构关键点能够区分的行为，背景简单，人数不多场景，如健身场景。 |
| 基于人体id的分类 | 1. 通过目标检测技术得到图像中的人；<br>2. 针对每个人通过图像分类技术得到具体的行为类别。 | 1.通过检测技术可以为分类剔除无关背景的干扰，提升最终识别精度；<br>2. 方案简单，易于训练；<br>3. 数据采集容易；<br>4. 可结合跳帧及结果复用逻辑，速度快； | 1. 缺少时序信息；<br>2. 精度不高； | 对时序信息要求不强的动作，且动作既可通过人也可通过人+物的方式判断，如打电话。 |
| 基于人体id的检测 | 1. 通过目标检测技术得到画面中的人；<br>2. 根据检测结果将人物从原图中抠出，再在扣得的图像中再次用目标检测技术检测与行为强相关的目标。 | 1. 方案简单，易于训练；<br> 2. 可解释性强；<br> 3. 数据采集容易；<br> 4. 可结合跳帧及结果复用逻辑，速度快； | 1. 缺少时序信息；<br>2. 分辨率较低情况下效果不佳；<br> 3. 密集场景容易发生动作误匹配 | 行为与某特定目标强相关的场景，且目标较小，需要两级检测才能准确定位，如吸烟。 |
| 基于视频分类的行为识别 | 应用视频分类技术对整个视频场景进行分类。 | 1.充分利用背景上下文和时序信息；<br>2. 可利用语音、字幕等多模态信息；<br>3. 不依赖检测及跟踪模型；<br>4. 可处理多人共同组成的动作； | 1. 无法定位到具体某个人的行为；<br>2. 场景泛化能力较弱；<br>3.真实数据采集困难； | 无需具体到人的场景的判定，即判断是否存在某种特定行为，多人或对背景依赖较强的动作，如监控画面中打架识别等场景。 |


下面我们以PaddleDetection支持的几个具体动作为例，介绍每个动作是为什么使用现有方案的：

### 吸烟

吸烟动作中具有香烟这个明显特征目标，因此我们可以认为当在某个人物的对应图像中检测到香烟时，该人物即在吸烟动作中。相比于基于视频或基于骨骼点的识别方案，训练检测模型需要采集的是图片级别而非视频级别的数据，可以明显减轻数据收集与标注的难度。此外，目标检测任务具有丰富的预训练模型资源，整体模型的效果会更有保障，

### 打电话

打电话动作中虽然有手机这个特征目标，但为了区分看手机等动作，以及考虑到在安防场景下打电话动作中会出现较多对手机的遮挡（如手对手机的遮挡、人头对手机的遮挡等等），不利于检测模型正确检测到目标。同时打电话通常持续的时间较长，且人物本身的动作不会发生太大变化，因此可以因此采用帧级别图像分类的策略。
    此外，打电话这个动作主要可以通过上半身判别，我们在训练和预测时采用了半身图片，去除冗余信息以降低模型训练的难度。

### 摔倒

摔倒是一个明显的时序行为的动作，可由一个人物本身进行区分，具有场景无关的特性。由于PP-Human的场景定位偏向安防监控场景，背景变化较为复杂，且部署上需要考虑到实时性，因此采用了基于骨骼点的行为识别方案，以获得更好的泛化性及运行速度。

### 打架

与上面的动作不同，打架是一个典型的多人组成的行为（即我们认为不能一个人单独打架）。因此我们不再通过检测与跟踪模型来提取行人及其ID，而是对整体视频片段进行处理。此外，打架场景下各个目标间的互相遮挡极为严重，关键点识别的准确性不高，采用基于骨骼点的方案难以保证精度。


## 数据准备

### 1.基于视频分类的行为识别方案
视频分类任务输入的视频格式一般为`.mp4`、`.avi`等格式视频或者是抽帧后的视频帧序列，标签则可以是`.txt`格式存储的文件。

对于打架识别任务，具体数据准备流程如下：

#### 数据集下载

打架识别基于6个公开的打架、暴力行为相关数据集合并后的数据进行模型训练。公开数据集具体信息如下：

| 数据集 | 下载连接 | 简介 | 标注 | 数量 | 时长 |
| ---- | ---- | ---------- | ---- | ---- | ---------- |
|  Surveillance Camera Fight Dataset| https://github.com/sayibet/fight-detection-surv-dataset | 裁剪视频，监控视角 | 视频级别 | 打架：150；非打架：150 | 2s |
| A Dataset for Automatic Violence Detection in Videos | https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos | 裁剪视频，室内自行录制 | 视频级别 | 暴力行为：115个场景，2个机位，共230 ；非暴力行为：60个场景，2个机位，共120 | 几秒钟 |
| Hockey Fight Detection Dataset | https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes?resource=download | 裁剪视频，非真实场景 | 视频级别 | 打架：500；非打架：500 | 2s |
| Video Fight Detection Dataset | https://www.kaggle.com/datasets/naveenk903/movies-fight-detection-dataset | 裁剪视频，非真实场景 | 视频级别 | 打架：100；非打架：101 | 2s |
| Real Life Violence Situations Dataset | https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset | 裁剪视频，非真实场景 | 视频级别 | 暴力行为：1000；非暴力行为：1000 | 几秒钟 |
| UBI Abnormal Event Detection Dataset| http://socia-lab.di.ubi.pt/EventDetection/ | 未裁剪视频，监控视角 | 帧级别 | 打架：216；非打架：784；裁剪后二次标注：打架1976，非打架1630 | 原视频几秒到几分钟不等，裁剪后2s |

打架（暴力行为）视频3956个，非打架（非暴力行为）视频3501个，共7457个视频，每个视频几秒钟。

#### 视频抽帧

为了加快训练速度，将视频进行抽帧。下面命令会根据视频的帧率FPS进行抽帧，如FPS=30，则每秒视频会抽取30帧图像。

```bash
cd ${PaddleVideo_root}
python data/ucf101/extract_rawframes.py dataset/ rawframes/ --level 2 --ext mp4
```
其中，视频存放在`dataset`目录下，打架（暴力）视频存放在`dataset/fight`中；非打架（非暴力）视频存放在`dataset/nofight`中。`rawframes`目录存放抽取的视频帧。

#### 训练集和验证集划分

打架识别验证集1500条，来自Surveillance Camera Fight Dataset、A Dataset for Automatic Violence Detection in Videos、UBI Abnormal Event Detection Dataset三个数据集。

也可根据下面的命令将数据按照8:2的比例划分成训练集和测试集：

```bash
python split_fight_train_test_dataset.py "rawframes" 2 0.8
```

参数说明：“rawframes”为视频帧存放的文件夹；2表示目录结构为两级，第二级表示每个行为对应的子文件夹；0.8表示训练集比例。

其中`split_fight_train_test_dataset.py`文件在`deploy/pipeline/tools`路径下。

执行完命令后会最终生成fight_train_list.txt和fight_val_list.txt两个文件。打架的标签为1，非打架的标签为0。

#### 视频裁剪
对于未裁剪的视频，需要先进行裁剪才能用于模型训练，`deploy/pipeline/tools/clip_video.py`中给出了视频裁剪的函数`cut_video`，输入为视频路径，裁剪的起始帧和结束帧以及裁剪后的视频保存路径。

### 2.基于图像分类的行为识别方案
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

### 3.基于检测的行为识别方案

基于检测的行为识别方案中，数据准备的流程与一般的检测模型一致，详情可参考[目标检测数据准备](../../tutorials/data/PrepareDetDataSet.md)。将图像和标注数据组织成PaddleDetection中支持的格式之一即可。

### 4.基于骨骼点的行为识别方案

基于骨骼点的行为识别方案是借助[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo)进行模型训练的。使用该方案训练的模型，可以参考[此文档](https://github.com/PaddlePaddle/PaddleVideo/tree/develop/applications/PPHuman#%E5%87%86%E5%A4%87%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE)准备训练数据。其主要流程包含以下步骤：

#### 数据格式说明
STGCN是一个基于骨骼点坐标序列进行预测的模型。在PaddleVideo中，训练数据为采用`.npy`格式存储的`Numpy`数据，标签则可以是`.npy`或`.pkl`格式存储的文件。对于序列数据的维度要求为`(N,C,T,V,M)`。

| 维度 | 大小 | 说明 |
| ---- | ---- | ---------- |
| N | 不定 | 数据集序列个数 |
| C | 2 | 关键点坐标维度，即(x, y) |
| T | 50 | 动作序列的时序维度（即持续帧数）|
| V | 17 | 每个人物关键点的个数，这里我们使用了`COCO`数据集的定义，具体可见[这里](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareKeypointDataSet_cn.md#COCO%E6%95%B0%E6%8D%AE%E9%9B%86) |
| M | 1 | 人物个数，这里我们每个动作序列只针对单人预测 |

#### 获取序列的骨骼点坐标
对于一个待标注的序列（这里序列指一个动作片段，可以是视频或有顺序的图片集合）。可以通过模型预测或人工标注的方式获取骨骼点（也称为关键点）坐标。
- 模型预测：可以直接选用[PaddleDetection KeyPoint模型系列](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/keypoint) 模型库中的模型，并根据`3、训练与测试 - 部署预测 - 检测+keypoint top-down模型联合部署`中的步骤获取目标序列的17个关键点坐标。
- 人工标注：若对关键点的数量或是定义有其他需求，也可以直接人工标注各个关键点的坐标位置，注意对于被遮挡或较难标注的点，仍需要标注一个大致坐标，否则后续网络学习过程会受到影响。

#### 统一序列的时序长度
由于实际数据中每个动作的长度不一，首先需要根据您的数据和实际场景预定时序长度（在PP-Human中我们采用50帧为一个动作序列），并对数据做以下处理：
- 实际长度超过预定长度的数据，随机截取一个50帧的片段
- 实际长度不足预定长度的数据：补0，直到满足50帧
- 恰好等于预定长度的数据： 无需处理

注意：在这一步完成后，请严格确认处理后的数据仍然包含了一个完整的行为动作，不会产生预测上的歧义，建议通过可视化数据的方式进行确认。

#### 保存为PaddleVideo可用的文件格式
在经过前两步处理后，我们得到了每个人物动作片段的标注，此时我们已有一个列表`all_kpts`，这个列表中包含多个关键点序列片段，其中每一个片段形状为(T, V, C) （在我们的例子中即(50, 17, 2)), 下面进一步将其转化为PaddleVideo可用的格式。
- 调整维度顺序： 可通过`np.transpose`和`np.expand_dims`将每一个片段的维度转化为(C, T, V, M)的格式。
- 将所有片段组合并保存为一个文件

注意：这里的`class_id`是`int`类型，与其他分类任务类似。例如`0：摔倒， 1：其他`。

至此，我们得到了可用的训练数据（`.npy`）和对应的标注文件（`.pkl`）。


## 模型优化

### 1. 摔倒--基于关键点的行为识别方案

#### 坐标归一化处理
在完成骨骼点坐标的获取后，建议根据各人物的检测框进行归一化处理，以消除人物位置、尺度的差异给网络带来的收敛难度。


### 2. 打架--基于视频分类的行为识别方案

#### VideoMix
[VideoMix](https://arxiv.org/abs/2012.03457)是视频数据增强的方法之一，是对图像数据增强CutMix的扩展，可以缓解模型的过拟合问题。

与Mixup将两个视频片段的每个像素点按照一定比例融合不同的是，VideoMix是每个像素点要么属于片段A要么属于片段B。输出结果是两个片段原始标签的加权和，权重是两个片段各自的比例。

在baseline的基础上加入VideoMix数据增强后，精度由87.53%提升至88.01%。

#### 更大的分辨率
由于监控摄像头角度、距离等问题，存在监控画面下人比较小的情况，小目标行为的识别较困难，尝试增大输入图像的分辨率，模型精度由88.01%提升至89.06%。


### 3. 打电话--基于图像分类的行为识别方案

#### 半身图预测
在打电话这一动作中，实际是通过上半身就能实现动作的区分的，因此在训练和预测过程中，将图像由行人全身图换为半身图


### 4. 抽烟--基于检测的行为识别方案

#### 更大的分辨率
烟头的检测在监控视角下是一个典型的小目标检测问题，使用更大的分辨率有助于提升模型整体的识别率

#### 预训练模型
加入小目标场景数据集VisDrone下的预训练模型进行训练，模型mAP由38.1提升到39.7。

## 新增行为
当希望新增行为时，首先请参考`方案选择`部分，综合考虑各方案的优劣势及数据采集难易程度，确定希望新增的行为采用何种识别方案进行实现。

### 1. 基于关键点的行为识别方案
基于关键点的行为识别方案中，行为识别模型使用的是[ST-GCN](https://arxiv.org/abs/1801.07455)，并在[PaddleVideo训练流程](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/stgcn.md)的基础上修改适配，完成模型训练及导出使用流程。

#### 模型训练与测试
- 按照`数据准备`, 准备训练数据
- 在PaddleVideo中，使用以下命令即可开始训练：
```bash
# current path is under root of PaddleVideo
python main.py -c applications/PPHuman/configs/stgcn_pphuman.yaml

# 由于整个任务可能过拟合,建议同时开启验证以保存最佳模型
python main.py --validate -c applications/PPHuman/configs/stgcn_pphuman.yaml
```

在训练完成后，采用以下命令进行预测：
```bash
python main.py --test -c applications/PPHuman/configs/stgcn_pphuman.yaml  -w output/STGCN/STGCN_best.pdparams
```

#### 模型导出
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

### 2. 基于视频分类的行为识别方案
目前打架识别模型使用的是[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo)套件中[PP-TSM](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/pp-tsm.md)，并在PP-TSM视频分类模型训练流程的基础上修改适配，完成模型训练。

请先参考[使用说明](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/usage.md)了解PaddleVideo模型库的使用。


| 任务 | 算法 | 精度 | 预测速度(ms) | 模型权重 | 预测部署模型 |
| ---- | ---- | ---------- | ---- | ---- | ---------- |
|  打架识别 | PP-TSM | 准确率：89.06% | T4, 2s视频128ms | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.pdparams) | [下载链接](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.zip) |

#### 模型训练
下载预训练模型：
```bash
wget https://videotag.bj.bcebos.com/PaddleVideo/PretrainModel/ResNet50_vd_ssld_v2_pretrained.pdparams
```

执行训练：
```bash
# 单卡训练
cd ${PaddleVideo_root}
python main.py --validate -c pptsm_fight_frames_dense.yaml
```

本方案针对的是视频的二分类问题，如果不是二分类，需要修改配置文件中`MODEL-->head-->num_classes`为具体的类别数目。


```bash
cd ${PaddleVideo_root}
# 多卡训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -B -m paddle.distributed.launch --gpus=“0,1,2,3” \
   --log_dir=log_pptsm_dense  main.py  --validate \
   -c pptsm_fight_frames_dense.yaml
```

#### 模型评估
训练好的模型下载：[https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.pdparams](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.pdparams)

模型评估：
```bash
cd ${PaddleVideo_root}
python main.py --test -c pptsm_fight_frames_dense.yaml \
   -w ppTSM_fight_best.pdparams
```

其中`ppTSM_fight_best.pdparams`为训练好的模型。

#### 模型导出

导出inference模型：

```bash
cd ${PaddleVideo_root}
python tools/export_model.py -c pptsm_fight_frames_dense.yaml \
                                -p ppTSM_fight_best.pdparams \
                                -o inference/ppTSM
```

### 3.基于图像分类的行为识别方案

#### 模型训练及测试
- 首先根据[Install PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/en/installation/install_paddleclas_en.md)完成PaddleClas的环境配置。
- 按照`数据准备`部分，完成训练/验证集图像的裁剪及标注文件准备。
- 模型训练: 参考[使用预训练模型进行训练](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/quick_start/quick_start_classification_new_user.md#422-%E4%BD%BF%E7%94%A8%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E8%BF%9B%E8%A1%8C%E8%AE%AD%E7%BB%83)完成模型的训练及精度验证

#### 模型导出
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

### 4.基于检测的行为识别方案

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
