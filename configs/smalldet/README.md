# PP-YOLOE-SOD 小目标检测模型(PP-YOLOE Small Object Detection)

<img src="https://user-images.githubusercontent.com/82303451/182520025-f6bd1c76-a9f9-4f8c-af9b-b37a403258d8.png" title="VisDrone" alt="VisDrone" width="300"><img src="https://user-images.githubusercontent.com/82303451/182521833-4aa0314c-b3f2-4711-9a65-cabece612737.png" title="VisDrone" alt="VisDrone" width="300"><img src="https://user-images.githubusercontent.com/82303451/182520038-cacd5d09-0b85-475c-8e59-72f1fc48eef8.png" title="DOTA" alt="DOTA" height="168"><img src="https://user-images.githubusercontent.com/82303451/182524123-dcba55a2-ce2d-4ba1-9d5b-eb99cb440715.jpeg" title="Xview" alt="Xview" height="168">

## 内容
- [简介](#简介)
- [切图使用说明](#切图使用说明)
    - [小目标数据集下载](#小目标数据集下载)
    - [统计数据集分布](#统计数据集分布)
    - [SAHI切图](#SAHI切图)
- [模型库](#模型库)
    - [VisDrone模型](#VisDrone模型)
    - [COCO模型](#COCO模型)
    - [切图模型](#切图模型)
    - [拼图模型](#拼图模型)
    - [注意事项](#注意事项)
- [模型库使用说明](#模型库使用说明)
    - [训练](#训练)
    - [评估](#评估)
    - [预测](#预测)
    - [部署](#部署)
- [引用](#引用)


## 简介
PaddleDetection团队提供了针对VisDrone-DET、DOTA水平框、Xview等小目标场景数据集的基于PP-YOLOE改进的检测模型 PP-YOLOE-SOD，以及提供了一套使用[SAHI](https://github.com/obss/sahi)(Slicing Aided Hyper Inference)工具的切图和拼图的方案。

  - PP-YOLOE-SOD 是PaddleDetection团队自研的小目标检测特色模型，使用**数据集分布相关的基于向量的DFL算法** 和 **针对小目标优化的中心先验优化策略**，并且**在模型的Neck(FPN)结构中加入Transformer模块**，以及结合增加P2层、使用large size等策略，最终在多个小目标数据集上达到极高的精度。

  - 切图拼图方案**适用于任何检测模型**，建议**使用 PP-YOLOE-SOD 结合切图拼图方案**一起使用以达到最佳的效果。

  - 官方 AI Studio 教程案例请参考 [基于PP-YOLOE-SOD的无人机航拍图像检测案例全流程实操](https://aistudio.baidu.com/aistudio/projectdetail/5036782)，欢迎一起动手实践学习。

  - 第三方 AI Studio 教程案例可参考 [PPYOLOE：遥感场景下的小目标检测与部署(切图版)](https://aistudio.baidu.com/aistudio/projectdetail/4493701) 和 [涨分神器！基于PPYOLOE的切图和拼图解决方案](https://aistudio.baidu.com/aistudio/projectdetail/4438275)，欢迎一起动手实践学习。

**注意:**
 - **不通过切图拼图而直接使用原图或子图**去训练评估预测，推荐使用 PP-YOLOE-SOD 模型，更多细节和消融实验可参照[COCO模型](#COCO模型)和[VisDrone模型](./visdrone)。
 - 是否需要切图然后使用子图去**训练**，建议首先参照[切图使用说明](#切图使用说明)中的[统计数据集分布](#统计数据集分布)分析一下数据集再确定，一般数据集中**所有的目标均极小**的情况下推荐切图去训练。
 - 是否需要切图然后使用子图去**预测**，建议在切图训练的情况下，配合着**同样操作的切图策略和参数**去预测(inference)效果更佳。但其实即便不切图训练，也可进行切图预测(inference)，只需**在常规的预测命令最后加上`--slice_infer`以及相关子图参数**即可。
 - 是否需要切图然后使用子图去**评估**，建议首先确保制作生成了合适的子图验证集，以及确保对应的标注框制作无误，并需要参照[模型库使用说明-评估](#评估)去**改动配置文件中的验证集(EvalDataset)的相关配置**，然后**在常规的评估命令最后加上`--slice_infer`以及相关子图参数**即可。
 - `--slice_infer`的操作在PaddleDetection中默认**子图预测框会自动组合并拼回原图**，默认返回的是原图上的预测框，此方法也**适用于任何训好的检测模型**，无论是否切图训练。


## 切图使用说明

### 小目标数据集下载
PaddleDetection团队整理提供的VisDrone-DET、DOTA水平框、Xview等小目标场景数据集的下载链接可以参照 [DataDownload.md](./DataDownload.md)。

### 统计数据集分布

对于待训的数据集(默认已处理为COCO格式，参照 [COCO格式数据集准备](../../docs/tutorials/data/PrepareDetDataSet.md#用户数据转成COCO数据)，首先统计**标注框的平均宽高占图片真实宽高的比例**分布：

以DOTA水平框数据集的train数据集为例：

```bash
python tools/box_distribution.py --json_path dataset/DOTA/annotations/train.json --out_img box_distribution.jpg --eval_size 640 --small_stride 8
```
  - `--json_path` ：待统计数据集 COCO 格式 annotation 的json标注文件路径
  - `--out_img` ：输出的统计分布图的路径
  - `--eval_size` ：推理尺度（默认640）
  - `--small_stride` ：模型最小步长（默认8）

统计结果打印如下：
```bash
Suggested reg_range[1] is 13 # DFL算法中推荐值，在 PP-YOLOE-SOD 模型的配置文件的head中设置为此值，效果最佳
Mean of all img_w is 2304.3981547196595 # 原图宽的平均值
Mean of all img_h is 2180.9354151880766 # 原图高的平均值
Median of ratio_w is 0.03799439775910364 # 标注框的宽与原图宽的比例的中位数
Median of ratio_h is 0.04074914637387802 # 标注框的高与原图高的比例的中位数
all_img with box:  1409 # 数据集图片总数(排除无框或空标注的图片)
all_ann:  98905 # 数据集标注框总数
Distribution saved as box_distribution.jpg
```

**注意:**
- 一般情况下，在原始数据集全部有标注框的图片中，**原图宽高的平均值大于1500像素，且有1/2以上的图片标注框的平均宽高与原图宽高比例小于0.04时(通过打印中位数得到该值)**，建议进行切图训练。
- `Suggested reg_range[1]` 为数据集在优化后DFL算法中推荐的`reg_range`上限，即`reg_max + 1`，在 PP-YOLOE-SOD 模型的配置文件的head中设置这个值。


### SAHI切图

针对需要切图的数据集，使用[SAHI](https://github.com/obss/sahi)库进行切图：

#### 安装SAHI库：

参考[SAHI installation](https://github.com/obss/sahi/blob/main/README.md#installation)进行安装，`pip install sahi`，参考[installation](https://github.com/obss/sahi/blob/main/README.md#installation)。

#### 基于SAHI切图

以DOTA水平框数据集的train数据集为例，切分后的**子图文件夹**与**子图json标注文件**共同保存在`dota_sliced`文件夹下，分别命名为`train_images_500_025`、`train_500_025.json`：

```bash
python tools/slice_image.py --image_dir dataset/DOTA/train/ --json_path dataset/DOTA/annotations/train.json --output_dir dataset/dota_sliced --slice_size 500 --overlap_ratio 0.25
```
  - `--image_dir`：原始数据集图片文件夹的路径
  - `--json_path`：原始数据集COCO格式的json标注文件的路径
  - `--output_dir`：切分后的子图及其json标注文件保存的路径
  - `--slice_size`：切分以后子图的边长尺度大小(默认切图后为正方形)
  - `--overlap_ratio`：切分时的子图之间的重叠率

**注意:**
- 如果切图然后使用子图去**训练**，则只能**离线切图**，即切完图后保存成子图，存放在内存空间中。
- 如果切图然后使用子图去**评估或预测**，则既可以**离线切图**，也可以**在线切图**，PaddleDetection中支持切图并自动拼图组合结果到原图上。


## 模型库

### [VisDrone模型](visdrone/)

|    模型   | COCOAPI mAP<sup>val<br>0.5:0.95 | COCOAPI mAP<sup>val<br>0.5 | COCOAPI mAP<sup>test_dev<br>0.5:0.95 | COCOAPI mAP<sup>test_dev<br>0.5 | MatlabAPI mAP<sup>test_dev<br>0.5:0.95 | MatlabAPI mAP<sup>test_dev<br>0.5 | 下载  | 配置文件 |
|:---------|:------:|:------:| :----: | :------:| :------: | :------:| :----: | :------:|
|PP-YOLOE-s|  23.5  |  39.9  |  19.4  |  33.6   |  23.68   |  40.66  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_80e_visdrone.pdparams) | [配置文件](visdrone/ppyoloe_crn_s_80e_visdrone.yml) |
|PP-YOLOE-P2-Alpha-s|    24.4  |  41.6  |  20.1  |  34.7  |  24.55   |  42.19  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_p2_alpha_80e_visdrone.pdparams) | [配置文件](visdrone/ppyoloe_crn_s_p2_alpha_80e_visdrone.yml) |
|**PP-YOLOE+_SOD-s**|  **25.1**  |  **42.8**  |  **20.7**  |  **36.2**   |  **25.16**  |  **43.86**   | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_sod_crn_s_80e_visdrone.pdparams) | [配置文件](visdrone/ppyoloe_plus_sod_crn_s_80e_visdrone.yml) |
|PP-YOLOE-l|  29.2  |  47.3  |  23.5  |  39.1   |  28.00   |  46.20  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_visdrone.pdparams) | [配置文件](visdrone/ppyoloe_crn_l_80e_visdrone.yml) |
|PP-YOLOE-P2-Alpha-l|  30.1  |  48.9  |  24.3  |  40.8   |  28.47   |  48.16  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_p2_alpha_80e_visdrone.pdparams) | [配置文件](visdrone/ppyoloe_crn_l_p2_alpha_80e_visdrone.yml) |
|**PP-YOLOE+_SOD-l**|  **31.9**  |  **52.1**  |  **25.6**  |  **43.5**   |  **30.25**  |  **51.18**   | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_sod_crn_l_80e_visdrone.pdparams) | [配置文件](visdrone/ppyoloe_plus_sod_crn_l_80e_visdrone.yml) |
|PP-YOLOE-Alpha-largesize-l|  41.9  |  65.0 |  32.3  |  53.0   |  37.13   |  61.15  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_alpha_largesize_80e_visdrone.pdparams) | [配置文件](visdrone/ppyoloe_crn_l_alpha_largesize_80e_visdrone.yml) |
|PP-YOLOE-P2-Alpha-largesize-l|  41.3  |  64.5  |  32.4  |  53.1   |  37.49   |  51.54  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_p2_alpha_largesize_80e_visdrone.pdparams) | [配置文件](visdrone/ppyoloe_crn_l_p2_alpha_largesize_80e_visdrone.yml) |
|PP-YOLOE+_largesize-l |  43.3  |  66.7 |  33.5  |  54.7   |  38.24   |  62.76  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_largesize_80e_visdrone.pdparams) | [配置文件](visdrone/ppyoloe_plus_crn_l_largesize_80e_visdrone.yml) |
|**PP-YOLOE+_SOD-largesize-l** |  42.7  |  65.9 |  **33.6**  |  **55.1**   |  **38.4**   |  **63.07**  | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone.pdparams) | [配置文件](visdrone/ppyoloe_plus_sod_crn_l_largesize_80e_visdrone.yml) |

**注意:**
  - 上表中的模型均为**使用原图训练**，也**使用原图评估预测**，AP精度均为**原图验证集**上评估的结果。
  - VisDrone-DET数据集**可使用原图训练，也可使用切图后训练**，通过数据集统计分布分析，推荐使用**原图训练**，推荐直接使用带**SOD**的模型配置文件去训练评估和预测部署，在显卡算力有限时也可使用切图后训练。
  - 上表中的模型指标均是使用VisDrone-DET的train子集作为训练集，使用VisDrone-DET的val子集和test_dev子集作为验证集。
  - **SOD**表示使用**基于向量的DFL算法**和针对小目标的**中心先验优化策略**，并**在模型的Neck结构中加入transformer**。
  - **P2**表示增加P2层(1/4下采样层)的特征，共输出4个PPYOLOEHead。
  - **Alpha**表示对CSPResNet骨干网络增加可一个学习权重参数Alpha参与训练。
  - **largesize**表示使用**以1600尺度为基础的多尺度训练**和**1920尺度预测**，相应的训练batch_size也减小，以速度来换取高精度。
  - MatlabAPI测试是使用官网评测工具[VisDrone2018-DET-toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit)。

<details>
<summary> 快速开始 </summary>

```shell
# 训练
python -m paddle.distributed.launch --log_dir=logs/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/smalldet/visdrone/ppyoloe_plus_sod_crn_l_80e_visdrone.yml --amp --eval
# 评估
python tools/eval.py -c configs/smalldet/visdrone/ppyoloe_plus_sod_crn_l_80e_visdrone.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_plus_sod_crn_l_80e_visdrone.pdparams
# 预测
python tools/infer.py -c configs/smalldet/visdrone/ppyoloe_plus_sod_crn_l_80e_visdrone.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_plus_sod_crn_l_80e_visdrone.pdparams --infer_img=demo/visdrone_0000315_01601_d_0000509.jpg --draw_threshold=0.25
```

</details>


### COCO模型

|    模型   | mAP<sup>val<br>0.5:0.95 | AP<sup>0.5 | AP<sup>0.75 | AP<sup>small | AP<sup>medium | AP<sup>large | AR<sup>small | AR<sup>medium | AR<sup>large | 下载链接  | 配置文件 |
|:--------:|:-----------------------:|:----------:|:-----------:|:------------:|:-------------:|:-----------:|:------------:|:-------------:|:------------:|:-------:|:-------:|
|PP-YOLOE+_l|             52.9       |    70.1    |    57.9     |     35.2     |     57.5      |     69.1     |     56.0     |     77.9             |     86.9     | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco.pdparams) | [配置文件](../ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml) |
|**PP-YOLOE+_SOD-l**|     53.0       |  **70.4**  |    57.7     |    **37.1**  |     57.5      |     69.0     |     **56.5**   |     77.5             |     86.7     | [下载链接](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_sod_crn_l_80e_coco.pdparams) | [配置文件](./ppyoloe_plus_sod_crn_l_80e_coco.yml) |

**注意:**
  - 上表中的模型均为**使用原图训练**，也**原图评估预测**，网络输入尺度为640x640，训练集为COCO的train2017，验证集为val2017，均为8卡总batch_size为64训练80 epoch。
  - **SOD**表示使用**基于向量的DFL算法**和针对小目标的**中心先验优化策略**，并**在模型的Neck结构中加入transformer**，可在 AP<sup>small 上提升1.9。

<details>
<summary> 快速开始 </summary>

```shell
# 训练
python -m paddle.distributed.launch --log_dir=logs/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/smalldet/ppyoloe_plus_sod_crn_l_80e_coco.yml --amp --eval
# 评估
python tools/eval.py -c configs/smalldet/ppyoloe_plus_sod_crn_l_80e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_plus_sod_crn_l_80e_coco.pdparams
# 预测
python tools/infer.py -c configs/smalldet/ppyoloe_plus_sod_crn_l_80e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_plus_sod_crn_l_80e_coco.pdparams --infer_img=demo/000000014439_640x640.jpg --draw_threshold=0.25
```

</details>


### 切图模型

|    模型   |       数据集     |  SLICE_SIZE  |  OVERLAP_RATIO  | 类别数  | mAP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | 下载链接  | 配置文件 |
|:---------|:---------------:|:---------------:|:---------------:|:------:|:-----------------------:|:-------------------:|:---------:| :-----: |
|PP-YOLOE-P2-l|   DOTA   |  500 | 0.25 | 15 |  53.9 |  78.6 | [下载链接](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_p2_crn_l_80e_sliced_DOTA_500_025.pdparams) | [配置文件](./ppyoloe_p2_crn_l_80e_sliced_DOTA_500_025.yml) |
|PP-YOLOE-P2-l|   Xview  |  400 | 0.25 | 60 |  14.9 | 27.0 | [下载链接](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_p2_crn_l_80e_sliced_xview_400_025.pdparams) | [配置文件](./ppyoloe_p2_crn_l_80e_sliced_xview_400_025.yml) |
|PP-YOLOE-l| VisDrone-DET|  640 | 0.25 | 10 |  38.5 |  60.2 | [下载链接](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams) | [配置文件](./ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml) |

**注意:**
  - 上表中的模型均为使用**切图后的子图训练**，且使用**切图后的子图评估预测**，AP精度均为**子图验证集**上评估的结果。
  - **SLICE_SIZE**表示使用SAHI工具切图后子图的边长大小，**OVERLAP_RATIO**表示切图的子图之间的重叠率。
  - VisDrone-DET的模型与[拼图模型](#拼图模型)表格中的VisDrone-DET是**同一个模型权重**，但此处AP精度是在**切图后的子图验证集**上评估的结果。

<details>
<summary> 快速开始 </summary>

```shell
# 训练
python -m paddle.distributed.launch --log_dir=logs/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml --amp --eval
# 子图直接评估
python tools/eval.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams
# 子图直接预测
python tools/infer.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams --infer_img=demo/visdrone_0000315_01601_d_0000509.jpg --draw_threshold=0.25
```

</details>


### 拼图模型

|    模型   |       数据集     |  SLICE_SIZE  |  OVERLAP_RATIO  | 类别数  | mAP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | 下载链接  | 配置文件 |
|:---------|:---------------:|:---------------:|:---------------:|:------:|:-----------------------:|:-------------------:|:---------:| :-----: |
|PP-YOLOE-l (原图直接评估)| VisDrone-DET|  640 | 0.25 | 10 |  29.7 |  48.5 | [下载链接](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams) | [配置文件](./ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml) |
|PP-YOLOE-l (切图拼图评估)| VisDrone-DET|  640 | 0.25 | 10 | 37.3 | 59.5 | [下载链接](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams) | [配置文件](./ppyoloe_crn_l_80e_sliced_visdrone_640_025_slice_infer.yml) |

**注意:**
  - 上表中的模型均为使用**切图后的子图**训练，评估预测时分为两种，**直接使用原图**评估预测，和**使用子图自动拼成原图**评估预测，AP精度均为**原图验证集**上评估的结果。。
  - **SLICE_SIZE**表示使用SAHI工具切图后子图的边长大小，**OVERLAP_RATIO**表示切图的子图之间的重叠率。
  - VisDrone-DET的模型与[切图模型](#切图模型)表格中的VisDrone-DET是**同一个模型权重**，但此处AP精度是在**原图验证集**上评估的结果，需要提前修改`ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml`里的`EvalDataset`的默认的子图验证集路径为以下**原图验证集路径**：
  ```
  EvalDataset:
    !COCODataSet
      image_dir: VisDrone2019-DET-val
      anno_path: val.json
      dataset_dir: dataset/visdrone
  ```

<details>
<summary> 快速开始 </summary>

```shell
# 训练
python -m paddle.distributed.launch --log_dir=logs/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml --amp --eval
# 原图直接评估，注意需要提前修改此yml中的 `EvalDataset` 的默认的子图验证集路径 为 原图验证集路径：
python tools/eval.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams
# 切图拼图评估，加上 --slice_infer，注意是使用的带 _slice_infer 后缀的yml配置文件
python tools/eval.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025_slice_infer.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams --slice_infer
# 切图拼图预测，加上 --slice_infer
python tools/infer.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams --infer_img=demo/visdrone_0000315_01601_d_0000509.jpg --draw_threshold=0.25 --slice_infer
```

</details>


### 注意事项

- 切图和拼图，需要使用[SAHI](https://github.com/obss/sahi)切图工具，需要首先安装：`pip install sahi`，参考[installation](https://github.com/obss/sahi/blob/main/README.md#installation)。
- DOTA水平框和Xview数据集均是**切图后训练**，AP指标为**切图后的子图val上的指标**。
- VisDrone-DET数据集请参照[visdrone](./visdrone)，**可使用原图训练，也可使用切图后训练**，这上面表格中的指标均是使用VisDrone-DET的val子集做验证而未使用test_dev子集。
- PP-YOLOE模型训练过程中使用8 GPUs进行混合精度训练，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- 常用训练验证部署等步骤请参考[ppyoloe](../ppyoloe#getting-start)。
- 自动切图和拼图的推理预测需添加设置`--slice_infer`，具体见下文[模型库使用说明](#模型库使用说明)中的[预测](#预测)和[部署](#部署)。
- 自动切图和拼图过程，参照[2.3 子图拼图评估](#评估)。


## 模型库使用说明

### 训练

#### 1.1 原图训练
首先将待训数据集制作成COCO数据集格式，然后按照PaddleDetection的模型的常规训练流程训练即可。

执行以下指令使用混合精度训练COCO数据集：

```bash
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/smalldet/ppyoloe_plus_sod_crn_l_80e_coco.yml --amp --eval
```

**注意:**
- 使用默认配置训练需要设置`--amp`以避免显存溢出，`--eval`表示边训边验证，会自动保存最佳精度的模型权重。

#### 1.2 原图训练
首先将待训数据集制作成COCO数据集格式，然后使用SAHI切图工具进行**离线切图**，对保存的子图按**常规检测模型的训练流程**走即可。
也可直接下载PaddleDetection团队提供的切图后的VisDrone-DET、DOTA水平框、Xview数据集。

执行以下指令使用混合精度训练VisDrone切图数据集：

```bash
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml --amp --eval
```


### 评估

#### 2.1 子图评估
**默认评估方式是子图评估**，子图数据集的验证集设置为：
```
EvalDataset:
  !COCODataSet
    image_dir: val_images_640_025
    anno_path: val_640_025.json
    dataset_dir: dataset/visdrone_sliced
```
按常规检测模型的评估流程，评估提前切好并存下来的子图上的精度：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams
```

#### 2.2 原图评估
修改验证集的标注文件路径为**原图标注文件**：
```
EvalDataset:
  !COCODataSet
    image_dir: VisDrone2019-DET-val
    anno_path: val.json
    dataset_dir: dataset/visdrone
```
直接评估原图上的精度：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams
```

#### 2.3 子图拼图评估
修改验证集的标注文件路径为**原图标注文件**：
```
# very slow, preferly eval with a determined weights(xx.pdparams)
# if you want to eval during training, change SlicedCOCODataSet to COCODataSet and delete sliced_size and overlap_ratio
EvalDataset:
  !SlicedCOCODataSet
    image_dir: VisDrone2019-DET-val
    anno_path: val.json
    dataset_dir: dataset/visdrone
    sliced_size: [640, 640]
    overlap_ratio: [0.25, 0.25]
```
会在评估过程中自动对原图进行切图最后再重组和融合结果来评估原图上的精度：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025_slice_infer.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams --slice_infer --combine_method=nms --match_threshold=0.6 --match_metric=ios
```

**注意:**
- 设置`--slice_infer`表示切图预测并拼装重组结果，如果不使用则不写，注意需要确保EvalDataset的数据集类是选用的SlicedCOCODataSet而不是COCODataSet；
- 设置`--slice_size`表示切图的子图尺寸大小，设置`--overlap_ratio`表示子图间重叠率，可以自行修改选择合适的子图尺度sliced_size和子图间重叠率overlap_ratio，如：
```
EvalDataset:
  !SlicedCOCODataSet
    image_dir: VisDrone2019-DET-val
    anno_path: val.json
    dataset_dir: dataset/visdrone
    sliced_size: [480, 480]
    overlap_ratio: [0.2, 0.2]
```
- 设置`--combine_method`表示子图结果重组去重的方式，默认是`nms`；
- 设置`--match_threshold`表示子图结果重组去重的阈值，默认是0.6；
- 设置`--match_metric`表示子图结果重组去重的度量标准，默认是`ios`表示交小比(两个框交集面积除以更小框的面积)，也可以选择交并比`iou`(两个框交集面积除以并集面积)，精度效果因数据集而而异，但选择`ios`预测速度会更快一点；


### 预测

#### 3.1 子图或原图直接预测
与评估流程基本相同，可以在提前切好并存下来的子图上预测，也可以对原图预测，如：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams --infer_img=demo/visdrone_0000315_01601_d_0000509.jpg --draw_threshold=0.25
```

#### 3.2 原图自动切图并拼图预测
也可以对原图进行自动切图并拼图重组来预测原图，如：
```bash
# 单张图
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams --infer_img=demo/visdrone_0000315_01601_d_0000509.jpg --draw_threshold=0.25 --slice_infer --slice_size 640 640 --overlap_ratio 0.25 0.25 --combine_method=nms --match_threshold=0.6 --match_metric=ios --save_results=True
# 或图片文件夹
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams --infer_dir=demo/ --draw_threshold=0.25 --slice_infer --slice_size 640 640 --overlap_ratio 0.25 0.25 --combine_method=nms --match_threshold=0.6 --match_metric=ios
```
- 设置`--slice_infer`表示切图预测并拼装重组结果，如果不使用则不写；
- 设置`--slice_size`表示切图的子图尺寸大小，设置`--overlap_ratio`表示子图间重叠率；
- 设置`--combine_method`表示子图结果重组去重的方式，默认是`nms`；
- 设置`--match_threshold`表示子图结果重组去重的阈值，默认是0.6；
- 设置`--match_metric`表示子图结果重组去重的度量标准，默认是`ios`表示交小比(两个框交集面积除以更小框的面积)，也可以选择交并比`iou`(两个框交集面积除以并集面积)，精度效果因数据集而而异，但选择`ios`预测速度会更快一点；
- 设置`--save_results`表示保存图片结果为json文件，一般只单张图预测时使用；


### 部署

#### 4.1 导出模型
```bash
# export model
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/smalldet/ppyoloe_crn_l_80e_sliced_visdrone_640_025.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_80e_sliced_visdrone_640_025.pdparams
```

#### 4.2 使用原图或子图直接推理
```bash
# deploy infer
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyoloe_crn_l_80e_sliced_visdrone_640_025 --image_file=demo/visdrone_0000315_01601_d_0000509.jpg --device=GPU --save_images --threshold=0.25
```

#### 4.3 使用原图自动切图并拼图重组结果来推理
```bash
# deploy slice infer
# 单张图
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyoloe_crn_l_80e_sliced_visdrone_640_025 --image_file=demo/visdrone_0000315_01601_d_0000509.jpg --device=GPU --save_images --threshold=0.25  --slice_infer --slice_size 640 640 --overlap_ratio 0.25 0.25 --combine_method=nms --match_threshold=0.6 --match_metric=ios  --save_results=True
# 或图片文件夹
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyoloe_crn_l_80e_sliced_visdrone_640_025 --image_dir=demo/ --device=GPU --save_images --threshold=0.25  --slice_infer --slice_size 640 640 --overlap_ratio 0.25 0.25 --combine_method=nms --match_threshold=0.6 --match_metric=ios
```
- 设置`--slice_infer`表示切图预测并拼装重组结果，如果不使用则不写；
- 设置`--slice_size`表示切图的子图尺寸大小，设置`--overlap_ratio`表示子图间重叠率；
- 设置`--combine_method`表示子图结果重组去重的方式，默认是`nms`；
- 设置`--match_threshold`表示子图结果重组去重的阈值，默认是0.6；
- 设置`--match_metric`表示子图结果重组去重的度量标准，默认是`ios`表示交小比(两个框交集面积除以更小框的面积)，也可以选择交并比`iou`(两个框交集面积除以并集面积)，精度效果因数据集而而异，但选择`ios`预测速度会更快一点；
- 设置`--save_results`表示保存图片结果为json文件，一般只单张图预测时使用；


## 引用
```
@article{akyon2022sahi,
  title={Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection},
  author={Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
  journal={2022 IEEE International Conference on Image Processing (ICIP)},
  doi={10.1109/ICIP46576.2022.9897990},
  pages={966-970},
  year={2022}
}

@inproceedings{xia2018dota,
  title={DOTA: A large-scale dataset for object detection in aerial images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3974--3983},
  year={2018}
}

@ARTICLE{9573394,
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Detection and Tracking Meet Drones Challenge},
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3119563}
}
```
