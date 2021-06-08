[English](FACE_DETECTION_en.md) | 简体中文
# 人脸检测模型

## 内容
- [简介](#简介)
- [模型库与基线](#模型库与基线)
- [快速开始](#快速开始)
    - [数据准备](#数据准备)
    - [训练与推理](#训练与推理)
    - [评估](#评估)
- [人脸关键点检测](#人脸关键点检测)
- [算法细节](#算法细节)
- [如何贡献代码](#如何贡献代码)

## 简介
FaceDetection的目标是提供高效、高速的人脸检测解决方案，包括最先进的模型和经典模型。

![](../images/12_Group_Group_12_Group_Group_12_935.jpg)

## 模型库与基线
下表中展示了PaddleDetection当前支持的网络结构，具体细节请参考[算法细节](#算法细节)。

|                          | 原始版本  | Lite版本 <sup>[1](#lite)</sup> | NAS版本 <sup>[2](#nas)</sup> |
|:------------------------:|:--------:|:--------------------------:|:------------------------:|
| [BlazeFace](#BlazeFace)  | ✓        |                          ✓ | ✓                        |
| [FaceBoxes](#FaceBoxes)  | ✓        |                          ✓ | x                        |

<a name="lite">[1]</a> `Lite版本`表示减少网络层数和通道数。  
<a name="nas">[2]</a> `NA版本`表示使用 `神经网络搜索`方法来构建网络结构。

### 模型库

#### WIDER-FACE数据集上的mAP

| 网络结构 | 类型     | 输入尺寸 | 图片个数/GPU | 学习率策略 | Easy Set  | Medium Set | Hard Set  | 下载 | 配置文件 |
|:------------:|:--------:|:----:|:-------:|:-------:|:---------:|:----------:|:---------:|:--------:|:--------:|
| BlazeFace    | 原始版本 | 640  |    8    | 32w     | **0.915** | **0.892**  | **0.797** | [模型](https://paddlemodels.bj.bcebos.com/object_detection/blazeface_original.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/face_detection/blazeface.yml) |
| BlazeFace    | Lite版本    | 640  |    8    | 32w     | 0.909     | 0.885      | 0.781     | [模型](https://paddlemodels.bj.bcebos.com/object_detection/blazeface_lite.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/face_detection/blazeface.yml) |
| BlazeFace    | NAS版本    | 640  |    8    | 32w     | 0.837     | 0.807      | 0.658     | [模型](https://paddlemodels.bj.bcebos.com/object_detection/blazeface_nas.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/face_detection/blazeface_nas.yml) |
| BlazeFace    | NAS_V2版本 | 640  |    8    | 32W     | 0.870     | 0.837      | 0.685     | [模型](https://paddlemodels.bj.bcebos.com/object_detection/blazeface_nas2.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/face_detection/blazeface_nas_v2.yml) |
| FaceBoxes    | 原始版本 | 640  |    8    | 32w     | 0.878     | 0.851      | 0.576     | [模型](https://paddlemodels.bj.bcebos.com/object_detection/faceboxes_original.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/face_detection/faceboxes.yml) |
| FaceBoxes    | Lite版本   | 640  |    8    | 32w     | 0.901     | 0.875      | 0.760     | [模型](https://paddlemodels.bj.bcebos.com/object_detection/faceboxes_lite.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/face_detection/faceboxes_lite.yml) |

**注意:**  
- 我们使用`tools/face_eval.py`中多尺度评估策略得到`Easy/Medium/Hard Set`里的mAP。具体细节请参考[在WIDER-FACE数据集上评估](#在WIDER-FACE数据集上评估)。
- BlazeFace-Lite的训练与测试使用 [blazeface.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/configs/face_detection/blazeface.yml)配置文件并且设置：`lite_edition: true`。

#### FDDB数据集上的mAP

| 网络结构 | Type     | Size | DistROC | ContROC |
|:------------:|:--------:|:----:|:-------:|:-------:|
| BlazeFace    | 原始版本 | 640  | **0.992**   | **0.762**   |
| BlazeFace    | Lite版本   | 640  | 0.990   | 0.756   |
| BlazeFace    | NAS版本    | 640  | 0.981   | 0.741   |
| FaceBoxes    | 原始版本 | 640  | 0.987   | 0.736   |
| FaceBoxes    | Lite版本   | 640  | 0.988   | 0.751   |

**注意:**  
- 我们在FDDB数据集上使用多尺度测试的方法得到mAP，具体细节请参考[在FDDB数据集上评估](#在FDDB数据集上评估)。

#### 推理时间和模型大小比较

| 网络结构 | 类型     | 输入尺寸 | P4(trt32) (ms) | CPU (ms) |  CPU (ms)(enable_mkldmm) | 高通骁龙855(armv8) (ms)   | 模型大小(MB) |
|:------------:|:--------:|:----:|:--------------:|:--------:|:--------:|:-------------------------------------:|:---------------:|
| BlazeFace    | 原始版本 | 128  | 1.387          | 23.461   | 4.92 |  6.036                                | 0.777           |
| BlazeFace    | Lite版本   | 128  | 1.323          | 12.802   | 7.16 | 6.193                                | 0.68            |
| BlazeFace    | NAS版本    | 128  | 1.03           | 6.714    | 3.641 | 2.7152                               | 0.234           |
| BlazeFace    | NAS_V2版本    | 128  | 0.909        |   9.58  | 7.903 | 3.499                               | 0.383           |
| FaceBoxes    | 原始版本 | 128  | 3.144          | 14.972   | 9,852 | 19.2196                              | 3.6             |
| FaceBoxes    | Lite版本   | 128  | 2.295          | 11.276   | 6.969 | 8.5278                               | 2               |
| BlazeFace    | 原始版本 | 320  | 3.01           | 132.408  | 20.762 | 70.6916                              | 0.777           |
| BlazeFace    | Lite版本   | 320  | 2.535          | 69.964   | 35.612 | 69.9438                              | 0.68            |
| BlazeFace    | NAS版本    | 320  | 2.392          | 36.962   | 14.443 | 39.8086                              | 0.234           |
| BlazeFace    | NAS_V2版本    | 320  | 1.487          | 52.038   | 38.693 | 56.137                              | 0.383           |
| FaceBoxes    | 原始版本 | 320  | 7.556          | 84.531   | 48.465 | 52.1022                              | 3.6             |
| FaceBoxes    | Lite版本   | 320  | 18.605         | 78.862   | 46.488 |  59.8996                              | 2               |
| BlazeFace    | 原始版本 | 640  | 8.885          | 519.364  | 78.825 | 149.896                              | 0.777           |
| BlazeFace    | Lite版本   | 640  | 6.988          | 284.13   | 131.385 | 149.902                              | 0.68            |
| BlazeFace    | NAS版本    | 640  | 7.448          | 142.91   | 56.725 | 69.8266                              | 0.234           |
| BlazeFace    | NAS_V2版本    | 640  | 4.201          | 197.695   | 153.626 | 88.278                             | 0.383           |
| FaceBoxes    | 原始版本 | 640  | 78.201         | 394.043  |  239.201 | 169.877                              | 3.6             |
| FaceBoxes    | Lite版本  | 640  | 59.47          | 313.683  | 168.73 | 139.918                              | 2               |


**注意:**  
- CPU: Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz。
- P4(trt32)和CPU的推理时间测试基于PaddlePaddle-1.8.0版本。
- ARM测试环境:
    - 高通骁龙855(armv8)；
    - 单线程；
    - Paddle-Lite develop版本。


## 快速开始

### 数据准备
我们使用[WIDER-FACE数据集](http://shuoyang1213.me/WIDERFACE/)进行训练和模型测试，官方网站提供了详细的数据介绍。
- WIDER-Face数据源:  
使用如下目录结构加载`wider_face`类型的数据集：

  ```
  dataset/wider_face/
  ├── wider_face_split
  │   ├── wider_face_train_bbx_gt.txt
  │   ├── wider_face_val_bbx_gt.txt
  ├── WIDER_train
  │   ├── images
  │   │   ├── 0--Parade
  │   │   │   ├── 0_Parade_marchingband_1_100.jpg
  │   │   │   ├── 0_Parade_marchingband_1_381.jpg
  │   │   │   │   ...
  │   │   ├── 10--People_Marching
  │   │   │   ...
  ├── WIDER_val
  │   ├── images
  │   │   ├── 0--Parade
  │   │   │   ├── 0_Parade_marchingband_1_1004.jpg
  │   │   │   ├── 0_Parade_marchingband_1_1045.jpg
  │   │   │   │   ...
  │   │   ├── 10--People_Marching
  │   │   │   ...
  ```

- 手动下载数据集：
要下载WIDER-FACE数据集，请运行以下命令：
```
cd dataset/wider_face && ./download.sh
```

- 自动下载数据集：
如果已经开始训练，但是数据集路径设置不正确或找不到路径, PaddleDetection会从[WIDER-FACE数据集](http://shuoyang1213.me/WIDERFACE/)自动下载它们，
下载后解压的数据集将缓存在`~/.cache/paddle/dataset/`中，并且之后的训练测试会自动加载它们。

#### 数据增强方法

- **尺度变换(Data-anchor-sampling):**
具体操作是:根据随机选择的人脸高和宽，获取到$v=\sqrt{width * height}$，之后再判断`v`的值范围，其中`v`值位于缩放区间`[16,32,64,128]`
假设`v=45`，则选定`32<v<64`, 以均匀分布的概率选取`[16,32,64]`中的任意一个值。若选中`64`，则该人脸的缩放区间在`[64 / 2, min(v * 2, 64 * 2)]`中选定。

- **其他方法:** 包括随机扰动、翻转、裁剪等。具体请参考[READER.md](../advanced_tutorials/READER.md)。

### 训练与推理
训练流程与推理流程方法与其他算法一致，请参考[GETTING_STARTED_cn.md](../tutorials/GETTING_STARTED_cn.md)。  
**注意:**
- `BlazeFace`和`FaceBoxes`训练是以每卡`batch_size=8`在4卡GPU上进行训练(总`batch_size`是32),并且训练320000轮
(如果你的GPU数达不到4，请参考[学习率计算规则表](../FAQ.md))。
- 人脸检测模型目前我们不支持边训练边评估。


### 评估
目前我们支持在`WIDER FACE`数据集和`FDDB`数据集上评估。首先运行`tools/face_eval.py`生成评估结果文件，其次使用matlab（WIDER FACE）
或OpenCV（FDDB）计算具体的评估指标。  
其中，运行`tools/face_eval.py`的参数列表如下：
- `-f` 或者 `--output_eval`: 评估生成的结果文件保存路径，默认是： `output/pred`；
- `-e` 或者 `--eval_mode`: 评估模式，包括 `widerface` 和 `fddb`，默认是`widerface`；
- `--multi_scale`: 如果在命令中添加此操作按钮，它将选择多尺度评估。默认值为`False`，它将选择单尺度评估。

#### 在WIDER-FACE数据集上评估
评估并生成结果文件：
```
export CUDA_VISIBLE_DEVICES=0
python -u tools/face_eval.py -c configs/face_detection/blazeface.yml \
       -o weights=output/blazeface/model_final \
       --eval_mode=widerface
```
评估完成后，将在`output/pred`中生成txt格式的测试结果。

- 下载官方评估脚本来评估AP指标：
```
wget http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip
unzip eval_tools.zip && rm -f eval_tools.zip
```
- 在`eval_tools/wider_eval.m`中修改保存结果路径和绘制曲线的名称：
```
# Modify the folder name where the result is stored.
pred_dir = './pred';  
# Modify the name of the curve to be drawn
legend_name = 'Fluid-BlazeFace';
```
- `wider_eval.m` 是评估模块的主要执行程序。运行命令如下：
```
matlab -nodesktop -nosplash -nojvm -r "run wider_eval.m;quit;"
```

#### 在FDDB数据集上评估
我们提供了一套FDDB数据集的评估流程(目前仅支持Linux系统)，其他具体细节请参考[FDDB官网](http://vis-www.cs.umass.edu/fddb/)。  

- 1)下载安装opencv：  
下载OpenCV: 进入[OpenCV library](https://opencv.org/releases/)手动下载  
安装OpenCV：请参考[OpenCV官方安装教程](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)通过源码安装。

- 2)下载数据集、评估代码以及格式化数据：  
```
./dataset/fddb/download.sh
```

- 3)编译FDDB评估代码：
进入`dataset/fddb/evaluation`目录下，修改MakeFile文件中内容如下：
```
evaluate: $(OBJS)
    $(CC) $(OBJS) -o $@ $(LIBS)
```
修改`common.hpp`中内容为如下形式：
```
#define __IMAGE_FORMAT__ ".jpg"
//#define __IMAGE_FORMAT__ ".ppm"
#define __CVLOADIMAGE_WORKING__
```
根据`grep -r "CV_RGB"`命令找到含有`CV_RGB`的代码段，将`CV_RGB`改成`Scalar`，并且在cpp中加入`using namespace cv;`，
然后编译：
```
make clean && make
```

- 4)开始评估:  
修改config文件中`dataset_dir`和`annotation`字段内容：
```
EvalReader:
  ...
  dataset:
    dataset_dir: dataset/fddb
    anno_path: FDDB-folds/fddb_annotFile.txt
    ...
```
评估并生成结果文件：
```
python -u tools/face_eval.py -c configs/face_detection/blazeface.yml \
       -o weights=output/blazeface/model_final \
       --eval_mode=fddb
```
评估完成后，将在`output/pred/pred_fddb_res.txt`中生成txt格式的测试结果。  
生成ContROC与DiscROC数据：  
```
cd dataset/fddb/evaluation
./evaluate -a ./FDDB-folds/fddb_annotFile.txt \
           -f 0 -i ./ -l ./FDDB-folds/filePath.txt -z .jpg \
           -d {RESULT_FILE} \
           -r {OUTPUT_DIR}
```
**注意:**  
(1)`RESULT_FILE`是`tools/face_eval.py`输出的FDDB预测结果文件；  
(2)`OUTPUT_DIR`是FDDB评估输出结果文件前缀，会生成两个文件`{OUTPUT_DIR}ContROC.txt`、`{OUTPUT_DIR}DiscROC.txt`；  
(3)参数用法及注释可通过执行`./evaluate --help`来获取。


## 人脸关键点检测

(1)下载PaddleDetection开放的WIDER-FACE数据集人脸关键点标注文件([链接](https://dataset.bj.bcebos.com/wider_face/wider_face_train_bbx_lmk_gt.txt))，并拷贝至`wider_face/wider_face_split`文件夹中：

```shell
cd dataset/wider_face/wider_face_split/
wget https://dataset.bj.bcebos.com/wider_face/wider_face_train_bbx_lmk_gt.txt
```

(2)使用`configs/face_detection/blazeface_keypoint.yml`配置文件进行训练与评估，使用方法与上一节内容一致。

### 模型评估

| 网络结构     | 输入尺寸 | 图片个数/GPU | 学习率策略 | Easy Set  | Medium Set | Hard Set  | 下载 | 配置文件 |
|:------------:|:----:|:-------:|:-------:|:---------:|:----------:|:---------:|:--------:|:--------:|
| BlazeFace Keypoint     | 640  |    16    | 16w     | 0.852     | 0.816      | 0.662     | [模型](https://paddlemodels.bj.bcebos.com/object_detection/blazeface_keypoint.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/face_detection/blazeface_keypoint.yml) |

![](../images/12_Group_Group_12_Group_Group_12_84.jpg)

## 算法细节

### BlazeFace
**简介:**  
[BlazeFace](https://arxiv.org/abs/1907.05047) 是Google Research发布的人脸检测模型。它轻巧并且性能良好，
专为移动GPU推理量身定制。在旗舰设备上，速度可达到200-1000+FPS。

**特点:**  
- 锚点策略在8×8（输入128x128）的特征图上停止，在该分辨率下每个像素点6个锚点；
- 5个单BlazeBlock和6个双BlazeBlock：5×5 depthwise卷积，可以保证在相同精度下网络层数更少；
- 用混合策略替换非极大值抑制算法，该策略将边界框的回归参数估计为重叠预测之间的加权平均值。

**版本信息:**
- 原始版本: 参考原始论文复现；
- Lite版本: 使用3x3卷积替换5x5卷积，更少的网络层数和通道数；
- NAS版本: 使用神经网络搜索算法构建网络结构，相比于`Lite`版本，NAS版本需要更少的网络层数和通道数。
- NAS_V2版本: 基于PaddleSlim中SANAS算法在blazeface-NAS的基础上搜索出来的结构，相比`NAS`版本，NAS_V2版本的精度平均高出3个点，在855芯片上的硬件延时相对`NAS`版本仅增加5%。

### FaceBoxes
**简介:**  
[FaceBoxes](https://arxiv.org/abs/1708.05234) 由Shifeng Zhang等人提出的高速和高准确率的人脸检测器，
被称为“高精度CPU实时人脸检测器”。 该论文收录于IJCB（2017）。

**特点:**
- 锚点策略分别在20x20、10x10、5x5（输入640x640）执行，每个像素点分别是3、1、1个锚点，对应密度系数是`1, 2, 4`(20x20)、4(10x10)、4(5x5)；
- 在基础网络中个别block中使用CReLU和inception的结构；
- 使用密度先验盒（density_prior_box）可提高检测精度。

**版本信息:**
- 原始版本: 参考原始论文进行修改；
- Lite版本: 使用更少的网络层数和通道数，具体可参考[代码](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/static/ppdet/modeling/architectures/faceboxes.py)。


## 如何贡献代码
我们非常欢迎您可以为PaddleDetection中的人脸检测模型提供代码，您可以提交PR供我们review；也十分感谢您的反馈，可以提交相应issue，我们会及时解答。
