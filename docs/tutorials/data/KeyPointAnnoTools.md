简体中文 | [English](KeyPointAnnoTools_en.md)

# 关键点检测标注工具

## 目录

[LabelMe](#LabelMe)

- [使用说明](#使用说明)
  - [安装](#安装)
  - [关键点数据说明](#关键点数据说明)
  - [图片标注过程](#图片标注过程)
- [标注格式](#标注格式)
  - [导出数据格式](#导出数据格式)
  - [格式转化总结](#格式转化总结)
  - [标注文件(json)-->COCO](#标注文件(json)-->COCO数据集)



## [LabelMe](https://github.com/wkentaro/labelme)

### 使用说明

#### 安装

具体安装操作请参考[LabelMe官方教程](https://github.com/wkentaro/labelme)中的Installation

<details>
<summary><b> Ubuntu</b></summary>

```
sudo apt-get install labelme

# or
sudo pip3 install labelme

# or install standalone executable from:
# https://github.com/wkentaro/labelme/releases
```

</details>

<details>
<summary><b> macOS</b></summary>

```
brew install pyqt  # maybe pyqt5
pip install labelme

# or
brew install wkentaro/labelme/labelme  # command line interface
# brew install --cask wkentaro/labelme/labelme  # app

# or install standalone executable/app from:
# https://github.com/wkentaro/labelme/releases
```

</details>



推荐使用Anaconda的安装方式

```
conda create –name=labelme python=3
conda activate labelme
pip install pyqt5
pip install labelme
```



#### 关键点数据说明

以COCO数据集为例，共需采集17个关键点

```
keypoint indexes:
        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'
```





#### 图片标注过程

启动labelme后，选择图片文件或者图片所在文件夹

左侧编辑栏选择`create polygons` ，右击图像区域选择标注形状，绘制好关键点后按下回车，弹出新的框填入标注关键点对应的标签

左侧菜单栏点击保存，生成`json`形式的**标注文件**

![操作说明](https://user-images.githubusercontent.com/34162360/178250648-29ee781a-676b-419c-83b1-de1e4e490526.gif)



### 标注格式

#### 导出数据格式

```
#生成标注文件
png/jpeg/jpg-->labelme标注-->json
```



#### 格式转化总结

```
#标注文件转化为COCO数据集格式
json-->labelme2coco.py-->COCO数据集
```





#### 标注文件(json)-->COCO数据集

使用[PaddleDetection提供的x2coco.py](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/tools/x2coco.py) 将labelme标注的数据转换为COCO数据集形式

```bash
python tools/x2coco.py \
                --dataset_type labelme \
                --json_input_dir ./labelme_annos/ \
                --image_input_dir ./labelme_imgs/ \
                --output_dir ./cocome/ \
                --train_proportion 0.8 \
                --val_proportion 0.2 \
                --test_proportion 0.0
```

用户数据集转成COCO数据后目录结构如下（注意数据集中路径名、文件名尽量不要使用中文，避免中文编码问题导致出错）：

```
dataset/xxx/
├── annotations
│   ├── train.json  # coco数据的标注文件
│   ├── valid.json  # coco数据的标注文件
├── images
│   ├── xxx1.jpg
│   ├── xxx2.jpg
│   ├── xxx3.jpg
│   |   ...
...
```
