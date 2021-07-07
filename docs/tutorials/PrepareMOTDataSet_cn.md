简体中文 | [English](GETTING_STARTED.md)

# 目录
## 多目标跟踪数据集准备
- [MOT数据集](#MOT数据集)
- [数据格式](#数据格式)
- [数据集目录](#数据集目录)
- [下载链接](#下载链接)
- [用户数据准备](#用户数据准备)
- [引用](#引用)

### MOT数据集
PaddleDetection使用和[JDE](https://github.com/Zhongdao/Towards-Realtime-MOT) 还有[FairMOT](https://github.com/ifzhang/FairMOT)相同的数据集，请先下载并准备好所有的数据集包括**Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17和MOT16**。此外还可以下载**MOT15和MOT20**数据集，如果您想使用这些数据集，请**遵循他们的License**。

### 数据格式
这几个相关数据集都遵循以下结构：
```
Caltech
   |——————images
   |        └——————00001.jpg
   |        |—————— ...
   |        └——————0000N.jpg
   └——————labels_with_ids
            └——————00001.txt
            |—————— ...
            └——————0000N.txt
MOT17
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train
```
所有数据集的标注是以统一数据格式提供的。各个数据集中每张图片都有相应的标注文本。给定一个图像路径，可以通过将字符串`images`替换为`labels_with_ids`并将`.jpg`替换为`.txt`来生成标注文本路径。在标注文本中，每行都描述一个边界框，格式如下：
```
[class] [identity] [x_center] [y_center] [width] [height]
```
**注意**:
- `class`为`0`，目前仅支持单类别多目标跟踪。
- `identity`是从`0`到`num_identifies-1`的整数(`num_identifies`是数据集中不同物体实例的总数)，如果此框没有`identity`标注，则为`-1`。
- `[x_center] [y_center] [width] [height]`是中心点坐标和宽高，注意它们的值是由图片的宽度/高度标准化的，因此它们是从0到1的浮点数。

### 数据集目录

首先按照以下命令下载image_lists.zip并解压放在`dataset/mot`目录下：
```
wget https://dataset.bj.bcebos.com/mot/image_lists.zip
```
然后依次下载各个数据集并解压，最终目录为：
```
dataset/mot
  |——————image_lists
            |——————caltech.10k.val  
            |——————caltech.all  
            |——————caltech.train  
            |——————caltech.val  
            |——————citypersons.train  
            |——————citypersons.val  
            |——————cuhksysu.train  
            |——————cuhksysu.val  
            |——————eth.train  
            |——————mot15.train  
            |——————mot16.train  
            |——————mot17.train  
            |——————mot20.train  
            |——————prw.train  
            |——————prw.val
  |——————Caltech
  |——————Cityscapes
  |——————CUHKSYSU
  |——————ETHZ
  |——————MOT15
  |——————MOT16
  |——————MOT17
  |——————MOT20
  |——————PRW
```

### 下载链接

#### Caltech Pedestrian
Baidu NetDisk:
[[0]](https://pan.baidu.com/s/1sYBXXvQaXZ8TuNwQxMcAgg)
[[1]](https://pan.baidu.com/s/1lVO7YBzagex1xlzqPksaPw)
[[2]](https://pan.baidu.com/s/1PZXxxy_lrswaqTVg0GuHWg)
[[3]](https://pan.baidu.com/s/1M93NCo_E6naeYPpykmaNgA)
[[4]](https://pan.baidu.com/s/1ZXCdPNXfwbxQ4xCbVu5Dtw)
[[5]](https://pan.baidu.com/s/1kcZkh1tcEiBEJqnDtYuejg)
[[6]](https://pan.baidu.com/s/1sDjhtgdFrzR60KKxSjNb2A)
[[7]](https://pan.baidu.com/s/18Zvp_d33qj1pmutFDUbJyw)

Google Drive: [[annotations]](https://drive.google.com/file/d/1h8vxl_6tgi9QVYoer9XcY9YwNB32TE5k/view?usp=sharing),
请从[这个页面](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/)下载所有的`.tar`结尾的图片文件, 并解压到`Caltech/images`目录。

你需要使用这个[工具](https://github.com/mitmul/caltech-pedestrian-dataset-converter) 将原始数据格式转换为jpeg图像。
原始数据集网址: [CaltechPedestrians](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)

#### CityPersons
Baidu NetDisk:
[[0]](https://pan.baidu.com/s/1g24doGOdkKqmbgbJf03vsw)
[[1]](https://pan.baidu.com/s/1mqDF9M5MdD3MGxSfe0ENsA)
[[2]](https://pan.baidu.com/s/1Qrbh9lQUaEORCIlfI25wdA)
[[3]](https://pan.baidu.com/s/1lw7shaffBgARDuk8mkkHhw)

Google Drive:
[[0]](https://drive.google.com/file/d/1DgLHqEkQUOj63mCrS_0UGFEM9BG8sIZs/view?usp=sharing)
[[1]](https://drive.google.com/file/d/1BH9Xz59UImIGUdYwUR-cnP1g7Ton_LcZ/view?usp=sharing)
[[2]](https://drive.google.com/file/d/1q_OltirP68YFvRWgYkBHLEFSUayjkKYE/view?usp=sharing)
[[3]](https://drive.google.com/file/d/1VSL0SFoQxPXnIdBamOZJzHrHJ1N2gsTW/view?usp=sharing)

原始数据集网址: [Citypersons pedestrian detection dataset](https://github.com/cvgroup-njust/CityPersons)

#### CUHK-SYSU
Baidu NetDisk:
[[0]](https://pan.baidu.com/s/1YFrlyB1WjcQmFW3Vt_sEaQ)

Google Drive:
[[0]](https://drive.google.com/file/d/1D7VL43kIV9uJrdSCYl53j89RE2K-IoQA/view?usp=sharing)

原始数据集网址: [CUHK-SYSU Person Search Dataset](http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html)

#### PRW
Baidu NetDisk:
[[0]](https://pan.baidu.com/s/1iqOVKO57dL53OI1KOmWeGQ)

Google Drive:
[[0]](https://drive.google.com/file/d/116_mIdjgB-WJXGe8RYJDWxlFnc_4sqS8/view?usp=sharing)


#### ETHZ (overlapping videos with MOT-16 removed):
Baidu NetDisk:
[[0]](https://pan.baidu.com/s/14EauGb2nLrcB3GRSlQ4K9Q)

Google Drive:
[[0]](https://drive.google.com/file/d/19QyGOCqn8K_rc9TXJ8UwLSxCx17e0GoY/view?usp=sharing)

原始数据集网址: [ETHZ pedestrian datset](https://data.vision.ee.ethz.ch/cvl/aess/dataset/)

#### MOT-17
Baidu NetDisk:
[[0]](https://pan.baidu.com/s/1lHa6UagcosRBz-_Y308GvQ)

Google Drive:
[[0]](https://drive.google.com/file/d/1ET-6w12yHNo8DKevOVgK1dBlYs739e_3/view?usp=sharing)

原始数据集网址: [MOT-17](https://motchallenge.net/data/MOT17/)

#### MOT-16
Baidu NetDisk:
[[0]](https://pan.baidu.com/s/10pUuB32Hro-h-KUZv8duiw)

Google Drive:
[[0]](https://drive.google.com/file/d/1254q3ruzBzgn4LUejDVsCtT05SIEieQg/view?usp=sharing)

原始数据集网址: [MOT-16](https://motchallenge.net/data/MOT16/)

#### MOT-15
原始数据集网址: [MOT-15](https://motchallenge.net/data/MOT15/)

#### MOT-20
原始数据集网址: [MOT-20](https://motchallenge.net/data/MOT20/)


### 用户数据准备

为了规范地进行训练和评测，用户数据需要转成和MOT-16数据集相同的目录和格式：
```
custom_data
   |——————images
   |        └——————test
   |        └——————train
   |                └——————seq1
   |                |        └——————gt
   |                |        |       └——————gt.txt
   |                |        └——————img1
   |                |        |       └——————000001.jpg
   |                |        |       |——————000002.jpg
   |                |        |       └—————— ...
   |                |        └——————seqinfo.ini
   |                └——————seq2
   |                └——————...
   └——————labels_with_ids
            └——————train
                    └——————seq1
                    |        └——————000001.txt
                    |        |——————000002.txt
                    |        └—————— ...
                    └——————seq2
                    └—————— ...
```

#### images文件夹
- `gt.txt`是原始标注文件，而训练所用标注是`labels_with_ids`文件夹。
- `img1`文件夹里是按照一定帧率抽好的图片。
- `seqinfo.ini`文件是视频信息描述文件，需要如下格式的信息：
```
[Sequence]
name=MOT16-02
imDir=img1
frameRate=30
seqLength=600
imWidth=1920
imHeight=1080
imExt=.jpg
```

`gt.txt`里是当前视频中所有图片的原始标注文件，每行都描述一个边界框，格式如下：
```
[frame_id],[identity],[bb_left],[bb_top],[width],[height],[x],[y],[z]
```
**注意**:
- `frame_id`为当前图片帧序号
- `identity`是从`0`到`num_identifies-1`的整数(`num_identifies`是数据集中不同物体实例的总数)，如果此框没有`identity`标注，则为`-1`
- `bb_left`是目标框的左边界的x坐标
- `bb_top`是目标框的上边界的y坐标
- `width，height`是真实的像素宽高
- `x,y,z`是3D中用到的，在2D中默认为`-1`


#### labels_with_ids文件夹
所有数据集的标注是以统一数据格式提供的。各个数据集中每张图片都有相应的标注文本。给定一个图像路径，可以通过将字符串`images`替换为`labels_with_ids`并将`.jpg`替换为`.txt`来生成标注文本路径。在标注文本中，每行都描述一个边界框，格式如下：
```
[class] [identity] [x_center] [y_center] [width] [height]
```
**注意**:
- `class`为`0`，目前仅支持单类别多目标跟踪。
- `identity`是从`0`到`num_identifies-1`的整数(`num_identifies`是数据集中不同物体实例的总数)，如果此框没有`identity`标注，则为`-1`。
- `[x_center] [y_center] [width] [height]`是中心点坐标和宽高，注意是由图片的宽度/高度标准化的，因此它们是从0到1的浮点数。

可采用如下脚本生成相应的`labels_with_ids`:
```
cd dataset/mot
python gen_labels_MOT.py
```

### 引用
Caltech:
```
@inproceedings{ dollarCVPR09peds,
       author = "P. Doll\'ar and C. Wojek and B. Schiele and  P. Perona",
       title = "Pedestrian Detection: A Benchmark",
       booktitle = "CVPR",
       month = "June",
       year = "2009",
       city = "Miami",
}
```
Citypersons:
```
@INPROCEEDINGS{Shanshan2017CVPR,
  Author = {Shanshan Zhang and Rodrigo Benenson and Bernt Schiele},
  Title = {CityPersons: A Diverse Dataset for Pedestrian Detection},
  Booktitle = {CVPR},
  Year = {2017}
 }

@INPROCEEDINGS{Cordts2016Cityscapes,
title={The Cityscapes Dataset for Semantic Urban Scene Understanding},
author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2016}
}
```
CUHK-SYSU:
```
@inproceedings{xiaoli2017joint,
  title={Joint Detection and Identification Feature Learning for Person Search},
  author={Xiao, Tong and Li, Shuang and Wang, Bochao and Lin, Liang and Wang, Xiaogang},
  booktitle={CVPR},
  year={2017}
}
```
PRW:
```
@inproceedings{zheng2017person,
  title={Person re-identification in the wild},
  author={Zheng, Liang and Zhang, Hengheng and Sun, Shaoyan and Chandraker, Manmohan and Yang, Yi and Tian, Qi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1367--1376},
  year={2017}
}
```
ETHZ:
```
@InProceedings{eth_biwi_00534,
author = {A. Ess and B. Leibe and K. Schindler and and L. van Gool},
title = {A Mobile Vision System for Robust Multi-Person Tracking},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR'08)},
year = {2008},
month = {June},
publisher = {IEEE Press},
keywords = {}
}
```
MOT-16&17:
```
@article{milan2016mot16,
  title={MOT16: A benchmark for multi-object tracking},
  author={Milan, Anton and Leal-Taix{\'e}, Laura and Reid, Ian and Roth, Stefan and Schindler, Konrad},
  journal={arXiv preprint arXiv:1603.00831},
  year={2016}
}
```
