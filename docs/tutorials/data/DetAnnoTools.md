简体中文 | [English](DetAnnoTools_en.md)



# 目标检测标注工具

## 目录

[LabelMe](#LabelMe)

* [使用说明](#使用说明)
  * [安装](#LabelMe安装)
  * [图片标注过程](#LabelMe图片标注过程)
* [标注格式](#LabelMe标注格式)
  * [导出数据格式](#LabelMe导出数据格式)
  * [格式转化总结](#格式转化总结)
  * [标注文件(json)-->VOC](#标注文件(json)-->VOC数据集)
  * [标注文件(json)-->COCO](#标注文件(json)-->COCO数据集)

[LabelImg](#LabelImg)

* [使用说明](#使用说明)
  * [LabelImg安装](#LabelImg安装)
  * [安装注意事项](#安装注意事项)
  * [图片标注过程](#LabelImg图片标注过程)
* [标注格式](#LabelImg标注格式)
  * [导出数据格式](#LabelImg导出数据格式)
  * [格式转换注意事项](#格式转换注意事项)



## [LabelMe](https://github.com/wkentaro/labelme)

### 使用说明

#### LabelMe安装

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





#### LabelMe图片标注过程

启动labelme后，选择图片文件或者图片所在文件夹

左侧编辑栏选择`create polygons`  绘制标注区域如下图所示（右击图像区域可以选择不同的标注形状），绘制好区域后按下回车，弹出新的框填入标注区域对应的标签，如：people

左侧菜单栏点击保存，生成`json`形式的**标注文件**

![](https://media3.giphy.com/media/XdnHZgge5eynRK3ATK/giphy.gif?cid=790b7611192e4c0ec2b5e6990b6b0f65623154ffda66b122&rid=giphy.gif&ct=g)



### LabelMe标注格式

#### LabelMe导出数据格式

```
#生成标注文件
png/jpeg/jpg-->labelme标注-->json
```





#### 格式转化总结

```
#标注文件转化为VOC数据集格式
json-->labelme2voc.py-->VOC数据集

#标注文件转化为COCO数据集格式
json-->labelme2coco.py-->COCO数据集
```





#### 标注文件(json)-->VOC数据集

使用[官方给出的labelme2voc.py](https://github.com/wkentaro/labelme/blob/main/examples/bbox_detection/labelme2voc.py)这份脚本

下载该脚本，在命令行中使用

```Te
python labelme2voc.py data_annotated(标注文件所在文件夹) data_dataset_voc(输出文件夹) --labels labels.txt
```

运行后，在指定的输出文件夹中会如下的目录

```
# It generates:
#   - data_dataset_voc/JPEGImages
#   - data_dataset_voc/Annotations
#   - data_dataset_voc/AnnotationsVisualization

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





## [LabelImg](https://github.com/tzutalin/labelImg)

### 使用说明

#### LabelImg安装

安装操作请参考[LabelImg官方教程](https://github.com/tzutalin/labelImg)

<details>
<summary><b> Ubuntu</b></summary>

```
sudo apt-get install pyqt5-dev-tools
sudo pip3 install -r requirements/requirements-linux-python3.txt
make qt5py3
python3 labelImg.py
python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

</details>

<details>
<summary><b>macOS</b></summary>

```
brew install qt  # Install qt-5.x.x by Homebrew
brew install libxml2

or using pip

pip3 install pyqt5 lxml # Install qt and lxml by pip

make qt5py3
python3 labelImg.py
python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

</details>



推荐使用Anaconda的安装方式

 首先下载并进入 [labelImg](https://github.com/tzutalin/labelImg#labelimg) 的目录

```
conda install pyqt=5
conda install -c anaconda lxml
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```





#### 安装注意事项

以Anaconda安装方式为例，比Labelme配置要麻烦一些

启动方式是通过python运行脚本`python labelImg.py <图片路径>`



#### LabelImg图片标注过程

启动labelImg后，选择图片文件或者图片所在文件夹

左侧编辑栏选择`创建区块`  绘制标注区，在弹出新的框选择对应的标签

左侧菜单栏点击保存，可以选择VOC/YOLO/CreateML三种类型的标注文件



![](https://user-images.githubusercontent.com/34162360/177526022-fd9c63d8-e476-4b63-ae02-76d032bb7656.gif)





### LabelImg标注格式

#### LabelImg导出数据格式

```
#生成标注文件
png/jpeg/jpg-->labelImg标注-->xml/txt/json
```



#### 格式转换注意事项

**PaddleDetection支持VOC或COCO格式的数据**，经LabelImg标注导出后的标注文件，需要修改为**VOC或COCO格式**，调整说明可以参考[准备训练数据](./PrepareDataSet.md#%E5%87%86%E5%A4%87%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE)
