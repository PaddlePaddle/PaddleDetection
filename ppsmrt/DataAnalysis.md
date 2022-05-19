# 数据分析功能说明

为了更好的帮助用户进行数据分析，从推荐更合适的模型，我们推出了**数据分析**功能，用户不需要上传原图，只需要上传标注好的文件格式即可进一步分析数据特点。

当前支持格式有：
* LabelMe标注数据格式
* 精灵标注数据格式
* LabelImg标注数据格式
* VOC数据格式
* COCO数据格式
* Seg数据格式

## LabelMe标注数据格式

1. 需要选定包含标注文件的zip格式压缩包。zip格式压缩包中包含一个annotations文件夹，文件夹中的内容为与标注图像相同数量的json文件，每一个json文件除后缀外与对应的图像同名。
2. 支持检测与分割任务。若提供的标注信息与所选择的任务类型不匹配，则将提示错误。
3. 对于检测任务，需提供rectangle类型标注信息；对于分割任务，需提供polygon类型标注信息。
<div align="center">
  <img src="https://user-images.githubusercontent.com/48433081/169194724-c3fff1db-78b0-4013-925b-b99e5f51e5f2.png"  width = "600" />  
</div>

## 精灵标注数据格式

1. 需要选定包含标注文件的zip格式压缩包。zip格式压缩包中包含一个annotations文件夹，文件夹中的内容为与标注图像相同数量的json文件，每一个json文件除后缀外与对应的图像同名。
2. 支持检测与分割任务。若提供的标注信息与所选择的任务类型不匹配，则将提示错误。
3. 对于检测任务，需提供bndbox或polygon类型标注信息；对于分割任务，需提供polygon类型标注信息。
<div align="center">
  <img src="https://user-images.githubusercontent.com/48433081/169194724-c3fff1db-78b0-4013-925b-b99e5f51e5f2.png"  width = "600" />  
</div>

## LabelImg标注数据格式

1. 需要选定包含标注文件的zip格式压缩包。zip格式压缩包中包含一个annotations文件夹，文件夹中的内容为与标注图像相同数量的xml文件，每一个xml文件除后缀外与对应的图像同名。
2. 仅支持检测任务。
3. 标注文件中必须提供bndbox字段信息；segmentation字段是可选的。

<div align="center">
  <img src="https://user-images.githubusercontent.com/48433081/169195232-2ccd4c07-8203-44a5-9911-97c092a228d8.png"  width = "600" />  
</div>

## VOC数据格式

1. 需要选定包含标注文件的zip格式压缩包。zip格式压缩包中包含一个annotations文件夹，文件夹中的内容为与标注图像相同数量的xml文件，每一个xml文件除后缀外与对应的图像同名。
2. 仅支持检测任务。
3. 标注文件中必须提供bndbox字段信息；segmentation字段是可选的。
<div align="center">
  <img src="https://user-images.githubusercontent.com/48433081/169195232-2ccd4c07-8203-44a5-9911-97c092a228d8.png"  width = "600" />  
</div>

## COCO数据格式

1. 需要选定包含标注文件的zip格式压缩包。zip格式压缩包中包含一个annotations文件夹，文件夹中仅存在一个名为annotation.json的文件。
2. 支持检测与分割任务。若提供的标注信息与所选择的任务类型不匹配，则将提示错误。
3. 对于检测任务，标注文件中必须包含bbox字段，segmentation字段是可选的；对于分割任务，标注文件中必须包含segmentation字段。
<div align="center">
  <img src="https://user-images.githubusercontent.com/48433081/169195416-eb12f1bb-6d18-4354-bad5-c18961aa049d.png"  width = "600" />  
</div>


## Seg数据格式

1. 需要选定包含标注文件的zip格式压缩包。zip格式压缩包中包含一个annotations文件夹，文件夹中的内容为与标注图像相同数量的png文件，每一个png文件除后缀外与对应的图像同名。
2. 仅支持分割任务。
3. 标注文件需要与原始图像在像素上严格保持一一对应，格式只可为png（后缀为.png或.PNG）。标注文件中的每个像素值为[0,255]区间内从0开始依序递增的整数ID，除255外，标注ID值的增加不能跳跃。在标注文件中，使用255表示需要忽略的像素，使用0表示背景类标注。

<div align="center">
  <img src="https://user-images.githubusercontent.com/48433081/169195389-85a9bda2-282b-452f-a809-d0100291f86f.png"  width = "600" />  
</div>
