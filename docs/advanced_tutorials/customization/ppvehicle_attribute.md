简体中文 | [English](./ppvehicle_attribute_en.md)

# 车辆属性识别任务二次开发

## 数据准备

### 数据格式

车辆属性模型采用VeRi数据集的属性，共计10种车辆颜色及9种车型, 具体如下：
```
# 车辆颜色
- "yellow"
- "orange"
- "green"
- "gray"
- "red"
- "blue"
- "white"
- "golden"
- "brown"
- "black"

# 车型
- "sedan"
- "suv"
- "van"
- "hatchback"
- "mpv"
- "pickup"
- "bus"
- "truck"
- "estate"
```

在标注文件中使用长度为19的序列来表示上述属性。

举例：

[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

前10位中，位序号0的值为1，表示车辆颜色为`"yellow"`。

后9位中，位序号11的值为1，表示车型为`"suv"`。


### 数据标注

理解了上面`数据格式`的含义后，就可以进行数据标注的工作。其本质是：每张车辆的图片，建立一组长度为19的标注项，分别对应各项属性值。

举例：

对于一张原始图片，

1） 使用检测框，标注图片中每台车辆的位置。

2） 每一个检测框（对应每辆车），包含一组19位的属性值数组，数组的每一位以0或1表示。对应上述19个属性分类。例如，如果颜色是'orange'，则数组索引为1的位置值为1，如果车型是'sedan'，则数组索引为10的位置值为1。

标注完成后利用检测框将每辆车截取成只包含单辆车的图片，则图片与19位属性标注建立了对应关系。也可先截取再进行标注，效果相同。


## 模型训练

数据标注完成后，就可以拿来做模型的训练，完成自定义模型的优化工作。

其主要有两步工作需要完成：1）将数据与标注数据整理成训练格式。2）修改配置文件开始训练。

### 训练数据格式

训练数据包括训练使用的图片和一个训练列表train.txt，其具体位置在训练配置中指定，其放置方式示例如下：
```
Attribute/
|-- data           训练图片文件夹
|   |-- 00001.jpg
|   |-- 00002.jpg
|   `-- 0000x.jpg
`-- train.txt      训练数据列表

```

train.txt文件内为所有训练图片名称（相对于根路径的文件路径）+ 19个标注值

其每一行表示一辆车的图片和标注结果。其格式为：

```
00001.jpg    0,0,1,0,....
```

注意：1)图片与标注值之间是以Tab[\t]符号隔开, 2)标注值之间是以逗号[,]隔开。该格式不能错，否则解析失败。

### 修改配置开始训练

首先执行以下命令下载训练代码（更多环境问题请参考[Install_PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/en/installation/install_paddleclas_en.md)）:

```shell
git clone https://github.com/PaddlePaddle/PaddleClas
```

需要在[配置文件](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0.yaml)中，修改的配置项如下：
```yaml
DataLoader:
  Train:
    dataset:
      name: MultiLabelDataset
      image_root: "dataset/VeRi/"                     # the root path of training images
      cls_label_path: "dataset/VeRi/train_list.txt"   # the location of the training list file
      label_ratio: True
      transform_ops:
        ...

  Eval:
    dataset:
      name: MultiLabelDataset
      image_root: "dataset/VeRi/"                     # the root path of evaluation images
      cls_label_path: "dataset/VeRi/val_list.txt"     # the location of the evaluation list file
      label_ratio: True
      transform_ops:
         ...
```

注意：
1. 这里image_root路径+train.txt中图片相对路径，对应图片的完整路径位置。
2. 如果有修改属性数量，则还需修改内容配置项中属性种类数量：
```yaml
# model architecture
Arch:
  name: "PPLCNet_x1_0"
  pretrained: True
  use_ssld: True
  class_num: 19           #属性种类数量
```

然后运行以下命令开始训练。

```
#多卡训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0.yaml

#单卡训练
python3 tools/train.py \
        -c ./ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0.yaml
```

训练完成后可以执行以下命令进行性能评估：
```
#多卡评估
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/eval.py \
        -c ./ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0.yaml \
        -o Global.pretrained_model=./output/PPLCNet_x1_0/best_model

#单卡评估
python3 tools/eval.py \
        -c ./ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0.yaml \
        -o Global.pretrained_model=./output/PPLCNet_x1_0/best_model
```


### 模型导出

使用下述命令将训练好的模型导出为预测部署模型。

```
python3 tools/export_model.py \
    -c ./ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_vehicle_attribute_model
```

导出模型后，如果希望在PP-Vehicle中使用，则需要下载[预测部署模型](https://bj.bcebos.com/v1/paddledet/models/pipeline/vehicle_attribute_model.zip)，解压并将其中的配置文件`infer_cfg.yml`文件，放置到导出的模型文件夹`PPLCNet_x1_0_vehicle_attribute_model`中。

使用时在PP-Vehicle中的配置文件`./deploy/pipeline/config/infer_cfg_ppvehicle.yml`中修改新的模型路径`model_dir`项，并开启功能`enable: True`。
```
VEHICLE_ATTR:
  model_dir: [YOUR_DEPLOY_MODEL_DIR]/PPLCNet_x1_0_vehicle_attribute_infer/   #新导出的模型路径位置
  enable: True                                                              #开启功能
```
然后可以使用-->至此即完成新增属性类别识别任务。

## 属性增减

该过程与行人属性的增减过程相似，如果需要增加、减少属性数量，则需要：

1)标注时需增加新属性类别信息或删减属性类别信息；

2)对应修改训练中train.txt所使用的属性数量和名称；

3)修改训练配置，例如``PaddleClas/blob/develop/ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0.yaml``文件中的属性数量,详细见上述`修改配置开始训练`部分。

增加属性示例：

1. 在标注数据时在19位后继续增加新的属性标注数值；
2. 在train.txt文件的标注数值中也增加新的属性数值。
3. 注意属性类型在train.txt中属性数值列表中的位置的对应关系需要固定。

<div width="500" align="center">
  <img src="../../images/add_attribute.png"/>
</div>

删减属性同理。


## 修改后处理代码

修改了属性定义后，pipeline后处理部分也需要做相应修改，主要影响结果可视化时的显示结果。

相应代码在[文件](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/deploy/pipeline/ppvehicle/vehicle_attr.py#L108)中`postprocess`函数。

其函数实现说明如下：

```python
    # 在类的初始化函数中，定义了颜色/车型的名称
    self.color_list = [
        "yellow", "orange", "green", "gray", "red", "blue", "white",
        "golden", "brown", "black"
    ]
    self.type_list = [
        "sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus", "truck",
        "estate"
    ]

    ...

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        im_results = result['output']
        batch_res = []
        for res in im_results:
            res = res.tolist()
            attr_res = []
            color_res_str = "Color: "
            type_res_str = "Type: "
            color_idx = np.argmax(res[:10])   # 前10项表示各项颜色得分，取得分最大项作为颜色结果
            type_idx = np.argmax(res[10:])    # 后9项表示各项车型得分，取得分最大项作为车型结果

            # 颜色和车型的得分都需要超过对应阈值，否则视为'UnKnown'
            if res[color_idx] >= self.color_threshold:
                color_res_str += self.color_list[color_idx]
            else:
                color_res_str += "Unknown"
            attr_res.append(color_res_str)

            if res[type_idx + 10] >= self.type_threshold:
                type_res_str += self.type_list[type_idx]
            else:
                type_res_str += "Unknown"
            attr_res.append(type_res_str)

            batch_res.append(attr_res)
        result = {'output': batch_res}
        return result
```
