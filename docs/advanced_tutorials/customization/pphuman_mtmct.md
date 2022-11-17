简体中文 | [English](./pphuman_mtmct_en.md)

# 跨镜跟踪任务二次开发

## 数据准备

### 数据格式

跨镜跟踪使用行人REID技术实现，其训练方式采用多分类模型训练，使用时取分类softmax头部前的特征作为检索特征向量。

因此其格式与多分类任务相同。每一个行人分配一个专属id，不同行人id不同，同一行人在不同图片中的id相同。

例如图片0001.jpg、0003.jpg是同一个人，0002.jpg、0004.jpg是不同的其他行人。则标注id为：

```
0001.jpg    00001
0002.jpg    00002
0003.jpg    00001
0004.jpg    00003
...
```

依次类推。

### 数据标注

理解了上面`标注`格式的含义后，就可以进行数据标注的工作。其本质是：每张单人图建立一个标注项，对应该行人分配的id。

举例：

对于一张原始图片，

1） 使用检测框，标注图片中每一个人的位置。

2） 每一个检测框（对应每一个人），包含一个int类型的id属性。例如，上述举例中的0001.jpg中的人，对应id：1.

标注完成后利用检测框将每一个人截取成单人图，其图片与id属性标注建立对应关系。也可先截成单人图再进行标注，效果相同。

## 模型训练


数据标注完成后，就可以拿来做模型的训练，完成自定义模型的优化工作。

其主要有两步工作需要完成：1）将数据与标注数据整理成训练格式。2）修改配置文件开始训练。

### 训练数据格式

训练数据包括训练使用的图片和一个训练列表bounding_box_train.txt，其具体位置在训练配置中指定，其放置方式示例如下：

```
REID/
|-- data           训练图片文件夹
|   |-- 00001.jpg
|   |-- 00002.jpg
|   `-- 0000x.jpg
`-- bounding_box_train.txt      训练数据列表

```

bounding_box_train.txt文件内为所有训练图片名称（相对于根路径的文件路径）+ 1个id标注值

其每一行表示一个人的图片和id标注结果。其格式为：

```
0001.jpg    00001
0002.jpg    00002
0003.jpg    00001
0004.jpg    00003
```
注意：图片与标注值之间是以Tab[\t]符号隔开。该格式不能错，否则解析失败。

### 修改配置开始训练

首先执行以下命令下载训练代码（更多环境问题请参考[Install_PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/en/installation/install_paddleclas_en.md)）:

```shell
git clone https://github.com/PaddlePaddle/PaddleClas
```


需要在配置文件[softmax_triplet_with_center.yaml](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml)中，修改的配置项如下：

```
  Head:
    name: "FC"
    embedding_size: *feat_dim
    class_num: &class_num 751                   #行人id总数量

DataLoader:
  Train:
    dataset:
        name: "Market1501"
        image_root: "./dataset/"                #训练图片根路径
        cls_label_path: "bounding_box_train"    #训练文件列表


  Eval:
    Query:
      dataset:
        name: "Market1501"
        image_root: "./dataset/"                #评估图片根路径
        cls_label_path: "query"                 #评估文件列表

```
注意：

1. 这里image_root路径+bounding_box_train.txt中图片相对路径，对应图片存放的完整路径。

然后运行以下命令开始训练。

```
#多卡训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml

#单卡训练
python3 tools/train.py \
    -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml
```

训练完成后可以执行以下命令进行性能评估：
```
#多卡评估
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/eval.py \
        -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml \
        -o Global.pretrained_model=./output/strong_baseline/best_model

#单卡评估
python3 tools/eval.py \
        -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml \
        -o Global.pretrained_model=./output/strong_baseline/best_model
```

### 模型导出

使用下述命令将训练好的模型导出为预测部署模型。

```
python3 tools/export_model.py \
    -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml \
    -o Global.pretrained_model=./output/strong_baseline/best_model \
    -o Global.save_inference_dir=deploy/models/strong_baseline_inference
```

导出模型后，下载[infer_cfg.yml](https://bj.bcebos.com/v1/paddledet/models/pipeline/REID/infer_cfg.yml)文件到新导出的模型文件夹'strong_baseline_inference'中。

使用时在PP-Human中的配置文件infer_cfg_pphuman.yml中修改模型路径`model_dir`并开启功能`enable`。
```
REID:
  model_dir: [YOUR_DEPLOY_MODEL_DIR]/strong_baseline_inference/
  enable: True
```
然后可以使用。至此完成模型开发。
