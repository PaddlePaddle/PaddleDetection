简体中文 | [English](./ppvehicle_violation_en.md)

# 车辆违章任务二次开发

车辆违章任务的二次开发，主要集中于车道线分割模型任务。采用PP-LiteSeg模型在车道线数据集bdd100k,上进行fine-tune得到，过程参考[PP-LiteSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/configs/pp_liteseg/README.md)。

## 数据准备

ppvehicle违法分析将车道线类别分为4类
```
0 背景
1 双黄线
2 实线
3 虚线

```

1. 对于bdd100k数据集，可以结合我们的提供的处理脚本[lane_to_mask.py](../../../deploy/pipeline/tools/lane_to_mask.py)和bdd100k官方[repo](https://github.com/bdd100k/bdd100k)将数据处理成分割需要的数据格式.

```
#首先执行以下命令clone bdd100k库：
git clone https://github.com/bdd100k/bdd100k.git

#拷贝lane_to_mask.py到bdd100k目录
cp PaddleDetection/deploy/pipeline/tools/lane_to_mask.py bdd100k/

#准备bdd100k环境
cd bdd100k && pip install -r requirements.txt

#数据转换
python lane_to_mask.py -i dataset/labels/lane/polygons/lane_train.json -o /output_path

# -i bdd100k数据集label的json路径，
# -o 生成的mask图像路径

```

2. 整理数据,按如下格式存放数据
```
dataset_root
    |
    |--images  
    |  |--train
    |       |--image1.jpg
    |       |--image2.jpg
    |       |--...
    |  |--val
    |       |--image3.jpg
    |       |--image4.jpg
    |       |--...
    |  |--test
    |       |--image5.jpg
    |       |--image6.jpg
    |       |--...
    |
    |--labels  
    |  |--train
    |       |--label1.jpg
    |       |--label2.jpg
    |       |--...
    |  |--val
    |       |--label3.jpg
    |       |--label4.jpg
    |       |--...
    |  |--test
    |       |--label5.jpg
    |       |--label6.jpg
    |       |--...
    |
```
运行[create_dataset_list.py](../../../deploy/pipeline/tools/create_dataset_list.py)生成txt文件
```
python create_dataset_list.py <dataset_root> #数据根目录
                              --type  custom #数据类型，支持cityscapes、custom


```
其他数据以及数据标注，可参考PaddleSeg[准备自定义数据集](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/data/marker/marker_cn.md)


## 模型训练

首先执行以下命令clone PaddleSeg库代码到训练机器：
```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
```

安装相关依赖环境：
```
cd PaddleSeg
pip install -r requirements.txt
```

### 准备配置文件
详细可参考PaddleSeg[准备配置文件](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/docs/config/pre_config_cn.md).
本例用pp_liteseg_stdc2_bdd100k_1024x512.yml示例

```
batch_size: 16
iters: 50000

train_dataset:
  type: Dataset
  dataset_root: data/bdd100k    #数据集路径  
  train_path: data/bdd100k/train.txt #数据集训练txt文件
  num_classes: 4                     #ppvehicle将道路分为4类
  mode: train
  transforms:
    - type: ResizeStepScaling
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop
      crop_size: [512, 1024]
    - type: RandomHorizontalFlip
    - type: RandomAffine
    - type: RandomDistort
      brightness_range: 0.5
      contrast_range: 0.5
      saturation_range: 0.5
    - type: Normalize

val_dataset:
  type: Dataset
  dataset_root: data/bdd100k    #数据集路径
  val_path: data/bdd100k/val.txt #数据集验证集txt文件
  num_classes: 4
  mode: val
  transforms:
    - type: Normalize

optimizer:
  type: sgd
  momentum: 0.9
  weight_decay: 4.0e-5

lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.01 #0.01
  end_lr: 0
  power: 0.9

loss:
  types:
    - type: MixedLoss
      losses:
        - type: CrossEntropyLoss
        - type: LovaszSoftmaxLoss
      coef: [0.6, 0.4]
    - type: MixedLoss
      losses:
        - type: CrossEntropyLoss
        - type: LovaszSoftmaxLoss
      coef: [0.6, 0.4]
    - type: MixedLoss
      losses:
        - type: CrossEntropyLoss
        - type: LovaszSoftmaxLoss
      coef: [0.6, 0.4]
  coef: [1, 1,1]


model:
  type: PPLiteSeg
  backbone:
    type: STDC2
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet2.tar.gz #预训练模型
```

### 执行训练

```
#单卡训练
export CUDA_VISIBLE_DEVICES=0 # Linux上设置1张可用的卡
# set CUDA_VISIBLE_DEVICES=0  # Windows上设置1张可用的卡

python train.py \
       --config configs/pp_liteseg/pp_liteseg_stdc2_bdd100k_1024x512.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output

```
### 训练参数解释
```
--do_eval 是否在保存模型时启动评估, 启动时将会根据mIoU保存最佳模型至best_model
--use_vdl 是否开启visualdl记录训练数据
--save_interval 500  模型保存的间隔步数
--save_dir output    模型输出路径
```

## 2、多卡训练
如果想要使用多卡训练的话，需要将环境变量CUDA_VISIBLE_DEVICES指定为多卡（不指定时默认使用所有的gpu)，并使用paddle.distributed.launch启动训练脚本（windows下由于不支持nccl，无法使用多卡训练）:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3 # 设置4张可用的卡
python -m paddle.distributed.launch train.py \
       --config configs/pp_liteseg/pp_liteseg_stdc2_bdd100k_1024x512.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```


训练完成后可以执行以下命令进行性能评估：
```
#单卡评估
python val.py \
       --config configs/pp_liteseg/pp_liteseg_stdc2_bdd100k_1024x512.yml \
       --model_path output/iter_1000/model.pdparams
```


### 模型导出

使用下述命令将训练好的模型导出为预测部署模型。

```
python export.py \
       --config configs/pp_liteseg/pp_liteseg_stdc2_bdd100k_1024x512.yml \
       --model_path output/iter_1000/model.pdparams \
       --save_dir output/inference_model
```


使用时在PP-Vehicle中的配置文件`./deploy/pipeline/config/infer_cfg_ppvehicle.yml`中修改`LANE_SEG`模块中的`model_dir`项.
```
LANE_SEG:
  lane_seg_config: deploy/pipeline/config/lane_seg_config.yml  
  model_dir:  output/inference_model
```

然后可以使用-->至此即完成更新车道线分割模型任务。
