简体中文 | [English](./ppvehicle_plate_en.md)

# 车牌识别任务二次开发

车牌识别任务，采用PP-OCRv3模型在车牌数据集上进行fine-tune得到，过程参考[PaddleOCR车牌应用介绍](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/applications/%E8%BD%BB%E9%87%8F%E7%BA%A7%E8%BD%A6%E7%89%8C%E8%AF%86%E5%88%AB.md)在CCPD2019数据集上进行了拓展。

## 数据准备

1. 对于CCPD2019、CCPD2020数据集，我们提供了处理脚本[ccpd2ocr_all.py](../../../deploy/pipeline/tools/ccpd2ocr_all.py), 使用时跟CCPD2019、CCPD2020数据集文件夹放在同一目录下，然后执行脚本即可在CCPD2019/PPOCR、CCPD2020/PPOCR目录下得到检测、识别模型的训练标注文件。训练时可以整合到一起使用。
2. 对于其他来源数据或者自标注数据，可以按如下格式整理训练列表文件：

- **车牌检测标注**

标注文件格式如下，中间用'\t'分隔：

```
" 图像文件路径                    标注框标注信息"
CCPD2020/xxx.jpg    [{"transcription": "京AD88888", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]
```

标注框标注信息是包含多个字典的list，有多少个标注框就有多少个字典对应，字典中的 `points` 表示车牌框的四个点的坐标(x, y)，从左上角的点开始顺时针排列。 `transcription` 表示当前文本框的文字，***当其内容为“###”时，表示该文本框无效，在训练时会跳过。***

- **车牌字符识别标注**

标注文件的格式如下，txt文件中默认请将图片路径和图片标签用'\t'分割，如用其他方式分割将造成训练报错。其中图片是对车牌字符的截图。

```
" 图像文件名                 字符标注信息 "
CCPD2020/crop_imgs/xxx.jpg   京AD88888
```

## 模型训练

首先执行以下命令clone PaddleOCR库代码到训练机器：
```
git clone git@github.com:PaddlePaddle/PaddleOCR.git
```

下载预训练模型：
```
#检测预训练模型：
mkdir models
cd models
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar
tar -xf ch_PP-OCRv3_det_distill_train.tar

#识别预训练模型：
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar
tar -xf ch_PP-OCRv3_rec_train.tar
cd ..
```

安装相关依赖环境：
```
cd PaddleOCR
pip install -r requirements.txt
```

然后进行训练相关配置修改。

### 修改配置

**检测模型配置项**

修改配置项包括以下3部分内容，可以在训练时以命令行修改，或者直接在配置文件`configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml`中修改：
1. 模型存储和训练相关:
- Global.pretrained_model: 指向前面下载的PP-OCRv3文本检测预训练模型地址
- Global.eval_batch_step: 模型多少step评估一次，一般设置为一个epoch对应的step数，可以从训练开始的log中读取。此处以[0, 772]为例，第一个数字表示从第0各step开始算起。
2. 优化器相关:
- Optimizer.lr.name: 学习率衰减器设为常量 Const
- Optimizer.lr.learning_rate: 做 fine-tune 实验，学习率需要设置的比较小，此处学习率设为配置文件中的0.05倍
- Optimizer.lr.warmup_epoch: warmup_epoch设为0
3. 数据集相关:
- Train.dataset.data_dir：指向训练集图片存放根目录
- Train.dataset.label_file_list：指向训练集标注文件
- Eval.dataset.data_dir：指向测试集图片存放根目录
- Eval.dataset.label_file_list：指向测试集标注文件

**识别模型配置项**

修改配置项包括以下3部分内容，可以在训练时以命令行修改，或者直接在配置文件`configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml`中修改：
1. 模型存储和训练相关:
- Global.pretrained_model: 指向PP-OCRv3文本识别预训练模型地址
- Global.eval_batch_step: 模型多少step评估一次，一般设置为一个epoch对应的step数，可以从训练开始的log中读取。此处以[0, 90]为例，第一个数字表示从第0各step开始算起。
2. 优化器相关
- Optimizer.lr.name: 学习率衰减器设为常量 Const
- Optimizer.lr.learning_rate: 做 fine-tune 实验，学习率需要设置的比较小，此处学习率设为配置文件中的0.05倍
- Optimizer.lr.warmup_epoch: warmup_epoch设为0
3. 数据集相关
- Train.dataset.data_dir：指向训练集图片存放根目录
- Train.dataset.label_file_list：指向训练集标注文件
- Eval.dataset.data_dir：指向测试集图片存放根目录
- Eval.dataset.label_file_list：指向测试集标注文件

### 执行训练

然后运行以下命令开始训练。如果在配置文件中已经做了修改，可以省略`-o`及其后面的内容。

**检测模型训练命令**

```
#单卡训练
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o \
    Global.pretrained_model=models/ch_PP-OCRv3_det_distill_train/student.pdparams \
    Global.save_model_dir=output/CCPD/det \
    Global.eval_batch_step="[0, 772]" \
    Optimizer.lr.name=Const \
    Optimizer.lr.learning_rate=0.0005 \
    Optimizer.lr.warmup_epoch=0 \
    Train.dataset.data_dir=/home/aistudio/ccpd_data/ \
    Train.dataset.label_file_list=[/home/aistudio/ccpd_data/train/det.txt]

#多卡训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o \
    Global.pretrained_model=models/ch_PP-OCRv3_det_distill_train/student.pdparams \
    Global.save_model_dir=output/CCPD/det \
    Global.eval_batch_step="[0, 772]" \
    Optimizer.lr.name=Const \
    Optimizer.lr.learning_rate=0.0005 \
    Optimizer.lr.warmup_epoch=0 \
    Train.dataset.data_dir=/home/aistudio/ccpd_data/ \
    Train.dataset.label_file_list=[/home/aistudio/ccpd_data/train/det.txt]

```

训练完成后可以执行以下命令进行性能评估：
```
#单卡评估
python tools/eval.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o \
    Global.pretrained_model=output/CCPD/det/best_accuracy.pdparams \
    Eval.dataset.data_dir=/home/aistudio/ccpd_data/ \
    Eval.dataset.label_file_list=[/home/aistudio/ccpd_data/test/det.txt]
```

**识别模型训练命令**

```
#单卡训练
python3 tools/train.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=models/ch_PP-OCRv3_rec_train/student.pdparams \
    Global.save_model_dir=output/CCPD/rec/ \
    Global.eval_batch_step="[0, 90]" \
    Optimizer.lr.name=Const \
    Optimizer.lr.learning_rate=0.0005 \
    Optimizer.lr.warmup_epoch=0 \
    Train.dataset.data_dir=/home/aistudio/ccpd_data \
    Train.dataset.label_file_list=[/home/aistudio/ccpd_data/train/rec.txt] \
    Eval.dataset.data_dir=/home/aistudio/ccpd_data \
    Eval.dataset.label_file_list=[/home/aistudio/ccpd_data/test/rec.txt]


#多卡训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    tools/train.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=models/ch_PP-OCRv3_rec_train/student.pdparams \
    Global.save_model_dir=output/CCPD/rec/ \
    Global.eval_batch_step="[0, 90]" \
    Optimizer.lr.name=Const \
    Optimizer.lr.learning_rate=0.0005 \
    Optimizer.lr.warmup_epoch=0 \
    Train.dataset.data_dir=/home/aistudio/ccpd_data \
    Train.dataset.label_file_list=[/home/aistudio/ccpd_data/train/rec.txt] \
    Eval.dataset.data_dir=/home/aistudio/ccpd_data \
    Eval.dataset.label_file_list=[/home/aistudio/ccpd_data/test/rec.txt]

```

训练完成后可以执行以下命令进行性能评估：
```
#单卡评估
python tools/eval.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=output/CCPD/rec/best_accuracy.pdparams \
    Eval.dataset.data_dir=/home/aistudio/ccpd_data/ \
    Eval.dataset.label_file_list=[/home/aistudio/ccpd_data/test/rec.txt]
```


### 模型导出

使用下述命令将训练好的模型导出为预测部署模型。

**检测模型导出**

```
python tools/export_model.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o \
    Global.pretrained_model=output/CCPD/det/best_accuracy.pdparams \
    Global.save_inference_dir=output/det/infer
```

**识别模型导出**

```
python tools/export_model.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=output/CCPD/rec/best_accuracy.pdparams \
    Global.save_inference_dir=output/CCPD/rec/infer
```


使用时在PP-Vehicle中的配置文件`./deploy/pipeline/config/infer_cfg_ppvehicle.yml`中修改`VEHICLE_PLATE`模块中的`det_model_dir`、`rec_model_dir`项，并开启功能`enable: True`。
```
VEHICLE_PLATE:
  det_model_dir: [YOUR_DET_INFERENCE_MODEL_PATH]                #设置检测模型路径
  det_limit_side_len: 736
  det_limit_type: "max"
  rec_model_dir: [YOUR_REC_INFERENCE_MODEL_PATH]                #设置识别模型路径
  rec_image_shape: [3, 48, 320]
  rec_batch_num: 6
  word_dict_path: deploy/pipeline/ppvehicle/rec_word_dict.txt
  enable: True                                                  #开启功能
```

然后可以使用-->至此即完成更新车牌识别模型任务。
