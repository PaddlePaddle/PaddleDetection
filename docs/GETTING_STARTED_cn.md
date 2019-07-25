# 开始

关于配置运行环境，请参考[安装指南](INSTALL_cn.md)


## 训练


#### 单GPU训练


```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/faster_rcnn_r50_1x.yml
```

#### 多GPU训练


```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/train.py -c configs/faster_rcnn_r50_1x.yml
```

- 数据集默认存储在`dataset/coco`中（可配置）。
- 若本地未找到数据集，将自动下载数据集并保存在`~/.cache/paddle/dataset`中。
- 预训练模型自动下载并保存在`〜/.cache/paddle/weights`中。
- 模型checkpoints默认保存在`output`中（可配置）。
- 更多参数配置，请参考配置文件。


可通过设置`--eval`在训练epoch中交替执行评估（已在在Pascal-VOC数据集上
用`SSD`检测器验证，不推荐在COCO数据集上的两阶段模型上执行交替评估）


## 评估


```bash
export CUDA_VISIBLE_DEVICES=0
# 若使用CPU，则执行
# export CPU_NUM=1
python tools/eval.py -c configs/faster_rcnn_r50_1x.yml
```

- 默认从`output`加载checkpoint（可配置）
- R-CNN和SSD模型目前暂不支持多GPU评估，将在后续版本支持


## 推断


- 单图片推断

```bash
export CUDA_VISIBLE_DEVICES=0
# 若使用CPU，则执行
# export CPU_NUM=1
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml --infer_img=demo/000000570688.jpg
```

- 多图片推断

```bash
export CUDA_VISIBLE_DEVICES=0
# 若使用CPU，则执行
# export CPU_NUM=1
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml --infer_dir=demo
```

可视化文件默认保存在`output`中，可通过`--output_dir=`指定不同的输出路径。

- 保存推断模型

```bash
export CUDA_VISIBLE_DEVICES=0
# or run on CPU with:
# export CPU_NUM=1
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml --infer_img=demo/000000570688.jpg \
                      --save_inference_model
```

通过设置`--save_inference_model`保存可供PaddlePaddle预测库加载的推断模型。


## FAQ

**Q:**  为什么我使用单GPU训练loss会出`NaN`? </br>
**A:**  默认学习率是适配多GPU训练(8x GPU)，若使用单GPU训练，须对应调整学习率（例如，除以8）。


**Q:**  如何减少GPU显存使用率? </br>
**A:**  可通过设置环境变量`FLAGS_conv_workspace_size_limit`为较小的值来减少显存消耗，并且不
会影响训练速度。以Mask-RCNN（R50）为例，设置`export FLAGS_conv_workspace_size_limit = 512`，
batch size可以达到每GPU 4 (Tesla V100 16GB)。
