# 开始

关于配置运行环境，请参考[安装指南](INSTALL_cn.md)


## 训练


#### 单GPU训练


```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
python tools/train.py -c configs/faster_rcnn_r50_1x.yml
```

#### 多GPU训练


```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:.
python tools/train.py -c configs/faster_rcnn_r50_1x.yml
```

#### CPU训练

```bash
export CPU_NUM=8
export PYTHONPATH=$PYTHONPATH:.
python tools/train.py -c configs/faster_rcnn_r50_1x.yml -o use_gpu=false
```

##### 可选参数

- `-r` or `--resume_checkpoint`: 从某一检查点恢复训练，例如: `-r output/faster_rcnn_r50_1x/10000`
- `--eval`: 是否边训练边测试，默认是 `False`
- `--output_eval`: 如果边训练边测试, 这个参数可以编辑评测保存json路径, 默认是当前目录。
- `-d` or `--dataset_dir`: 数据集路径, 同配置文件里的`dataset_dir`. 例如: `-d dataset/coco`
- `-o`: 设置配置文件里的参数内容。 例如: `-o max_iters=180000`

##### 例子

- 边训练边测试
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:.
python -u tools/train.py -c configs/faster_rcnn_r50_1x.yml --eval
```
可通过设置`--eval`在训练epoch中交替执行评估, 评估在每个snapshot_iter时开始。可在配置文件的`snapshot_iter`处修改。
如果验证集很大，测试将会比较耗时，影响训练速度，建议减少评估次数，或训练完再进行评估。当边训练边测试时，在每次snapshot_iter会评测出最佳mAP模型保存到
`best_model`文件夹下，`best_model`的路径和`model_final`的路径相同。

- 设置配置文件参数 && 指定数据集路径
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:.
python -u tools/train.py -c configs/faster_rcnn_r50_1x.yml \
                         -d dataset/coco
```

##### 提示

- `CUDA_VISIBLE_DEVICES` 参数可以指定不同的GPU。例如: `export CUDA_VISIBLE_DEVICES=0,1,2,3`. GPU计算规则可以参考 [FAQ](#faq)
- 数据集默认存储在`dataset/coco`中（可配置）。
- 若本地未找到数据集，将自动下载数据集并保存在`~/.cache/paddle/dataset`中。
- 预训练模型自动下载并保存在`〜/.cache/paddle/weights`中。
- 模型checkpoints默认保存在`output`中（可配置）。
- 更多参数配置，请参考[配置文件](../configs)。
- RCNN系列模型CPU训练在PaddlePaddle 1.5.1及以下版本暂不支持，将在下个版本修复。


## 评估


```bash
# GPU评估
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
python tools/eval.py -c configs/faster_rcnn_r50_1x.yml
```

#### 可选参数

- `-d` or `--dataset_dir`: 数据集路径, 同配置文件里的`dataset_dir`。例如: `-d dataset/coco`
- `--output_eval`: 这个参数可以编辑评测保存json路径, 默认是当前目录。
- `-o`: 设置配置文件里的参数内容。 例如: `-o weights=output/faster_rcnn_r50_1x/model_final`
- `--json_eval`: 是否通过已存在的bbox.json或者mask.json进行评估。默认是`False`。json文件路径通过`-f`指令来设置。

#### 例子

- 设置配置文件参数 && 指定数据集路径
```bash
# GPU评估
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
python -u tools/eval.py -c configs/faster_rcnn_r50_1x.yml \
                        -o weights=output/faster_rcnn_r50_1x/model_final \
                        -d dataset/coco
```

- 通过json文件评估
```bash
# GPU评估
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
python tools/eval.py -c configs/faster_rcnn_r50_1x.yml \
             --json_eval \
             -f evaluation/
```

json文件必须命名为bbox.json或者mask.json，放在`evaluation/`目录下，或者不加`-f`参数，默认为当前目录。

#### 提示

- 默认从`output`加载checkpoint（可配置）
- R-CNN和SSD模型目前暂不支持多GPU评估，将在后续版本支持


## 推断


- 单图片推断

```bash
# GPU推断
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml --infer_img=demo/000000570688.jpg
```

- 多图片推断

```bash
# GPU推断
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml --infer_dir=demo
```

#### 可选参数

- `--output_dir`: 输出推断后可视化文件。
- `--draw_threshold`: 设置推断的阈值。默认是0.5.
- `--save_inference_model`: 设为`True`时，将预测模型保存到output_dir中.

#### 例子

- 设置输出路径 && 设置推断阈值
```bash
# GPU推断
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml \
                      --infer_img=demo/000000570688.jpg \
                      --output_dir=infer_output/ \
                      --draw_threshold=0.5 \
                      -o weights=output/faster_rcnn_r50_1x/model_final
```


可视化文件默认保存在`output`中，可通过`--output_dir=`指定不同的输出路径。  
`--draw_threshold` 是个可选参数. 根据 [NMS](https://ieeexplore.ieee.org/document/1699659) 的计算，不同阈值会产生不同的结果。如果用户需要对自定义路径的模型进行推断，可以设置`-o weights`指定模型路径。

- 保存推断模型

```bash
# GPU推断
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml --infer_img=demo/000000570688.jpg \
                      --save_inference_model
```

通过设置`--save_inference_model`保存可供PaddlePaddle预测库加载的推断模型。


## FAQ

**Q:**  为什么我使用单GPU训练loss会出`NaN`? </br>
**A:**  默认学习率是适配多GPU训练(8x GPU)，若使用单GPU训练，须对应调整学习率（例如，除以8）。  
计算规则表如下所示，它们是等价的: </br>  


| GPU数  | 学习率  | 最大轮数 | 变化节点       |  
| :---------: | :------------: | :-------: | :--------------: |  
| 2           | 0.0025         | 720000    | [480000, 640000] |
| 4           | 0.005          | 360000    | [240000, 320000] |
| 8           | 0.01           | 180000    | [120000, 160000] |


**Q:**  如何减少GPU显存使用率? </br>
**A:**  可通过设置环境变量`FLAGS_conv_workspace_size_limit`为较小的值来减少显存消耗，并且不
会影响训练速度。以Mask-RCNN（R50）为例，设置`export FLAGS_conv_workspace_size_limit = 512`，
batch size可以达到每GPU 4 (Tesla V100 16GB)。
