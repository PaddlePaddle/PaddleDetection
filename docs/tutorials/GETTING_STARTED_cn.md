[English](GETTING_STARTED.md) | 简体中文


# 入门使用

## 安装

关于安装配置运行环境，请参考[安装指南](INSTALL_cn.md)


## 准备数据
- 首先按照[准备数据文档](PrepareDataSet.md) 准备数据。  
- 然后设置`configs/datasets`中相应的coco或voc等数据配置文件中的数据路径。


## 训练/评估/预测

PaddleDetection提供了`训练`/`评估`/`预测`等功能，支持通过传入不同可选参数实现特定功能

```bash
# GPU单卡训练
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml
# GPU多卡训练
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml
# GPU评估
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml
# 预测
python tools/infer.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml --infer_img=demo/000000570688.jpg
```


### 参数列表

以下列表可以通过`--help`查看

|         FLAG             |     支持脚本    |        用途        |      默认值       |         备注         |
| :----------------------: | :------------: | :---------------: | :--------------: | :-----------------: |
|          -c              |      ALL       |  指定配置文件  |  None  |  **必选**，例如-c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml |
|          -o              |      ALL       |  设置或更改配置文件里的参数内容  |  None  |  相较于`-c`设置的配置文件有更高优先级，例如：`-o use_gpu=False`  |
|        --eval            |     train      |  是否边训练边测试  |  False  |  如需指定，直接`--eval`即可 |
|   -r/--resume_checkpoint |     train      |  恢复训练加载的权重路径  |  None  |  例如：`-r output/faster_rcnn_r50_1x_coco/10000`  |
|       --slim_config             |     ALL      |  模型压缩策略配置文件  |  None  |  例如`--slim_config configs/slim/prune/yolov3_prune_l1_norm.yml`  |
|        --use_vdl          |   train/infer   |  是否使用[VisualDL](https://github.com/paddlepaddle/visualdl)记录数据，进而在VisualDL面板中显示  |  False  |  VisualDL需Python>=3.5   |
|        --vdl\_log_dir     |   train/infer   |  指定 VisualDL 记录数据的存储路径  |  train:`vdl_log_dir/scalar` infer: `vdl_log_dir/image`  |  VisualDL需Python>=3.5   |
|      --output_eval       |   eval |  评估阶段保存json路径  | None  |  例如 `--output_eval=eval_output`, 默认为当前路径  |
|       --json_eval        |       eval     |  是否通过已存在的bbox.json或者mask.json进行评估  |  False  |  如需指定，直接`--json_eval`即可， json文件路径在`--output_eval`中设置  |
|      --classwise         |       eval     |  是否评估单类AP和绘制单类PR曲线  |  False  |  如需指定，直接`--classwise`即可 |
|       --output_dir       |      infer/export_model     |  预测后结果或导出模型保存路径  |  `./output`  |  例如`--output_dir=output`  |
|    --draw_threshold      |      infer     |  可视化时分数阈值  |  0.5  |  例如`--draw_threshold=0.7`  |
|      --infer_dir         |       infer     |  用于预测的图片文件夹路径  |  None  |    `--infer_img`和`--infer_dir`必须至少设置一个 |
|      --infer_img         |       infer     |  用于预测的图片路径  |  None  |  `--infer_img`和`--infer_dir`必须至少设置一个，`infer_img`具有更高优先级  |
|      --save_txt          |       infer     |  是否在文件夹下将图片的预测结果保存到文本文件中        |  False  |  可选  |

## 使用示例

### 模型训练

- 边训练边评估

  ```bash
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml --eval
  ```

  在训练中交替执行评估, 评估在每个epoch训练结束后开始。每次评估后还会评出最佳mAP模型保存到`best_model`文件夹下。

  如果验证集很大，测试将会比较耗时，建议调整`configs/runtime.yml` 文件中的 `snapshot_epoch`配置以减少评估次数，或训练完成后再进行评估。


- Fine-tune其他任务

  使用预训练模型fine-tune其他任务时，可以直接加载预训练模型，形状不匹配的参数将自动忽略，例如：

  ```bash
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  # 如果模型中参数形状与加载权重形状不同，将不会加载这类参数
  python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
                           -o pretrain_weights=output/faster_rcnn_r50_1x_coco/model_final \
  ```

**提示:**  

- `CUDA_VISIBLE_DEVICES` 参数可以指定不同的GPU。例如: `export CUDA_VISIBLE_DEVICES=0,1,2,3`
- 若本地未找到数据集，将自动下载数据集并保存在`~/.cache/paddle/dataset`中。
- 预训练模型自动下载并保存在`〜/.cache/paddle/weights`中。
- 模型checkpoints默认保存在`output`中，可通过修改配置文件`configs/runtime.yml`中`save_dir`进行配置。


### 模型评估

- 指定权重和数据集路径

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python -u tools/eval.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
                          -o weights=https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams
  ```

  评估模型可以为本地路径，例如`output/faster_rcnn_r50_1x_coco/model_final`, 也可以是[MODEL_ZOO](../MODEL_ZOO_cn.md)中给出的模型链接。


- 通过json文件评估

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python tools/eval.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
             --json_eval \
             -output_eval evaluation/
  ```

  json文件必须命名为bbox.json或者mask.json，放在`evaluation/`目录下。



### 模型预测

- 设置输出路径 && 设置预测阈值

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python tools/infer.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
                      --infer_img=demo/000000570688.jpg \
                      --output_dir=infer_output/ \
                      --draw_threshold=0.5 \
                      -o weights=output/faster_rcnn_r50_fpn_1x_coco/model_final \
                      --use_vdl=Ture
  ```

  `--draw_threshold` 是个可选参数. 根据 [NMS](https://ieeexplore.ieee.org/document/1699659) 的计算，
  不同阈值会产生不同的结果。



## 预测部署

请参考[预测部署文档](../../deploy/README.md)。


## 模型压缩

请参考[模型压缩文档](../../configs/slim/README.md)。
