# 入门使用

关于配置运行环境，请参考[安装指南](INSTALL_cn.md)


## 训练/评估/预测

PaddleDetection提供了训练/评估/预测，支持通过不同可选参数实现特定功能

```bash
# 设置PYTHONPATH路径
export PYTHONPATH=$PYTHONPATH:.
# GPU训练 支持单卡，多卡训练，通过CUDA_VISIBLE_DEVICES指定卡号
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/train.py -c configs/faster_rcnn_r50_1x.yml
# GPU评估
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c configs/faster_rcnn_r50_1x.yml
# 预测
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml --infer_img=demo/000000570688.jpg
```

### 可选参数列表

以下列表可以通过`--help`查看

|         FLAG             |     支持脚本    |        用途        |      默认值       |         备注         |
| :----------------------: | :------------: | :---------------: | :--------------: | :-----------------: |
|          -c              |      ALL       |  指定配置文件  |  None  |  **配置模块说明请参考[配置模块](../advanced_tutorials/config_doc/CONFIG_cn.md)** |
|          -o              |      ALL       |  设置配置文件里的参数内容  |  None  |  使用-o配置相较于-c选择的配置文件具有更高的优先级。例如：`-o use_gpu=False max_iter=10000`  |  
|   -r/--resume_checkpoint |     train      |  从某一检查点恢复训练  |  None  |  `-r output/faster_rcnn_r50_1x/10000`  |
|        --eval            |     train      |  是否边训练边测试  |  False  |    |
|      --output_eval       |     train/eval |  编辑评测保存json路径  |  当前路径  |  `--output_eval ./json_result`  |
|       --fp16             |     train      |  是否使用混合精度训练模式  |  False  |  需使用GPU训练  |
|       --loss_scale       |     train      |  设置混合精度训练模式中损失值的缩放比例  |  8.0  |  需先开启`--fp16`后使用  |  
|       --json_eval        |       eval     |  是否通过已存在的bbox.json或者mask.json进行评估  |  False  |  json文件路径在`--output_eval`中设置  |
|       --output_dir       |      infer     |  输出预测后可视化文件  |  `./output`  |  `--output_dir output`  |
|    --draw_threshold      |      infer     |  可视化时分数阈值  |  0.5  |  `--draw_threshold 0.7`  |
|      --infer_dir         |       infer     |  用于预测的图片文件夹路径  |  None  |    |
|      --infer_img         |       infer     |  用于预测的图片路径  |  None  |  相较于`--infer_dir`具有更高优先级  |
|        --use_vdl          |   train/infer   |  是否使用[VisualDL](https://github.com/paddlepaddle/visualdl)记录数据，进而在VisualDL面板中显示  |  False  |  VisualDL需Python>=3.5    |
|        --vdl\_log_dir     |   train/infer   |  指定 VisualDL 记录数据的存储路径  |  train:`vdl_log_dir/scalar` infer: `vdl_log_dir/image`  |  VisualDL需Python>=3.5   |


## 使用示例

### 模型训练

- 边训练边测试

  ```bash
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python -u tools/train.py -c configs/faster_rcnn_r50_1x.yml --eval
  ```

  在训练中交替执行评估, 评估在每个snapshot\_iter时开始。每次评估后还会评出最佳mAP模型保存到`best_model`文件夹下。

  如果验证集很大，测试将会比较耗时，建议减少评估次数，或训练完再进行评估。


- Fine-tune其他任务

  使用预训练模型fine-tune其他任务时，可以直接加载预训练模型，形状不匹配的参数将自动忽略，例如：

  ```bash
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python -u tools/train.py -c configs/faster_rcnn_r50_1x.yml \
                         -o pretrain_weights=output/faster_rcnn_r50_1x/model_final \
  ```

  也可以显示的指定忽略参数名，可采用如下两种方式：

  1. 在YAML配置文件中设置`finetune_exclude_pretrained_params`
  2. 在命令行中添加-o finetune\_exclude\_pretrained_params对预训练模型进行选择性加载。

  ```bash
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python -u tools/train.py -c configs/faster_rcnn_r50_1x.yml \
                         -o pretrain_weights=output/faster_rcnn_r50_1x/model_final \
                            finetune_exclude_pretrained_params=['cls_score','bbox_pred']
  ```

  详细说明请参考[迁移学习文档](../advanced_tutorials/TRANSFER_LEARNING_cn.md)

- 使用Paddle OP组建的YOLOv3损失函数训练YOLOv3

  为了便于用户重新设计修改YOLOv3的损失函数，我们也提供了不使用`fluid.layer.yolov3_loss`接口而是在python代码中使用Paddle OP的方式组建YOLOv3损失函数,
  可通过如下命令用Paddle OP组建YOLOv3损失函数版本的YOLOv3模型：

  ```bash
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python -u tools/train.py -c configs/yolov3_darknet.yml \
                           -o use_fine_grained_loss=true
  ```

  Paddle OP组建YOLOv3损失函数代码位于`ppdet/modeling/losses/yolo_loss.py`

**提示:**  

- `CUDA_VISIBLE_DEVICES` 参数可以指定不同的GPU。例如: `export CUDA_VISIBLE_DEVICES=0,1,2,3`. GPU计算规则可以参考 [FAQ](../FAQ.md)
- 若本地未找到数据集，将自动下载数据集并保存在`~/.cache/paddle/dataset`中。
- 预训练模型自动下载并保存在`〜/.cache/paddle/weights`中。
- 模型checkpoints默认保存在`output`中，可通过修改配置文件中save_dir进行配置。

### 混合精度训练

通过设置 `--fp16` 命令行选项可以启用混合精度训练。目前混合精度训练已经在Faster-FPN, Mask-FPN 及 Yolov3 上进行验证，几乎没有精度损失（小于0.2 mAP)。

建议使用多进程方式来进一步加速混合精度训练。示例如下。

```bash
python -m paddle.distributed.launch --selected_gpus 0,1,2,3,4,5,6,7 tools/train.py --fp16 -c configs/faster_rcnn_r50_fpn_1x.yml
```

如果训练过程中loss出现`NaN`，请尝试调节`--loss_scale`选项数值，细节请参看混合精度训练相关的[Nvidia文档](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#mptrain)。

另外，请注意将配置文件中的 `norm_type` 由 `affine_channel` 改为 `bn`。


### 模型评估

- 指定权重和数据集路径

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python -u tools/eval.py -c configs/faster_rcnn_r50_1x.yml \
                        -o weights=https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar \
  ```

  评估模型可以为本地路径，例如`output/faster_rcnn_r50_1x/model_final`, 也可以是[MODEL_ZOO](../MODEL_ZOO_cn.md)中给出的模型链接。

- 通过json文件评估

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python -u tools/eval.py -c configs/faster_rcnn_r50_1x.yml \
             --json_eval \
             --output_eval evaluation/
  ```

  json文件必须命名为bbox.json或者mask.json，放在`evaluation/`目录下。

**提示:**

- R-CNN和SSD模型目前暂不支持多GPU评估，将在后续版本支持


### 模型预测

- 设置输出路径 && 设置预测阈值

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python -u tools/infer.py -c configs/faster_rcnn_r50_1x.yml \
                      --infer_img=demo/000000570688.jpg \
                      --output_dir=infer_output/ \
                      --draw_threshold=0.5 \
                      -o weights=output/faster_rcnn_r50_1x/model_final \
  ```


  `--draw_threshold` 是个可选参数. 根据 [NMS](https://ieeexplore.ieee.org/document/1699659) 的计算，
  不同阈值会产生不同的结果。如果用户需要对自定义路径的模型进行预测，可以设置`-o weights`指定模型路径。

  此预测过程依赖PaddleDetection源码，如果您想使用C++进行服务器端预测、或在移动端预测、或使用PaddleServing部署、或独立于PaddleDetection源码使用Python预测可以参考[模型导出教程](../advanced_tutorials/deploy/EXPORT_MODEL.md)和推理部署。
