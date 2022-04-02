# 非结构化稀疏在 PicoDet 上的应用教程

## 1. 介绍
在模型压缩中，常见的稀疏方式为结构化稀疏和非结构化稀疏，前者在某个特定维度（特征通道、卷积核等等）上对卷积、矩阵乘法进行剪枝操作，然后生成一个更小的模型结构，这样可以复用已有的卷积、矩阵乘计算，无需特殊实现推理算子；后者以每一个参数为单元进行稀疏化，然而并不会改变参数矩阵的形状，所以更依赖于推理库、硬件对于稀疏后矩阵运算的加速能力。我们在 PP-PicoDet （以下简称PicoDet） 模型上运用了非结构化稀疏技术，在精度损失较小时，获得了在 ARM CPU 端推理的显著性能提升。本文档会介绍如何非结构化稀疏训练 PicoDet，关于非结构化稀疏的更多介绍请参照[这里](https://github.com/PaddlePaddle/PaddleSlim/tree/release/2.4/demo/dygraph/unstructured_pruning)。

## 2. 版本要求
```bash
PaddlePaddle >= 2.1.2
PaddleSlim develop分支 （pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple）
```

## 3. 数据准备
同 PicoDet

## 4. 预训练模型
在非结构化稀疏训练中，我们规定预训练模型是已经收敛完成的模型参数，所以需要额外在相关配置文件中声明。

声明预训练模型地址的配置文件：./configs/picodet/pruner/picodet_m_320_coco_pruner.yml
预训练模型地址请参照 PicoDet 文档：./configs/picodet/README.md

## 5. 自定义稀疏化的作用范围
为达到最佳推理加速效果，我们建议只对 1x1 卷积层进行稀疏化，其他层参数保持稠密。另外，有些层对于精度影响较大（例如head的最后几层，se-block的若干层），我们同样不建议对他们进行稀疏化，我们支持开发者通过传入自定义函数的形式，方便的指定哪些层不参与稀疏。例如，基于picodet_m_320这个模型，我们稀疏时跳过了后4层卷积以及6层se-block中的卷积，自定义函数如下：

```python
NORMS_ALL = [ 'BatchNorm', 'GroupNorm', 'LayerNorm', 'SpectralNorm', 'BatchNorm1D',
    'BatchNorm2D', 'BatchNorm3D', 'InstanceNorm1D', 'InstanceNorm2D',
    'InstanceNorm3D', 'SyncBatchNorm', 'LocalResponseNorm' ]

def skip_params_self(model):
    skip_params = set()
    for _, sub_layer in model.named_sublayers():
        if type(sub_layer).__name__.split('.')[-1] in NORMS_ALL:
            skip_params.add(sub_layer.full_name())
        for param in sub_layer.parameters(include_sublayers=False):
            cond_is_conv1x1 = len(param.shape) == 4 and param.shape[2] == 1 and param.shape[3] == 1
            cond_is_head_m = cond_is_conv1x1 and param.shape[0] == 112 and param.shape[1] == 128
            cond_is_se_block_m = param.name.split('.')[0] in ['conv2d_17', 'conv2d_18', 'conv2d_56', 'conv2d_57', 'conv2d_75', 'conv2d_76']
            if not cond_is_conv1x1 or cond_is_head_m or cond_is_se_block_m:
                skip_params.add(param.name)
    return skip_params
```

## 6. 训练
我们已经将非结构化稀疏的核心功能通过 API 调用的方式嵌入到了训练中，所以如果您没有更细节的需求，直接运行 6.1 的命令启动训练即可。同时，为帮助您根据自己的需求更改、适配代码，我们也提供了更为详细的使用介绍，请参照 6.2。

### 6.1 直接使用
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3.7 -m paddle.distributed.launch --log_dir=log_test --gpus 0,1,2,3 tools/train.py -c configs/picodet/pruner/picodet_m_320_coco_pruner.yml --slim_config configs/slim/prune/picodet_m_unstructured_prune_75.yml --eval
```

### 6.2 详细介绍
- 自定义稀疏化的作用范围：可以参照本教程的第 5 节
- 如何添加稀疏化训练所需的 4 行代码

```python
# after constructing model and before training

# Pruner Step1: configs
configs = {
    'pruning_strategy': 'gmp',
    'stable_iterations': self.stable_epochs * steps_per_epoch,
    'pruning_iterations': self.pruning_epochs * steps_per_epoch,
    'tunning_iterations': self.tunning_epochs * steps_per_epoch,
    'resume_iteration': 0,
    'pruning_steps': self.pruning_steps,
    'initial_ratio': self.initial_ratio,
}

# Pruner Step2: construct a pruner object
self.pruner = GMPUnstructuredPruner(
    model,
    ratio=self.cfg.ratio,
    skip_params_func=skip_params_self, # Only pass in this value when you design your own skip_params function. And the following argument (skip_params_type) will be ignored.
    skip_params_type=self.cfg.skip_params_type,
    local_sparsity=True,
    configs=configs)

# training
for epoch_id in range(self.start_epoch, self.cfg.epoch):
    model.train()
    for step_id, data in enumerate(self.loader):
        # model forward
        outputs = model(data)
        loss = outputs['loss']
        # model backward
        loss.backward()
        self.optimizer.step()

        # Pruner Step3: step during training
        self.pruner.step()

    # Pruner Step4: save the sparse model
    self.pruner.update_params()
    # model-saving API
```

## 7. 模型评估与推理部署
这部分与 PicoDet 文档中基本一致，只是在转换到 PaddleLite 模型时，需要添加一个输入参数（sparse_model）：

```bash
paddle_lite_opt --model_dir=inference_model/picodet_m_320_coco --valid_targets=arm --optimize_out=picodet_m_320_coco_fp32_sparse --sparse_model=True
```

**注意：** 目前稀疏化推理适用于 PaddleLite的 FP32 和 INT8 模型，所以执行上述命令时，请不要打开 FP16 开关。

## 8. 稀疏化结果
我们在75%和85%稀疏度下，训练得到了 FP32 PicoDet-m模型，并在 SnapDragon-835设备上实测推理速度，效果如下表。其中：
- 对于 m 模型，mAP损失1.5，获得了 34\%-58\% 的加速性能
- 同样对于 m 模型，除4线程推理速度基本持平外，单线程推理速度、mAP、模型体积均优于 s 模型。


| Model     | Input size | Sparsity | mAP<sup>val<br>0.5:0.95 | Size<br><sup>(MB) | Latency single-thread<sup><small>[Lite](#latency)</small><sup><br><sup>(ms) |  speed-up single-thread |  Latency 4-thread<sup><small>[Lite](#latency)</small><sup><br><sup>(ms) |  speed-up 4-thread |  Download  | SlimConfig |
| :-------- | :--------: |:--------: | :---------------------: | :----------------: | :----------------: |:----------------: | :---------------: | :-----------------------------: | :-----------------------------: | :----------------------------------------: |
| PicoDet-m-1.0 |  320*320   |   0      |          30.9         | 8.9 |  127     | 0    |  43     |    0       | [model](https://paddledet.bj.bcebos.com/models/picodet_m_320_coco.pdparams)&#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_m_320_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/picodet/picodet_m_320_coco.yml)|
| PicoDet-m-1.0 |  320*320   |   75%    |          29.4         | 5.6 |  **80**  | 58%  | **32**  |   34%      | [model](https://paddledet.bj.bcebos.com/models/slim/picodet_m_320__coco_sparse_75.pdparams)&#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_m_320__coco_sparse_75.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/slim/prune/picodet_m_unstructured_prune_75.yml)|
| PicoDet-s-1.0 |  320*320   |   0      |          27.1         | 4.6 |    68    |  0   |    26   |    0       | [model](https://paddledet.bj.bcebos.com/models/picodet_s_320_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_s_320_coco.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/picodet/picodet_s_320_coco.yml)|
| PicoDet-m-1.0 |  320*320   |   85%    |          27.6         | 4.1 |  **65**  | 96%  |  **27** |   59%      | [model](https://paddledet.bj.bcebos.com/models/slim/picodet_m_320__coco_sparse_85.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_picodet_m_320__coco_sparse_85.log) | [config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/slim/prune/picodet_m_unstructured_prune_85.yml)|

**注意：**
- 上述模型体积是**部署模型体积**，即 PaddleLite 转换得到的 *.nb 文件的体积。
- 加速一栏我们按照 FPS 增加百分比计算，即：$(dense\_latency - sparse\_latency) / sparse\_latency$
- 上述稀疏化训练时，我们额外添加了一种数据增强方式到 _base_/picodet_320_reader.yml，代码如下。但是不添加的话，预期mAP也不会有明显下降（<0.1），且对速度和模型体积没有影响。
```yaml
worker_num: 6
TrainReader:
  sample_transforms:
  - Decode: {}
  - RandomCrop: {}
  - RandomFlip: {prob: 0.5}
  - RandomExpand: {fill_value: [123.675, 116.28, 103.53]}
  - RandomDistort: {}
  batch_transforms:
etc.
```
