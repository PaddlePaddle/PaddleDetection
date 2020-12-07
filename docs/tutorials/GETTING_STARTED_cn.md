# 入门使用

## 安装

`dygraph`分支需要安装每日版本的PaddlePaddle，PaddlePaddle中`c0a991c8740b413559bfc894aa5ae1d5ed3704b5`这个commit会影响精度，建议安装这个commit之前的版本。


## 训练/评估/预测

PaddleDetection提供了训练/评估/预测，支持通过不同可选参数实现特定功能

#### 训练
```bash
# GPU训练 支持单卡，多卡训练，通过CUDA_VISIBLE_DEVICES指定卡号
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --selected_gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/faster_rcnn_r50_fpn_1x_coco.yml
```

#### 评估
```bash
# 使用单卡评估
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/faster_rcnn_r50_fpn_1x_coco.yml
```

#### 预测
```bash
# 预测
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/faster_rcnn_r50_fpn_1x_coco.yml --infer_img=demo/000000570688.jpg
```

### 可选参数列表

以下列表可以通过`--help`查看

|         FLAG             |     支持脚本    |        用途        |      默认值       |         备注         |
| :----------------------: | :------------: | :---------------: | :--------------: | :-----------------: |
|          -c              |      ALL       |  指定配置文件  |  None  |  **配置模块说明请参考[配置模块](../advanced_tutorials/config_doc/CONFIG_cn.md)** |
|          -o              |      ALL       |  设置配置文件里的参数内容  |  None  |  使用-o配置相较于-c选择的配置文件具有更高的优先级。例如：`-o use_gpu=False max_iter=10000`  |  
|       --weight_type      |     train      |  加载预训练模型类型  |   pretrain  |  可设置为: `pretrain`/`finetune`/`resume`, `--weight_type='resume'`  |
|        --eval            |     train      |  是否边训练边测试  |  False  |    |
|      --output_eval       |     train/eval |  编辑评测保存json路径  |  当前路径  |  `--output_eval ./json_result`  |
|       --fp16             |     train      |  是否使用混合精度训练模式  |  False  |  需使用GPU训练  |
|       --loss_scale       |     train      |  设置混合精度训练模式中损失值的缩放比例  |  8.0  |  需先开启`--fp16`后使用  |  
|       --json_eval        |       eval     |  是否通过已存在的bbox.json或者mask.json进行评估  |  False  |  json文件路径在`--output_eval`中设置  |
|       --output_dir       |      infer     |  输出预测后可视化文件  |  `./output`  |  `--output_dir output`  |
|    --draw_threshold      |      infer     |  可视化时分数阈值  |  0.5  |  `--draw_threshold 0.7`  |
|      --infer_dir         |       infer     |  用于预测的图片文件夹路径  |  None  |    |
|      --infer_img         |       infer     |  用于预测的图片路径  |  None  |  相较于`--infer_dir`具有更高优先级  |