# 多机训练

## 简介

* 分布式训练的高性能，是飞桨的核心优势技术之一，在分类任务上，分布式训练可以达到几乎线性的加速比。
[Fleet](https://github.com/PaddlePaddle/Fleet) 是用于 PaddlePaddle 分布式训练的高层 API，基于这套接口用户可以很容易切换到分布式训练程序。
为了可以同时支持单机训练和多机训练，[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/) 采用 Fleet API 接口，可以同时支持单机训练和多机训练。

更多的分布式训练可以参考 [Fleet API设计文档](https://github.com/PaddlePaddle/Fleet/blob/develop/README.md)。


## 使用方法

* 使用`tools/train_multi_machine.py`可以启动基于Fleet的训练，目前同时支持单机单卡、单机多卡与多机多卡的训练过程。

* 可选参数列表与`tools/train.py`完全相同，可以参考[入门使用文档](./GETTING_STARTED_cn.md)。

### 单机训练

* 训练脚本如下所示。

```bash
# 设置PYTHONPATH路径
export PYTHONPATH=$PYTHONPATH:.
# 设置GPU卡号信息
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 启动训练
python -m paddle.distributed.launch \
    --selected_gpus 0,1,2,3,4,5,6,7 \
    tools/train_multi_machine.py \
        -c configs/faster_rcnn_r50_fpn_1x.yml
```

### 多机训练

* 训练脚本如下所示，其中ip1和ip2分别表示不同机器的ip地址，`PADDLE_TRAINER_ID`环境变量也是根据`cluster_node_ips`提供的ip顺序依次增大。
* 注意：在这里如果需要启动多机实验，需要保证不同的机器的运行代码是完全相同的。

```
export PYTHONPATH=$PYTHONPATH:.
# 设置GPU卡号信息
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 启动训练
node_ip=`hostname -i`
python -m paddle.distributed.launch \
     --use_paddlecloud \
    --cluster_node_ips ${ip1},${ip2} \
    --node_ip ${node_ip} \
    tools/train_multi_machine.py \
        -c configs/faster_rcnn_r50_fpn_1x.yml \
```

## 训练时间统计

* 以Faster RCNN R50_vd FPN 1x实验为例，下面给出了基于Fleet分布式训练，不同机器的训练时间对比。
    * 这里均是在V100 GPU上展开的实验。
    * 1x实验指的是8卡，单卡batch size为2时，训练的minibatch数量为90000（当训练卡数或者batch size变化时，对应的学习率和总的迭代轮数也需要变化）。



|         模型             |     训练策略 |  机器数量    | 每台机器的GPU数量  |   训练时间    | COCO bbox mAP    | 加速比 |
| :----------------------: | :------------: | :------------: | :---------------: | :----------: | :-----------: | :-----------: |
|          Faster RCNN R50_vd FPN | 1x              |      1       |  4  |  15.1h  |  38.3% | - |
|          Faster RCNN R50_vd FPN | 1x              |      2       |  4  |  9.8h  |  38.2% | 76% |
|          Faster RCNN R50_vd FPN | 1x              |      1       |  8  |  8.6h  |  38.2% | - |
|          Faster RCNN R50_vd FPN | 1x              |      2       |  8  |  5.1h  |  38.0% | 84% |

* 由上图可知，2机8卡相比于单机8卡，加速比可以达到84%，2即4卡相比于单机4卡，加速比可以达到76%，而且精度几乎没有损失。
* 1x实验相当于COCO数据集训练了约13个epoch，因此在trainer数量很多的时候，每个trainer可能无法训练完1个epoch，这会导致精度出现一些差异，这可以通过适当增加迭代轮数实现精度的对齐，我们实验发现，在训练多尺度3x实验时(配置文件：[configs/rcnn_enhance/faster_rcnn_dcn_r50_vd_fpn_3x_server_side.yml](../../configs/rcnn_enhance/faster_rcnn_dcn_r50_vd_fpn_3x_server_side.yml))，分布式训练与单机训练的模型精度是可以对齐的。
