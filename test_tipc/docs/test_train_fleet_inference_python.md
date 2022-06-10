# Linux GPU/CPU 多机多卡训练推理测试

Linux GPU/CPU 多机多卡训练推理测试的主程序为`test_train_fleet_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

|   算法名称   | 模型名称 | 多机多卡 |
|:--------:|   :----:  |    :----:  |
| PP-YOLOE | ppyoloe_crn_s_300e_coco     | 分布式训练 |


- 推理相关：

|   算法名称   |           模型名称           | device_CPU | device_GPU | batchsize |
|:--------:|:------------------------:|   :----:   |  :----:  |:---------:|
| PP-YOLOE | ppyoloe_crn_s_300e_coco  |  支持 | 支持 |   1, 2    |


## 2. 测试流程

运行环境配置请参考[文档](./install.md)的内容配置TIPC的运行环境。

### 2.1 功能测试

#### 2.1.1 修改配置文件

首先，修改配置文件中的`ip`设置:  假设两台机器的`ip`地址分别为`192.168.0.1`和`192.168.0.2`，则对应的配置文件`gpu_list`字段需要修改为`gpu_list:192.168.0.1,192.168.0.2;0,1`； `ip`地址查看命令为`ifconfig`。


#### 2.1.2 准备数据

运行`prepare.sh`准备数据和模型，以配置文件`test_tipc/configs/ppyoloe/ppyoloe_crn_s_300e_coco_train_linux_gpu_fleet_normal_infer_python_linux_gpu_cpu.txt`为例，数据准备命令如下所示。

```shell
bash test_tipc/prepare.sh test_tipc/configs/ppyoloe/ppyoloe_crn_s_300e_coco_train_linux_gpu_fleet_normal_infer_python_linux_gpu_cpu.txt lite_train_lite_infer
```

**注意：** 由于是多机训练，这里需要在所有的节点上均运行启动上述命令，准备数据。

#### 2.1.3 修改起始端口并开始测试

在多机的节点上使用下面的命令设置分布式的起始端口（否则后面运行的时候会由于无法找到运行端口而hang住），一般建议设置在`10000~20000`之间。

```shell
export FLAGS_START_PORT=17000
```

以配置文件`test_tipc/configs/ppyoloe/ppyoloe_crn_s_300e_coco_train_linux_gpu_fleet_normal_infer_python_linux_gpu_cpu.txt`为例，测试方法如下所示。

```shell
bash test_tipc/test_train_inference_python.sh  test_tipc/configs/ppyoloe/ppyoloe_crn_s_300e_coco_train_linux_gpu_fleet_normal_infer_python_linux_gpu_cpu.txt lite_train_lite_infer
```

**注意：** 由于是多机训练，这里需要在所有的节点上均运行启动上述命令进行测试。


#### 2.1.4 输出结果

输出结果如下，表示命令运行成功。

```bash
 Run successfully with command - python3.7 -m paddle.distributed.launch --ips=192.168.0.1,192.168.0.2 --gpus=0,1
 tools/train.py -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml -o log_iter=1 use_gpu=True save_dir=./test_tipc/outpu
t/ppyoloe_crn_s_300e_coco/norm_train_gpus_0,1_autocast_null_nodes_2 epoch=1 pretrain_weights=https://paddledet.bj.bc
ebos.com/models/ppyoloe_crn_s_300e_coco.pdparams TrainReader.batch_size=2 filename=ppyoloe_crn_s_300e_coco    !

 ......
 Run successfully with command - python3.7 ./deploy/python/infer.py --device=cpu --enable_mkldnn=False --cpu_threads
=4 --model_dir=./test_tipc/output/ppyoloe_crn_s_300e_coco/norm_train_gpus_0,1_autocast_null_nodes_2/ppyoloe_crn_s_30
0e_coco --batch_size=2 --image_dir=./dataset/coco/test2017/ --run_benchmark=False --trt_max_shape=1600 > ./test_tipc
/output/ppyoloe_crn_s_300e_coco/python_infer_cpu_usemkldnn_False_threads_4_precision_fluid_batchsize_2.log 2>&1 !
```

**注意：** 由于分布式训练时，仅在`trainer_id=0`所在的节点中保存模型，因此其他的节点中在运行模型导出与推理时会报错，为正常现象。
