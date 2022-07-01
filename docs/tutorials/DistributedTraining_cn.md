[English](DistributedTraining_en.md) | 简体中文


# 分布式训练

## 1. 简介

* 分布式训练指的是将训练任务按照一定方法拆分到多个计算节点进行计算，再按照一定的方法对拆分后计算得到的梯度等信息进行聚合与更新。飞桨分布式训练技术源自百度的业务实践，在自然语言处理、计算机视觉、搜索和推荐等领域经过超大规模业务检验。分布式训练的高性能，是飞桨的核心优势技术之一，PaddleDetection同时支持单机训练与多机训练。更多关于分布式训练的方法与文档可以参考：[分布式训练快速开始教程](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/ps_quick_start.html)。

## 2. 使用方法

### 2.1 单机训练

* 以PP-YOLOE-s为例，本地准备好数据之后，使用`paddle.distributed.launch`或者`fleetrun`的接口启动训练任务即可。下面为运行脚本示例。

```bash
fleetrun \
--selected_gpu 0,1,2,3,4,5,6,7 \
tools/train.py -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml \
--eval &>logs.txt 2>&1 &
```

### 2.2 多机训练

* 相比单机训练，多机训练时，只需要添加`--ips`的参数，该参数表示需要参与分布式训练的机器的ip列表，不同机器的ip用逗号隔开。下面为运行代码示例。

```shell
ip_list="10.127.6.17,10.127.5.142,10.127.45.13,10.127.44.151"
fleetrun \
--ips=${ip_list} \
--selected_gpu 0,1,2,3,4,5,6,7 \
tools/train.py -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml \
--eval &>logs.txt 2>&1 &
```

**注：**
* 不同机器的ip信息需要用逗号隔开，可以通过`ifconfig`或者`ipconfig`查看。
* 不同机器之间需要做免密设置，且可以直接ping通，否则无法完成通信。
* 不同机器之间的代码、数据与运行命令或脚本需要保持一致，且所有的机器上都需要运行设置好的训练命令或者脚本。最终`ip_list`中的第一台机器的第一块设备是trainer0，以此类推。
* 不同机器的起始端口可能不同，建议在启动多机任务前，在不同的机器中设置相同的多机运行起始端口，命令为`export FLAGS_START_PORT=17000`，端口值建议在`10000~20000`之间。


## 3. 性能效果测试

* 在单机和4机8卡V100的机器上，基于[PP-YOLOE-s](../../configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml)进行模型训练，模型的训练耗时情况如下所示。

机器 | 精度 | 耗时
-|-|-
单机8卡 | 42.7% | 39h
4机8卡 | 42.1% | 13h
