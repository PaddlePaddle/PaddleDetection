English | [简体中文](DistributedTraining_cn.md)


## 1. Usage

### 1.1 Single-machine

* Take PP-YOLOE-s as an example, after preparing the data locally, use the interface of `paddle.distributed.launch` or `fleetrun` to start the training task. Below is an example of running the script.

```bash
fleetrun \
--selected_gpu 0,1,2,3,4,5,6,7 \
tools/train.py -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml \
--eval &>logs.txt 2>&1 &
```

### 1.2 Multi-machine

* Compared with single-machine training, when training on multiple machines, you only need to add the `--ips` parameter, which indicates the ip list of machines that need to participate in distributed training. The ips of different machines are separated by commas. Below is an example of running code.

```shell
ip_list="10.127.6.17,10.127.5.142,10.127.45.13,10.127.44.151"
fleetrun \
--ips=${ip_list} \
--selected_gpu 0,1,2,3,4,5,6,7 \
tools/train.py -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml \
--eval &>logs.txt 2>&1 &
```

**注：**
* The ip information of different machines needs to be separated by commas, which can be viewed through `ifconfig` or `ipconfig`.
* Password-free settings are required between different machines, and they can be pinged directly, otherwise the communication cannot be completed.
* The code, data, and running commands or scripts between different machines need to be consistent, and the set training commands or scripts need to be run on all machines. The first device of the first machine in the final `ip_list` is trainer0, and so on.
* The starting port of different machines may be different. It is recommended to set the same starting port for multi-machine running in different machines before starting the multi-machine task. The command is `export FLAGS_START_PORT=17000`, and the port value is recommended to be `10000~20000`.


## 2. Performance

* On single-machine and 4-machine 8-card V100 machines, model training is performed based on [PP-YOLOE-s](../../configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml). The model training time is as follows.

Machine | mAP | Time cost
-|-|-
single machine | 42.7% | 39h
4 machines | 42.1% | 13h
