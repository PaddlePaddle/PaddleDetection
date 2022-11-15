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

**Note:**
* The ip information of different machines needs to be separated by commas, which can be viewed through `ifconfig` or `ipconfig`.
* Password-free settings are required between different machines, and they can be pinged directly, otherwise the communication cannot be completed.
* The code, data, and running commands or scripts between different machines need to be consistent, and the set training commands or scripts need to be run on all machines. The first device of the first machine in the final `ip_list` is trainer0, and so on.
* The starting port of different machines may be different. It is recommended to set the same starting port for multi-machine running in different machines before starting the multi-machine task. The command is `export FLAGS_START_PORT=17000`, and the port value is recommended to be `10000~20000`.


## 2. Performance

* We conducted model training on 3x8 V100 GPUs. Accuracy, training time, and multi machine acceleration ratio of different models are shown below.

| Model    | Dataset | Configuration   | 8 GPU training time / Accuracy | 3x8 GPU training time / Accuracy | Acceleration ratio  |
|:---------:|:--------:|:--------:|:--------:|:--------:|:------:|
|  PP-YOLOE-s  | Objects365 | [ppyoloe_crn_s_300e_coco.yml](../../configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml)  | 301h/- | 162h/17.7%  | **1.85** |
|  PP-YOLOE-l  | Objects365 | [ppyoloe_crn_l_300e_coco.yml](../../configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml)  | 401h/- | 178h/30.3%  | **2.25** |


* We conducted model training on 4x8 V100 GPUs. Accuracy, training time, and multi machine acceleration ratio of different models are shown below.


| Model    | Dataset | Configuration   | 8 GPU training time / Accuracy | 4x8 GPU training time / Accuracy | Acceleration ratio  |
|:---------:|:--------:|:--------:|:--------:|:--------:|:------:|
|  PP-YOLOE-s  | COCO | [ppyoloe_crn_s_300e_coco.yml](../../configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml)  | 39h/42.7% | 13h/42.1%  | **3.0** |
|  PP-YOLOE-m  | Objects365 | [ppyoloe_crn_m_300e_coco.yml](../../configs/ppyoloe/ppyoloe_crn_m_300e_coco.yml)  | 337h/- | 112h/24.6%  | **3.0** |
|  PP-YOLOE-x  | Objects365 | [ppyoloe_crn_x_300e_coco.yml](../../configs/ppyoloe/ppyoloe_crn_x_300e_coco.yml)  | 464h/- | 125h/32.1%  | **3.4** |


* **Note**
    * When the number of GPU cards for training is too large, the accuracy will be slightly lost (about 1%). At this time, you can try to warmup the training process or increase some training epochs to reduce the lost.
    * The configuration files here are provided based on COCO datasets. If you need to train on other datasets, you need to modify the dataset path.
    * For the multi-machine training process of `PP-YOLOE` series, the batch size of single card is set as 8 and learning rate is same as that of single machine.
