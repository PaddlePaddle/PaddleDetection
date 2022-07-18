# PP-YOLOE 小目标检测模型

PaddleDetection团队提供了针对DOTA、Xview等小目标场景数据集的基于PP-YOLOE的检测模型，用户可以下载模型进行使用。


|    模型   |       数据集     |  SLICE_SIZE  |  OVERLAP_RATIO  | 类别数  | mAP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | 下载链接  | 配置文件 |
|:---------|:---------------:|:---------------:|:---------------:|:------:|:-----------------------:|:-------------------:|:---------:| :-----: |
|PP-YOLOE-l|   Xview  |  400 | 0.25 | 60 |  14.5 | 26.8 | [下载链接](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_p2_crn_l_80e_sliced_xview_400_025.pdparams) | [配置文件](./ppyoloe_p2_crn_l_80e_sliced_xview_400_025.yml) |
|PP-YOLOE-l|   DOTA   |  500 | 0.25 | 15 |  46.8 |  72.6 | [下载链接](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_p2_crn_l_80e_sliced_DOTA_500_025.pdparams) | [配置文件](./ppyoloe_p2_crn_l_80e_sliced_DOTA_500_025.yml) |


**注意:**
- **SLICE_SIZE**表示使用SAHI工具切图后子图的大小(SLICE_SIZE*SLICE_SIZE)；**OVERLAP_RATIO**表示切图重叠率。
- PP-YOLOE模型训练过程中使用8 GPUs进行混合精度训练，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- 具体使用教程请参考[ppyoloe](../ppyoloe#getting-start)。


# SAHI切图工具使用说明

## 1. 数据集下载

### DOTA据集：

DOTA数据集使用**DOTA-v1.0**水平框数据，整理后的COCO格式数据集annotations文件[下载链接](https://bj.bcebos.com/v1/paddledet/data/smalldet/DOTA.zip)，检测其中的**15类**，
包括 `plane(0), baseball-diamond(1), bridge(2), ground-track-field(3), small-vehicle(4), large-vehicle(5), ship(6), tennis-court(7),basketball-court(8), storage-tank(9), soccer-ball-field(10), roundabout(11), harbor(12), swimming-pool(13), helicopter(14)`，
图片及原始数据集[下载链接](https://captain-whu.github.io/DOAI2019/dataset.html)。

### Xview据集：

Xview数据集整理后的COCO格式数据集[下载链接](https://bj.bcebos.com/v1/paddledet/data/smalldet/DOTA.zip)，检测其中的**60类**，
具体类别为：

<details>

`Fixed-wing Aircraft(0),
Small Aircraft(1),
Cargo Plane(2),
Helicopter(3),
Passenger Vehicle(4),
Small Car(5),
Bus(6),
Pickup Truck(7),
Utility Truck(8),
Truck(9),
Cargo Truck(10),
Truck w/Box(11),
Truck Tractor(12),
Trailer(13),
Truck w/Flatbed(14),
Truck w/Liquid(15),
Crane Truck(16),
Railway Vehicle(17),
Passenger Car(18),
Cargo Car(19),
Flat Car(20),
Tank car(21),
Locomotive(22),
Maritime Vessel(23),
Motorboat(24),
Sailboat(25),
Tugboat(26),
Barge(27),
Fishing Vessel(28),
Ferry(29),
Yacht(30),
Container Ship(31),
Oil Tanker(32),
Engineering Vehicle(33),
Tower crane(34),
Container Crane(35),
Reach Stacker(36),
Straddle Carrier(37),
Mobile Crane(38),
Dump Truck(39),
Haul Truck(40),
Scraper/Tractor(41),
Front loader/Bulldozer(42),
Excavator(43),
Cement Mixer(44),
Ground Grader(45),
Hut/Tent(46),
Shed(47),
Building(48),
Aircraft Hangar(49),
Damaged Building(50),
Facility(51),
Construction Site(52),
Vehicle Lot(53),
Helipad(54),
Storage Tank(55),
Shipping container lot(56),
Shipping Container(57),
Pylon(58),
Tower(59)
`

</details>
，原始数据集[下载链接](https://challenge.xviewdataset.org/download-links)。

## 2. 统计数据集分布

首先统计所用数据集标注框的平均宽高占图片真实宽高的比例分布：

```bash
python slice_tools/distribution.py --json_path ../../dataset/DOTA/annotations/train.json --out_img dis.jpg
```
- `--json_path` ：待统计数据集COCO 格式 annotation 的json文件路径
- `--out_img` ：输出的统计分布图路径

以DOTA数据集的train数据集为例，统计结果打印如下：
```bash
Median of ratio_w is 0.03799439775910364
Median of ratio_h is 0.04074914637387802
all_img with box:  1409
all_ann:  98905
Distribution saved as dis.jpg
```

当原始数据集全部有标注框的图片中，有1/2以上的图片标注框平均宽高与原图宽高比例小于0.04时，建议进行切图训练。

## 2. SAHI切图

针对需要切图的数据集，使用[SAHI](https://github.com/obss/sahi)进行切分：

### 安装SAHI：

参考[SAHI installation](https://github.com/obss/sahi/blob/main/README.md#installation)进行安装

```bash
pip install sahi
```

### 基于SAHI切图：

```bash
python slice_tools/sahi_data_slice.py --image_dir ../../dataset/DOTA/train/ --json_path ../../dataset/DOTA/annotations/train.json --output_dir ../../dataset/dota_sliced --slice_size 500 --overlap_ratio 0.25
```

- `--image_dir` ：原始数据集图片路径
- `--json_path` ：原始数据集COCO 格式 annotation 的json文件路径
- `--output_dir` ：切分后的图片及annotation的保存路径
- `--slice_size` ：切分以后图片的长和宽的大小
- `--overlap_ratio` ：切分时的图片重叠率
- 以上述代码为例，切分后的图片文件夹与annotation共同保存在`dota_sliced`文件夹下，分别命名为`train_images_580_025`、`train_580_025.json`。
