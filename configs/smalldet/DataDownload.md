# 小目标数据集下载汇总

## 目录
- [数据集准备](#数据集准备)
    - [VisDrone-DET](#VisDrone-DET)
    - [DOTA水平框](#DOTA水平框)
    - [Xview](#Xview)
    - [用户自定义数据集](#用户自定义数据集)

## 数据集准备

### VisDrone-DET

VisDrone-DET是一个无人机航拍场景的小目标数据集，整理后的COCO格式VisDrone-DET数据集[下载链接](https://bj.bcebos.com/v1/paddledet/data/smalldet/visdrone.zip)，切图后的COCO格式数据集[下载链接](https://bj.bcebos.com/v1/paddledet/data/smalldet/visdrone_sliced.zip)，检测其中的**10类**，包括 `pedestrian(1), people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10)`，原始数据集[下载链接](https://github.com/VisDrone/VisDrone-Dataset)。
具体使用和下载请参考[visdrone](../visdrone)。

### DOTA水平框

DOTA是一个大型的遥感影像公开数据集，这里使用**DOTA-v1.0**水平框数据集，切图后整理的COCO格式的DOTA水平框数据集[下载链接](https://bj.bcebos.com/v1/paddledet/data/smalldet/dota_sliced.zip)，检测其中的**15类**，
包括 `plane(0), baseball-diamond(1), bridge(2), ground-track-field(3), small-vehicle(4), large-vehicle(5), ship(6), tennis-court(7),basketball-court(8), storage-tank(9), soccer-ball-field(10), roundabout(11), harbor(12), swimming-pool(13), helicopter(14)`，
图片及原始数据集[下载链接](https://captain-whu.github.io/DOAI2019/dataset.html)。

### Xview

Xview是一个大型的航拍遥感检测数据集，目标极小极多，切图后整理的COCO格式数据集[下载链接](https://bj.bcebos.com/v1/paddledet/data/smalldet/xview_sliced.zip)，检测其中的**60类**，
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

，原始数据集[下载链接](https://challenge.xviewdataset.org/)。


### 用户自定义数据集

用户自定义数据集准备请参考[DET数据集标注工具](../../docs/tutorials/data/DetAnnoTools.md)和[DET数据集准备教程](../../docs/tutorials/data/PrepareDetDataSet.md)去准备。
