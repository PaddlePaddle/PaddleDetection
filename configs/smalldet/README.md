简体中文 | [English](README.md)

# PP-YOLOE Smalldet 检测模型


|    模型   |       数据集     |  SLICE_SIZE  |  OVERLAP_RATIO  | 类别数  | mAP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | 下载链接  | 配置文件 |
|:---------|:---------------:|:---------------:|:---------------:|:------:|:-----------------------:|:-------------------:|:---------:| :-----: |
|PP-YOLOE-l|   Xview  |  400 | 0.25 | 60 |  14.5 | 26.8 | [下载链接](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_crn_l_xview_400_025.pdparams) | [配置文件](./ppyoloe_crn_l_80e_sliced_xview_400_025.yml) |
|PP-YOLOE-l|   DOTA   |  500 | 0.25 | 15 |  46.8 |  72.6 | [下载链接](https://bj.bcebos.com/v1/paddledet/models/ppyoloe_crn_l_dota_500_025.pdparams) | [配置文件](./ppyoloe_crn_l_80e_sliced_DOTA_500_025.yml) |


**注意:**
- **SLICE_SIZE**表示使用SAHI工具切图后子图的大小(SLICE_SIZE*SLICE_SIZE)；**OVERLAP_RATIO**表示切图重叠率。
- PP-YOLOE模型训练过程中使用8 GPUs进行混合精度训练，如果**GPU卡数**或者**batch size**发生了改变，你需要按照公式 **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)** 调整学习率。
- 具体使用教程请参考[ppyoloe](../ppyoloe#getting-start)。
