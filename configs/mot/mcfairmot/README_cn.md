简体中文 | [English](README.md)

# MCFairMOT (Multi-class FairMOT)

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [快速开始](#快速开始)
- [引用](#引用)

## 内容

MCFairMOT是[FairMOT](https://arxiv.org/abs/2004.01888)的多类别扩展版本。

### PP-Tracking 实时多目标跟踪系统
此外，PaddleDetection还提供了[PP-Tracking](../../../deploy/pptracking/README.md)实时多目标跟踪系统。PP-Tracking是基于PaddlePaddle深度学习框架的业界首个开源的实时多目标跟踪系统，具有模型丰富、应用广泛和部署高效三大优势。
PP-Tracking支持单镜头跟踪(MOT)和跨镜头跟踪(MTMCT)两种模式，针对实际业务的难点和痛点，提供了行人跟踪、车辆跟踪、多类别跟踪、小目标跟踪、流量统计以及跨镜头跟踪等各种多目标跟踪功能和应用，部署方式支持API调用和GUI可视化界面，部署语言支持Python和C++，部署平台环境支持Linux、NVIDIA Jetson等。

### AI Studio公开项目案例
PP-Tracking 提供了AI Studio公开项目案例，教程请参考[PP-Tracking之手把手玩转多目标跟踪](https://aistudio.baidu.com/aistudio/projectdetail/3022582)。

## 模型库

### MCFairMOT 在VisDrone2019 MOT val-set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |  IDS   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :---:  | :------: | :----: |:----: |
| DLA-34         | 1088x608 |  24.3  |  41.6  |  2314  |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone.pdparams) | [配置文件](./mcfairmot_dla34_30e_1088x608_visdrone.yml) |
| HRNetV2-W18    | 1088x608 |  20.4  |  39.9  |  2603  |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone.pdparams) | [配置文件](./mcfairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone.yml) |
| HRNetV2-W18    | 864x480 |  18.2  |  38.7  |  2416  |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_864x480_visdrone.pdparams) | [配置文件](./mcfairmot_hrnetv2_w18_dlafpn_30e_864x480_visdrone.yml) |
| HRNetV2-W18    | 576x320 |  12.0  |  33.8  |  2178  |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_576x320_visdrone.pdparams) | [配置文件](./mcfairmot_hrnetv2_w18_dlafpn_30e_576x320_visdrone.yml) |

**注意:**
 - MOTA是VisDrone2019 MOT数据集10类目标的平均MOTA, 其值也等于所有评估的视频序列的平均MOTA，此处提供数据集[下载链接](https://bj.bcebos.com/v1/paddledet/data/mot/visdrone_mcmot.zip)。
 - MCFairMOT模型均使用4个GPU进行训练，训练30个epoch。DLA-34骨干网络的每个GPU上batch size为6，HRNetV2-W18骨干网络的每个GPU上batch size为8。

### MCFairMOT 在VisDrone Vehicle val-set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |  IDS   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :---:  | :------: | :----: |:----: |
| DLA-34         | 1088x608 |  37.7  |  56.8  |  199  |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone_vehicle_bytetracker.pdparams) | [配置文件](./mcfairmot_dla34_30e_1088x608_visdrone_vehicle_bytetracker.yml) |
| HRNetV2-W18    | 1088x608 |  35.6  |  56.3  |  190  |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone_vehicle_bytetracker.pdparams) | [配置文件](./mcfairmot_hrnetv2_w18_dlafpn_30e_1088x608_visdrone_vehicle_bytetracker.yml) |

**注意:**
 - MOTA是VisDrone Vehicle数据集4类车辆目标的平均MOTA, 该数据集是VisDrone数据集中抽出4类车辆类别组成的，此处提供数据集[下载链接](https://bj.bcebos.com/v1/paddledet/data/mot/visdrone_mcmot_vehicle.zip)。
 - MCFairMOT模型此处使用的跟踪器是使用的ByteTracker。

### MCFairMOT 在VisDrone Vehicle val-set上离线量化结果
|    骨干网络      |  压缩策略 | 预测时延（T4） |预测时延（V100）| 配置文件 |压缩算法配置文件 |
| :--------------| :------- | :------: | :----: | :----: | :----: |
| DLA-34         | baseline |    41.3  |    21.9 |[配置文件](./mcfairmot_dla34_30e_1088x608_visdrone_vehicle_bytetracker.yml)|    -     |
| DLA-34         | 离线量化   |  37.8    |  21.2  |[配置文件](./mcfairmot_dla34_30e_1088x608_visdrone_vehicle_bytetracker.yml)|[配置文件](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/configs/slim/post_quant/mcfairmot_ptq.yml)|

## 快速开始

### 1. 训练
使用4个GPU通过如下命令一键式启动训练
```bash
python -m paddle.distributed.launch --log_dir=./mcfairmot_dla34_30e_1088x608_visdrone/ --gpus 0,1,2,3 tools/train.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml
```

### 2. 评估
使用单张GPU通过如下命令一键式启动评估
```bash
# 使用PaddleDetection发布的权重
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone.pdparams

# 使用训练保存的checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/eval_mot.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml -o weights=output/mcfairmot_dla34_30e_1088x608_visdrone/model_final.pdparams
```
**注意:**
 - 默认评估的是VisDrone2019 MOT val-set数据集, 如需换评估数据集可参照以下代码修改`configs/datasets/mcmot.yml`：
  ```
  EvalMOTDataset:
    !MOTImageFolder
      dataset_dir: dataset/mot
      data_root: your_dataset/images/val
      keep_ori_im: False # set True if save visualization images or video
  ```
 - 多类别跟踪结果会存于`{output_dir}/mot_results/`中，里面每个视频序列对应一个txt，每个txt文件每行信息是`frame,id,x1,y1,w,h,score,cls_id,-1,-1`, 此外`{output_dir}`可通过`--output_dir`设置。

### 3. 预测
使用单个GPU通过如下命令预测一个视频，并保存为视频
```bash
# 预测一个视频
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone.pdparams --video_file={your video name}.mp4  --save_videos
```
**注意:**
 - 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。

### 4. 导出预测模型
```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone.pdparams
```

### 5. 用导出的模型基于Python去预测
```bash
python deploy/pptracking/python/mot_jde_infer.py --model_dir=output_inference/mcfairmot_dla34_30e_1088x608_visdrone --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**注意:**
 - 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。
 - 多类别跟踪结果txt文件每行信息是`frame,id,x1,y1,w,h,score,cls_id,-1,-1`。

### 6. 离线量化

使用 VisDrone Vehicle val-set 对离线量化模型进行校准，运行方式：
```bash
CUDA_VISIBLE_DEVICES=0 python3.7 tools/post_quant.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone_vehicle_bytetracker.yml --slim_config=configs/slim/post_quant/mcfairmot_ptq.yml
```
**注意:**
 - 离线量化默认使用的是VisDrone Vehicle val-set数据集以及4类车辆跟踪模型。

## 引用
```
@article{zhang2020fair,
  title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}

@ARTICLE{9573394,
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Detection and Tracking Meet Drones Challenge},
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3119563}
}

@article{zhang2021bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}
```
