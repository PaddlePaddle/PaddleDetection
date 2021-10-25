简体中文 | [English](README.md)

# MCFairMOT (Multi-class FairMOT)

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [快速开始](#快速开始)
- [引用](#引用)

## 内容

MCFairMOT是[FairMOT](https://arxiv.org/abs/2004.01888)的多类别扩展版本。

## 模型库

### MCFairMOT DLA-34 在VisDrone2019 MOT val-set上结果
|    骨干网络      |  输入尺寸 |  MOTA  |  IDF1  |  IDS   |   FPS   |  下载链接 | 配置文件 |
| :--------------| :------- | :----: | :----: | :---:  | :------: | :----: |:----: |
| DLA-34         | 1088x608 |  24.3  |  41.6  |  2314  |    -     |[下载链接](https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone.pdparams) | [配置文件](./mcfairmot_dla34_30e_1088x608_visdrone.yml) |


## 快速开始

### 1. 训练
使用8个GPU通过如下命令一键式启动训练
```bash
python -m paddle.distributed.launch --log_dir=./mcfairmot_dla34_30e_1088x608_visdrone/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml
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
 默认评估的是VisDrone2019 MOT val-set数据集, 如需换评估数据集可参照以下代码修改`configs/datasets/mcmot.yml`：

### 3. 预测
使用单个GPU通过如下命令预测一个视频，并保存为视频
```bash
# 预测一个视频
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone.pdparams --video_file={your video name}.mp4  --save_videos
```
**注意:**
 请先确保已经安装了[ffmpeg](https://ffmpeg.org/ffmpeg.html), Linux(Ubuntu)平台可以直接用以下命令安装：`apt-get update && apt-get install -y ffmpeg`。

### 4. 导出预测模型
```bash
CUDA_VISIBLE_DEVICES=0 python tools/export_model.py -c configs/mot/mcfairmot/mcfairmot_dla34_30e_1088x608_visdrone.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/mcfairmot_dla34_30e_1088x608_visdrone.pdparams
```

### 5. 用导出的模型基于Python去预测
```bash
python deploy/python/mot_jde_infer.py --model_dir=output_inference/mcfairmot_dla34_30e_1088x608_visdrone --video_file={your video name}.mp4 --device=GPU --save_mot_txts
```
**注意:**
 跟踪模型是对视频进行预测，不支持单张图的预测，默认保存跟踪结果可视化后的视频，可添加`--save_mot_txts`表示保存跟踪结果的txt文件，或`--save_images`表示保存跟踪结果可视化图片。


## 引用
```
@article{zhang2020fair,
  title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}

@article{zhu2018vision,
  title={Vision meets drones: A challenge},
  author={Zhu, Pengfei and Wen, Longyin and Bian, Xiao and Ling, Haibin and Hu, Qinghua},
  journal={arXiv preprint arXiv:1804.07437},
  year={2018}
}

@article{zhu2020vision,
  title={Vision Meets Drones: Past, Present and Future},
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Hu, Qinghua and Ling, Haibin},
  journal={arXiv preprint arXiv:2001.06303},
  year={2020}
}

@InProceedings{bdd100k,
    author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen,
              Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```
