# JDE (Towards-Realtime-MOT)

## Model Zoo

### JDE on MOT-16 training set

| 骨架网络           | 输入尺寸  | MOTA   | IDF1   |  IDS  |   FP  |   FN  |   FPS  | 下载  | 配置文件 |
| :-----------------| :------- | :----: | :----: | :---: | :----: | :---: | :---: |:---: | :---: |
| DarkNet53(paper)  | 1088x608 |  74.8  |  67.3  | 1189  |  5558  | 21505 |  22.2 | ---- | ---- |
| DarkNet53(paper)  | 864x480  |  70.8	|  65.8  | 1279  |  5653  | 25806 |  30.3 | ---- | ---- |
| DarkNet53(paper)  | 576x320  |  63.7  |  63.3  | 1307  |  6657  | 32794 |  37.9 | ---- | ---- |
| DarkNet53         | 1088x608 |    -   |    -   |   -   |    -   |   -   |   -   |[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/jde_darknet53_30e_1088x608.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/jde/jde_darknet53_30e_1088x608.yml) |
| DarkNet53         | 864x480  |    -   |    -   |   -   |    -   |   -   |   -   |[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/jde_darknet53_30e_864x480.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/jde/jde_darknet53_30e_864x480.yml) |
| DarkNet53         | 576x320  |    -   |    -   |   -   |    -   |   -   |   -   |[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/dygraph/jde_darknet53_30e_576x320.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/jde/jde_darknet53_30e_576x320.yml) |


**注意：** JDE均使用8GPU训练，在总batchsize为16的情况下训练30个epoch。


## Citations
```
@article{wang2019towards,
  title={Towards Real-Time Multi-Object Tracking},
  author={Wang, Zhongdao and Zheng, Liang and Liu, Yixuan and Wang, Shengjin},
  journal={arXiv preprint arXiv:1909.12605},
  year={2019}
}
```
