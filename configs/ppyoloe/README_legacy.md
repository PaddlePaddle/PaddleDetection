# PP-YOLOE Legacy Model Zoo (2022.03)

## Legacy Model Zoo
|          Model           | Epoch | GPU number | images/GPU |  backbone  | input shape | Box AP<sup>val<br>0.5:0.95 | Box AP<sup>test<br>0.5:0.95 | Params(M) | FLOPs(G) | V100 FP32(FPS) | V100 TensorRT FP16(FPS) | download | config  |
|:------------------------:|:-------:|:-------:|:--------:|:----------:| :-------:| :------------------: | :-------------------: |:---------:|:--------:|:---------------:| :---------------------: | :------: | :------: |
| PP-YOLOE-s                  | 400 |     8      |    32    | cspresnet-s |     640     |       43.4        |        43.6         |   7.93    |  17.36   |   208.3   |  333.3   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_400e_coco.pdparams) | [config](./ppyoloe_crn_s_400e_coco.yml)                   |
| PP-YOLOE-s                  | 300 |     8      |    32    | cspresnet-s |     640     |       43.0        |        43.2         |   7.93    |  17.36   |   208.3   |  333.3   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams) | [config](./ppyoloe_crn_s_300e_coco.yml)                   |
| PP-YOLOE-m                  | 300 |     8      |    28    | cspresnet-m |     640     |       49.0        |        49.1         |   23.43   |  49.91   |   123.4   |  208.3   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_300e_coco.pdparams) | [config](./ppyoloe_crn_m_300e_coco.yml)                   |
| PP-YOLOE-l                  | 300 |     8      |    20    | cspresnet-l |     640     |       51.4        |        51.6         |   52.20   |  110.07  |   78.1    |  149.2   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams) | [config](./ppyoloe_crn_l_300e_coco.yml)                   |
| PP-YOLOE-x                  | 300 |     8      |    16    | cspresnet-x |     640     |       52.3        |        52.4         |   98.42   |  206.59  |   45.0    |   95.2   | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_x_300e_coco.pdparams) | [config](./ppyoloe_crn_x_300e_coco.yml)                   |

### Comprehensive Metrics
|          Model           | Epoch | AP<sup>0.5:0.95 | AP<sup>0.5 |  AP<sup>0.75  | AP<sup>small  | AP<sup>medium | AP<sup>large | AR<sup>small | AR<sup>medium | AR<sup>large | download | config  |
|:----------------------:|:-----:|:---------------:|:----------:|:-------------:| :------------:| :-----------: | :----------: |:------------:|:-------------:|:------------:| :-----: | :-----: |
| PP-YOLOE-s             | 400 |      43.4      |     60.0    |     47.5      |     25.7      |      47.8     |     59.2     |     43.9     |      70.8     |   81.9         | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_400e_coco.pdparams) | [config](./ppyoloe_crn_s_400e_coco.yml)|
| PP-YOLOE-s             | 300 |      43.0      |     59.6    |     47.2      |     26.0      |      47.4     |     58.7     |     45.1     |      70.6     |   81.4         | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams) | [config](./ppyoloe_crn_s_300e_coco.yml)|
| PP-YOLOE-m             | 300 |      49.0      |     65.9    |     53.8      |     30.9      |      53.5     |     65.3     |     50.9     |      74.4     |   84.7         | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_300e_coco.pdparams) | [config](./ppyoloe_crn_m_300e_coco.yml)|
| PP-YOLOE-l             | 300 |      51.4      |     68.6    |     56.2      |     34.8      |      56.1     |     68.0     |     53.1     |      76.8     |   85.6         | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams) | [config](./ppyoloe_crn_l_300e_coco.yml)|
| PP-YOLOE-x             | 300 |      52.3      |     69.5    |     56.8      |     35.1      |      57.0     |     68.6     |     55.5     |      76.9     |   85.7         | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_crn_x_300e_coco.pdparams) | [config](./ppyoloe_crn_x_300e_coco.yml)|


**Notes:**

- PP-YOLOE is trained on COCO train2017 dataset and evaluated on val2017 & test-dev2017 dataset.
- The model weights in the table of Comprehensive Metrics are **the same as** that in the original Model Zoo, and evaluated on **val2017**.
- PP-YOLOE used 8 GPUs for training, if **GPU number** or **mini-batch size** is changed, **learning rate** should be adjusted according to the formula **lr<sub>new</sub> = lr<sub>default</sub> * (batch_size<sub>new</sub> * GPU_number<sub>new</sub>) / (batch_size<sub>default</sub> * GPU_number<sub>default</sub>)**.
- PP-YOLOE inference speed is tesed on single Tesla V100 with batch size as 1, **CUDA 10.2**, **CUDNN 7.6.5**, **TensorRT 6.0.1.8** in TensorRT mode.

## Appendix

Ablation experiments of PP-YOLOE.

| NO.  |        Model                 | Box AP<sup>val</sup> | Params(M) | FLOPs(G) | V100 FP32 FPS |
| :--: | :---------------------------: | :------------------: | :-------: | :------: | :-----------: |
|  A   | PP-YOLOv2          |         49.1         |   54.58   |  115.77   |     68.9     |
|  B   | A + Anchor-free    |         48.8         |   54.27   |  114.78   |     69.8     |
|  C   | B + CSPRepResNet   |         49.5         |   47.42   |  101.87   |     85.5     |
|  D   | C + TAL            |         50.4         |   48.32   |  104.75   |     84.0     |
|  E   | D + ET-Head        |         50.9         |   52.20   |  110.07   |     78.1     |
