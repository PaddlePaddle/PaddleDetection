# DETRs Beat YOLOs on Real-time Object Detection

## Introduction
We propose a **R**eal-**T**ime **DE**tection **TR**ansformer (RT-DETR), the first real-time end-to-end object detector to our best knowledge. Specifically, we design an efficient hybrid encoder to efficiently process multi-scale features by decoupling the intra-scale interaction and cross-scale fusion, and propose IoU-aware query selection to improve the initialization of object queries. In addition, our proposed detector supports flexibly adjustment of the inference speed by using different decoder layers without the need for retraining, which facilitates the practical application of real-time object detectors. Our RT-DETR-L achieves 53.0% AP on COCO val2017 and 114 FPS on T4 GPU, while RT-DETR-X achieves 54.8% AP and 74 FPS, outperforming all YOLO detectors of the same scale in both speed and accuracy. Furthermore, our RT-DETR-R50 achieves 53.1% AP and 108 FPS, outperforming DINO-Deformable-DETR-R50 by 2.2% AP in accuracy and by about 21 times in FPS. For more details, please refer to our [paper](https://arxiv.org/abs/2304.08069).

<div align="center">
  <img src="https://user-images.githubusercontent.com/17582080/232390925-54e58fe6-1c17-4610-90b9-7e5525577d80.png" width=500 />
</div>


## Model Zoo

### Model Zoo on COCO

| Model | Epoch | backbone  | input shape | $AP^{val}$ | $AP^{val}_{50}$| Params(M) | FLOPs(G) |  T4 TensorRT FP16(FPS) | Pretrained Model | config |
|:--------------:|:-----:|:----------:| :-------:|:--------------------------:|:---------------------------:|:---------:|:--------:| :---------------------: |:------------------------------------------------------------------------------------:|:-------------------------------------------:|
| RT-DETR-R50 | 80 |  ResNet-50 | 640 | 53.1 | 71.3 | 42 | 136 | 108 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams) | [config](./rtdetr_r50vd_6x_coco.yml)
| RT-DETR-R101 | 80 |  ResNet-101 | 640 | 54.3 | 72.7 | 76 | 259 | 74 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r101vd_6x_coco.pdparams) | [config](./rtdetr_r101vd_6x_coco.yml)
| RT-DETR-L | 80 |  HGNetv2 | 640 | 53.0 | 71.6 | 32 | 110 | 114 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_hgnetv2_l_6x_coco.pdparams) | [comming soon](rtdetr_hgnetv2_l_6x_coco.yml)
| RT-DETR-X | 80 |  HGNetv2 | 640 | 54.8 | 73.1 | 67 | 234 | 74 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_hgnetv2_x_6x_coco.pdparams) | [comming soon](rtdetr_hgnetv2_x_6x_coco.yml)

**Notes:**
- RT-DETR uses 4GPU to train.
- RT-DETR is trained on COCO train2017 dataset and evaluated on val2017 results of `mAP(IoU=0.5:0.95)`.

GPU multi-card training
```bash
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --fleet --eval
```

## Citations
```
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
