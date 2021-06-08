## Large-scale practical object detection models (676 categories)

### Introduction

* Unlike the image classification task, in the object detection task, it is necessary to mark not only the category of the object in the image, but also the position of the object, which takes higher cost for labeling. Open Images V5, Objects365 and COCO datasets are commonly used datasets for objecet detection tasks. The basic information of these three datasets is as follows.

|   Dataset          | Classes | Images    | Bounding boxes |
|--------------------|---------|-----------|----------------|
| COCO               | 80      | 123,287   | 886,284        |
| Objects365 | 365     | 600,000   | 10,000,000     |
| Open Images V5              | 500     | 1,743,042 | 14,610,229     |


There are relatively not enough categories in the above dataset (compared to 1000 categories in the ImageNet1k classification dataset). In order to provide more practical server-side object detection models, which are convenient for users to use directly without finetuning anymore, PaddleDetection combines [Practical Server-side detection method base on RCNN](./SERVER_SIDE_en.md), merges Open image V5 and Objects365 dataset to generate a new training set containing 676 categories. The label list can be here: [label list containing 676 categories](../../dataset/voc/generic_det_label_list.txt). Some practical server-side models are trained on the dataset, which are suitable for most application scenarios. It is convenient for users to directly infer or deploy. Users can also finetune on their own datasets based on the provided pretrained models to accelerate convergence and achieve higher performance.


### Model zoo

| Backbone       | Type     |      Download       | Configs |
| :---------------| :---------------| :---------------| :---------------
| ResNet50-vd-FPN-Dcnv2         | Cascade Faster     |  [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r50_vd_fpn_gen_server_side.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/rcnn_enhance/generic/cascade_rcnn_dcn_r50_vd_fpn_gen_server_side.yml) |
| ResNet101-vd-FPN-Dcnv2         | Cascade Faster     |  [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r101_vd_fpn_gen_server_side.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/rcnn_enhance/generic/cascade_rcnn_dcn_r101_vd_fpn_gen_server_side.yml) |
| CBResNet101-vd-FPN         | Cascade Faster     |  [model](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_cbr101_vd_fpn_server_side.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/static/configs/rcnn_enhance/generic/cascade_rcnn_cbr101_vd_fpn_server_side.yml) |
