# Object detection grad_cam heatmap

## 1.Introduction
Calculate the cam (class activation map) of the object predict bbox based on the backbone feature map

## 2.Usage
* Taking PP-YOLOE as an example, after preparing the data, specify the network configuration file, model weight address, image path and output folder path, and then use the script to call tools/cam_ppdet.py to calculate the grad_cam heat map of the prediction box. Below is an example run script.
```shell
python tools/cam_ppdet.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml --infer_img demo/000000014439.jpg --cam_out cam_ppyoloe -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
```

* **Arguments**

|         FLAG             |                                                            description                                                            |
| :----------------------: |:---------------------------------------------------------------------------------------------------------------------------------:|
|          -c              |                                                        Select config file                                                         |
|          --infer_img              |                                                            Image path                                                             |
|          --cam_out              |                                                       Directory for output                                                        |
|          -o              | Set parameters in configure file, for example: -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams |

* result

<center>
<img src="../images/grad_cam_ppyoloe_demo.jpg" width="500" >
</center>
<br><center>cam_ppyoloe/225.jpg</center></br>


## 3.Currently supports networks based on FasterRCNN and YOLOv3 series.
* FasterRCNN bbox heat map visualization script
```bash
python tools/cam_ppdet.py -c configs/faster_rcnn/faster_rcnn_r50_vd_fpn_2x_coco.yml --infer_img demo/000000014439.jpg  --cam_out cam_faster_rcnn -o weights=https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_fpn_ssld_2x_coco.pdparams
```
* PPYOLOE bbox heat map visualization script
```bash
python tools/cam_ppdet.py -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml --infer_img demo/000000014439.jpg --cam_out cam_ppyoloe -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams
```