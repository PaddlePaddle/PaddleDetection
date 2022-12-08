# Co-tuning for Transfer Learning

## Model Zoo
| 骨架网络             | 网络类型       | 每张GPU图片个数 |推理时间(fps) | Box AP |  配置文件  |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: |
| ResNet50-vd             | Faster         |    1    |     ----     |  60.1  | [配置文件](./faster_rcnn_r50_vd_fpn_1x_coco_cotuning_roadsign.yml) |
## Citations
```
@article{you2020co,
  title={Co-tuning for transfer learning},
  author={You, Kaichao and Kou, Zhi and Long, Mingsheng and Wang, Jianmin},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={17236--17246},
  year={2020}
}
```