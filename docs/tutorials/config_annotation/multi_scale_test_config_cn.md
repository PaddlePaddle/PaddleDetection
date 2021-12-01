# 多尺度测试的配置

标签: 配置

---
```yaml

##################################### 多尺度测试的配置 #####################################

EvalReader:
  sample_transforms:
  - Decode: {}
  - MultiscaleTestResize: {origin_target_size: [800, 1333], target_size: [700 , 900]}
  - NormalizeImage: {is_scale: true, mean: [0.485,0.456,0.406], std: [0.229, 0.224,0.225]}
  - Permute: {}

TestReader:
  sample_transforms:
  - Decode: {}
  - MultiscaleTestResize: {origin_target_size: [800, 1333], target_size: [700 , 900]}
  - NormalizeImage: {is_scale: true, mean: [0.485,0.456,0.406], std: [0.229, 0.224,0.225]}
  - Permute: {}
```

---

多尺度测试是一种TTA方法（测试时增强），可以用于提高目标检测的准确率

输入图像首先被缩放为不同尺度的图像，然后模型对这些不同尺度的图像进行预测，最后将这些不同尺度上的预测结果整合为最终预测结果。（这里使用了**NMS**来整合不同尺度的预测结果）

## _MultiscaleTestResize_ 选项

`MultiscaleTestResize` 选项用于开启多尺度测试. 

`origin_target_size: [800, 1333]` 项代表输入图像首先缩放为短边为800，最长边不超过1333.

`target_size: [700 , 900]` 项设置不同的预测尺度。

通过在`EvalReader.sample_transforms`或`TestReader.sample_transforms`中设置`MultiscaleTestResize`项，可以在评估过程或预测过程中开启多尺度测试。

---

###注意

目前多尺度测试只支持CascadeRCNN, FasterRCNN and MaskRCNN网络, 并且batch size需要是1.