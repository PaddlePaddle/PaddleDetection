# Multi Scale Test Configuration

Tags: Configuration

---
```yaml

##################################### Multi scale test configuration #####################################

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

Multi Scale Test is a TTA (Test Time Augmentation) method, it can improve object detection performance. 

The input image will be scaled into different scales, then model generated predictions (bboxes) at different scales, finally all the predictions will be combined to generate final prediction. (Here **NMS** is used to aggregate the predictions.)

## _MultiscaleTestResize_ option

`MultiscaleTestResize` option is used to enable multi scale test prediction. 

`origin_target_size: [800, 1333]` means the input image will be scaled to 800 (for short edge) and 1333 (max edge length cannot be greater than 1333) at first

`target_size: [700 , 900]` property is used to specify different scales. 

It can be plugged into evaluation process or test (inference) process, by adding `MultiscaleTestResize` entry to `EvalReader.sample_transforms` or `TestReader.sample_transforms`

---

###Note

Now only CascadeRCNN, FasterRCNN and MaskRCNN are supported for multi scale testing. And batch size must be 1.