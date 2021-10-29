# PaddleDetection Model Export Tutorial

## 一、Model Export
This section describes how to use the `tools/export_model.py` script to export models.
### Export model input and output description
- Input variables and input shapes are as follows:

  |  Input Name  | Input Shape     | Meaning                                                                                                                   |
  | :----------: | --------------- | ------------------------------------------------------------------------------------------------------------------------- |
  |    image     | [None, 3, H, W] | Enter the network image. None indicates the Batch dimension. If the input image size is variable length, H and W are None |
  |   im_shape   | [None, 2]       | The size of the image after resize is expressed as H,W, and None represents the Batch dimension                           |
  | scale_factor | [None, 2]       | The input image size is larger than the real image size, denoted byscale_y, scale_x                                       |

**Attention**For details about the preprocessing method, see the Test Reader section in the configuration file.


-The output of the dynamic and static derived model in Paddle Detection is unified as follows:

  - bbox, the output of NMS, in the shape of [N, 6], where N is the number of prediction boxes, and 6 is [class_id, score, x1, y1, x2, y2].
  - bbox\_num, Each picture corresponds to the number of prediction boxes. For example, batch size is 2 and the output is [N1, N2], indicating that the first picture contains N1 prediction boxes and the second picture contains N2 prediction boxes, and the total number of prediction boxes is the same as the first dimension N output by NMS
  - mask, If the network contains a mask, the mask branch is printed

**Attention**The model-to-static export does not support cases where numpy operations are included in the model structure.


### 2、Start Parameters

|     FLAG     |               USE               |       DEFAULT        |                                 NOTE                                  |
| :----------: | :-----------------------------: | :------------------: | :-------------------------------------------------------------------: |
|      -c      | Specifying a configuration file |         None         |                                                                       |
| --output_dir |         Model save path         | `./output_inference` | The model is saved in the `output/default_file_name/` path by default |

### 3、Example

Using the trained model for trial use, the script is as follows:

```bash
# The YOLOv3 model is exported
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --output_dir=./inference_model \
 -o weights=weights/yolov3_darknet53_270e_coco.pdparams
```
The prediction model will be exported to the `inference_model/yolov3_darknet53_270e_coco` directory. `infer_cfg.yml`, `model.pdiparams`,  `model.pdiparams.info`, `model.pdmodel` respectively.


### 4、Sets the input size of the export model
When using Fluid TensorRT for prediction, since <= TensorRT 5.1 only supports fixed-length input, the image size of the `data` layer of the saved model needs to be the same as the actual input image size. Fluid C++ prediction engine does not have this limitation. Setting `image_shape` in Test Reader changes the size of the input image in the saved model. The following is an example:


```bash
#Export the YOLOv3 model with the input 3x640x640
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --output_dir=./inference_model \
 -o weights=weights/yolov3_darknet53_270e_coco.pdparams TestReader.inputs_def.image_shape=[3,640,640]
```
