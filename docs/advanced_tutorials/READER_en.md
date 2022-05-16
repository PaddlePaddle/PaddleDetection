# Data Processing Module

## Directory
- [Data Processing Module](#data-processing-module)
  - [Directory](#directory)
    - [1.Introduction](#1introduction)
    - [2.Dataset](#2dataset)
      - [2.1COCO Dataset](#21coco-dataset)
      - [2.2Pascal VOC dataset](#22pascal-voc-dataset)
      - [2.3Customize Dataset](#23customize-dataset)
    - [3.Data preprocessing](#3data-preprocessing)
      - [3.1Data Enhancement Operator](#31data-enhancement-operator)
      - [3.2Custom data enhancement operator](#32custom-data-enhancement-operator)
    - [4.Reader](#4reader)
    - [5.Configuration and Operation](#5configuration-and-operation)
      - [5.1Configuration](#51configuration)
      - [5.2run](#52run)

### 1.Introduction
All code logic for Paddle Detection's data processing module in `ppdet/data/`, the data processing module is used to load data and convert it into a format required for training, evaluation and reasoning of object Detection models. The main components of the data processing module are as follows:
The main components of the data processing module are as follows:
```bash
  ppdet/data/
  ├── reader.py     # Reader module based on Dataloader encapsulation
  ├── source  # Data source management module
  │   ├── dataset.py      # Defines the data source base class from which various datasets are inherited
  │   ├── coco.py         # The COCO dataset parses and formats the data
  │   ├── voc.py          # Pascal VOC datasets parse and format data
  │   ├── widerface.py    # The WIDER-FACE dataset parses and formats data
  │   ├── category.py    # Category information for the relevant dataset
  ├── transform  # Data preprocessing module
  │   ├── batch_operators.py  # Define all kinds of preprocessing operators based on batch data
  │   ├── op_helper.py    # The auxiliary function of the preprocessing operator
  │   ├── operators.py    # Define all kinds of preprocessing operators based on single image
  │   ├── gridmask_utils.py    # GridMask data enhancement function
  │   ├── autoaugment_utils.py  # AutoAugment auxiliary function
  ├── shm_utils.py     # Auxiliary functions for using shared memory
  ```


### 2.Dataset
The dataset is defined in the `source` directory, where `dataset.py` defines the base class `DetDataSet` of the dataset. All datasets inherit from the base class, and the `DetDataset` base class defines the following methods:

|          Method           |                    Input                     |                  Output                   |                                                      Note                                                       |
| :-----------------------: | :------------------------------------------: | :---------------------------------------: | :-------------------------------------------------------------------------------------------------------------: |
|        \_\_len\_\_        |                      no                      | int, the number of samples in the dataset |                                        Filter out the unlabeled samples                                         |
|      \_\_getitem\_\_      |         int, The index of the sample         |      dict, Index idx to sample ROIDB      |                                      Get the sample roidb after transform                                       |
| check_or_download_dataset |                      no                      |                    no                     | Check whether the dataset exists, if not, download, currently support COCO, VOC, Widerface and other datasets |
|        set_kwargs         | Optional arguments, given as key-value pairs |                    no                     |                     Currently used to support receiving mixup, cutMix and other parameters                      |
|       set_transform       |       A series of transform functions        |                    no                     |                                    Set the transform function of the dataset                                    |
|         set_epoch         |              int, current epoch              |                    no                     |                                Interaction between dataset and training process                                 |
|       parse_dataset       |                      no                      |                    no                     |                                     Used to read all samples from the data                                      |
|         get_anno          |                      no                      |                    no                     |                                   Used to get the path to the annotation file                                   |

When a dataset class inherits from `DetDataSet`, it simply implements the Parse dataset function. parse_dataset set dataset root path dataset_dir, image folder image dir, annotated file path anno_path retrieve all samples and save them in a list roidbs Each element in the list is a sample XXX rec(such as coco_rec or voc_rec), represented by dict, which contains the sample image, gt_bbox, gt_class and other fields. The data structure of xxx_rec in COCO and Pascal-VOC datasets is defined as follows:
  ```python
  xxx_rec = {
      'im_file': im_fname,         # The full path to an image
      'im_id': np.array([img_id]), # The ID number of an image
      'h': im_h,                   # Height of the image
      'w': im_w,                   # The width of the image
      'is_crowd': is_crowd,        # Community object, default is 0 (VOC does not have this field)
      'gt_class': gt_class,        # ID number of an enclosure label name
      'gt_bbox': gt_bbox,          # label box coordinates(xmin, ymin, xmax, ymax)
      'gt_poly': gt_poly,          # Segmentation mask. This field only appears in coco_rec and defaults to None
      'difficult': difficult       # Is it a difficult sample? This field only appears in voc_rec and defaults to 0
  }
  ```

The contents of the xxx_rec can also be controlled by the Data fields parameter of `DetDataSet`, that is, some unwanted fields can be filtered out, but in most cases you do not need to change them. The default configuration in `configs/datasets` will do.

In addition, a dictionary `cname2cid` holds the mapping of category names to IDS in the Parse dataset function. In coco dataset, can use [coco API](https://github.com/cocodataset/cocoapi) from the label category name of the file to load dataset, and set up the dictionary. In the VOC dataset, if `use_default_label=False` is set, the category list will be read from `label_list.txt`, otherwise the VOC default category list will be used.

#### 2.1COCO Dataset
COCO datasets are currently divided into COCO2014 and COCO2017, which are mainly composed of JSON files and image files, and their organizational structure is shown as follows:
  ```
  dataset/coco/
  ├── annotations
  │   ├── instances_train2014.json
  │   ├── instances_train2017.json
  │   ├── instances_val2014.json
  │   ├── instances_val2017.json
  │   │   ...
  ├── train2017
  │   ├── 000000000009.jpg
  │   ├── 000000580008.jpg
  │   │   ...
  ├── val2017
  │   ├── 000000000139.jpg
  │   ├── 000000000285.jpg
  │   │   ...
  ```
class `COCODataSet` is defined and registered on `source/coco.py`. And implements the parse the dataset method, called [COCO API](https://github.com/cocodataset/cocoapi) to load and parse COCO format data source ` roidbs ` and ` cname2cid `, See `source/coco.py` source code for details. Converting other datasets to COCO format can be done by referring to [converting User Data to COCO Data](../tutorials/PrepareDataSet_en.md#convert-user-data-to-coco-data)
And implements the parse the dataset method, called [COCO API](https://github.com/cocodataset/cocoapi) to load and parse COCO format data source `roidbs` and `cname2cid`, See `source/coco.py` source code for details. Converting other datasets to COCO format can be done by referring to [converting User Data to COCO Data](../tutorials/PrepareDataSet_en.md#convert-user-data-to-coco-data)


#### 2.2Pascal VOC dataset
The dataset is currently divided into VOC2007 and VOC2012, mainly composed of XML files and image files, and its organizational structure is shown as follows:
```
  dataset/voc/
  ├── trainval.txt
  ├── test.txt
  ├── label_list.txt (optional)
  ├── VOCdevkit/VOC2007
  │   ├── Annotations
  │       ├── 001789.xml
  │       │   ...
  │   ├── JPEGImages
  │       ├── 001789.jpg
  │       │   ...
  │   ├── ImageSets
  │       |   ...
  ├── VOCdevkit/VOC2012
  │   ├── Annotations
  │       ├── 2011_003876.xml
  │       │   ...
  │   ├── JPEGImages
  │       ├── 2011_003876.jpg
  │       │   ...
  │   ├── ImageSets
  │       │   ...
  ```
The `VOCDataSet` dataset is defined and registered in `source/voc.py` . It inherits the `DetDataSet` base class and rewrites the `parse_dataset` method to parse XML annotations in the VOC dataset. Update `roidbs` and `cname2cid`. To convert other datasets to VOC format, refer to [User Data to VOC Data](../tutorials/PrepareDataSet_en.md#convert-user-data-to-voc-data)


#### 2.3Customize Dataset
If the COCO dataset and VOC dataset do not meet your requirements, you can load your dataset by customizing it. There are only two steps to implement a custom dataset

1. create`source/xxx.py`, define class `XXXDataSet` extends from `DetDataSet` base class, complete registration and serialization, and rewrite `parse_dataset`methods to update `roidbs` and `cname2cid`:
  ```python
  from ppdet.core.workspace import register, serializable

  #Register and serialize
  @register
  @serializable
  class XXXDataSet(DetDataSet):
      def __init__(self,
                  dataset_dir=None,
                  image_dir=None,
                  anno_path=None,
                  ...
                  ):
          self.roidbs = None
          self.cname2cid = None
          ...

      def parse_dataset(self):
          ...
          Omit concrete parse data logic
          ...
          self.roidbs, self.cname2cid = records, cname2cid
  ```

2. Add a reference to `source/__init__.py`:
  ```python
  from . import xxx
  from .xxx import *
  ```
Complete the above two steps to add the new Data source `XXXDataSet`, you can refer to [Configure and Run](#5.Configuration-and-Operation) to implement the use of custom datasets.

### 3.Data preprocessing

#### 3.1Data Enhancement Operator
A variety of data enhancement operators are supported in PaddleDetection, including single image data enhancement operator and batch data enhancement operator. You can choose suitable operators to use in combination. Single image data enhancement operators are defined in `transform/operators.py`. The supported single image data enhancement operators are shown in the following table:
|              Name              |                                                                                                                                 Function                                                                                                                                 |
| :----------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|             Decode             |                                                                                                     Loads an image from an image file or memory buffer in RGB format                                                                                                     |
|            Permute             |                                                                                                             If the input is HWC, the sequence changes to CHW                                                                                                             |
|       RandomErasingImage       |                                                                                                                       Random erasure of the image                                                                                                                        |
|         NormalizeImage         |                                                                     The pixel value of the image is normalized. If is scale= True is set, the pixel value is divided by 255.0 before normalization.                                                                      |
|            GridMask            |                                                                                                                        GridMask data is augmented                                                                                                                        |
|         RandomDistort          |                                                                                                   Random disturbance of image brightness, contrast, saturation and hue                                                                                                   |
|          AutoAugment           |                                                                                                 Auto Augment data, which contains a series of data augmentation methods                                                                                                  |
|           RandomFlip           |                                                                                                                   Randomly flip the image horizontally                                                                                                                   |
|             Resize             |                                                                                                        Resize the image and transform the annotation accordingly                                                                                                         |
|      MultiscaleTestResize      |                                                                                                          Rescale the image to each size of the multi-scale list                                                                                                          |
|          RandomResize          |                                                                               Random Resize of images can be resized to different sizes and different interpolation strategies can be used                                                                               |
|          RandomExpand          |                                                                                 Place the original image into an expanded image filled with pixel mean, crop, scale, and flip the image                                                                                  |
|        CropWithSampling        | Several candidate frames are generated according to the scaling ratio and length-width ratio, and then the prunning results that meet the requirements are selected according to the area intersection ratio (IoU) between these candidate frames and the marking frames |
| CropImageWithDataAchorSampling |                                                       Based on Crop Image, in face detection, the Image scale is randomly transformed to a certain range of scale, which greatly enhances the scale change of face                                                       |
|           RandomCrop           |                                                                                   The principle is the same as CropImage, which is processed with random proportion and IoU threshold                                                                                    |
|        RandomScaledCrop        |                                                                        According to the long edge, the image is randomly clipped and the corresponding transformation is made to the annotations                                                                         |
|             Cutmix             |                                                                                                              Cutmix data enhancement, Mosaic of two images                                                                                                               |
|             Mixup              |                                                                                                              Mixup data enhancement to scale up two images                                                                                                               |
|          NormalizeBox          |                                                                                                                        Bounding box is normalized                                                                                                                        |
|             PadBox             |                                                                                        If the number of bounding boxes is less than num Max boxes, zero is populated into bboxes                                                                                         |
|         BboxXYXY2XYWH          |                                                                                       Bounding Box is converted from (xmin,ymin,xmax,ymin) form to (xmin,ymin, Width,height) form                                                                                        |
|              Pad               |                                                                          The image Pad is an integer multiple of a certain number or the specified size, and supports the way of specifying Pad                                                                          |
|           Poly2Mask            |                                                                                                                      Poly2Mask data enhancement ｜                                                                                                                       |

Batch data enhancement operators are defined in `transform/batch_operators.py`. The list of operators currently supported is as follows:
|       Name        |                                                       Function                                                       |
| :---------------: | :------------------------------------------------------------------------------------------------------------------: |
|     PadBatch      | Pad operation is performed on each batch of data images randomly to make the images in the batch have the same shape |
| BatchRandomResize |            Resize a batch of images so that the images in the batch are randomly scaled to the same size             |
|   Gt2YoloTarget   |                              Generate the objectives of YOLO series models from GT data                              |
|   Gt2FCOSTarget   |                                  Generate the target of the FCOS model from GT data                                  |
|   Gt2TTFTarget    |                                     Generate TTF Net model targets from GT data                                      |
|  Gt2Solov2Target  |                                   Generate targets for SOL Ov2 models from GT data                                   |

**A few notes:**
- The input of Data enhancement operator is sample or samples, and each sample corresponds to a sample of RoIDBS output by `DetDataSet` mentioned above, such as coco_rec or voc_rec
- Single image data enhancement operators (except Mixup, Cutmix, etc.) can also be used in batch data processing. However, there are still some differences between single image processing operators and Batch image processing operators. Taking Random Resize and Batch Random Resize as an example, Random Resize will randomly scale each picture in a Batch. However, the shapes of each image after Resize are different. Batch Random Resize means that all images in a Batch will be randomly scaled to the same shape.
- In addition to Batch Random Resize, the Batch data enhancement operators defined in `transform/batch_operators.py` receive input images in the form of CHW, so please use Permute before using these Batch data enhancement operators . If the Gt2xxx Target operator is used, it needs to be placed further back. The Normalize Box operator is recommended to be placed before Gt2xxx Target. After summarizing these constraints, the order of the recommended preprocessing operator is:
  ```
    - XXX: {}
    - ...
    - BatchRandomResize: {...} # Remove it if not needed, and place it in front of Permute if necessary
    - Permute: {} # flush privileges
    - NormalizeBox: {} # If necessary, it is recommended to precede Gt2XXXTarget
    - PadBatch: {...} # If not, you can remove it. If necessary, it is recommended to place it behind Permute
    - Gt2XXXTarget: {...} # It is recommended to place with Pad Batch in the last position
  ```

#### 3.2Custom data enhancement operator
If you need to customize data enhancement operators, you need to understand the logic of data enhancement operators. The Base class of the data enhancement Operator is the `transform/operators.py`class defined in `BaseOperator`, from which both the single image data enhancement Operator and the batch data enhancement Operator inherit. Refer to the source code for the complete definition. The following code shows the key functions of the `BaseOperator` class: the apply and __call__ methods
  ``` python
  class BaseOperator(object):

    ...

    def apply(self, sample, context=None):
        return sample

    def __call__(self, sample, context=None):
        if isinstance(sample, Sequence):
            for i in range(len(sample)):
                sample[i] = self.apply(sample[i], context)
        else:
            sample = self.apply(sample, context)
        return sample
  ```
__call__ method is call entry of `BaseOperator`, Receive one sample(single image) or multiple samples (multiple images) as input, and call the Apply function to process one or more samples. In most cases, you simply inherit from `BaseOperator` and override the apply method or override the __call__ method, as shown below. Define a XXXOp that inherits from Base Operator and register it:
  ```python
  @register_op
  class XXXOp(BaseOperator):
    def __init__(self,...):

      super(XXXImage, self).__init__()
      ...

    # In most cases, you just need to override the Apply method
    def apply(self, sample, context=None):
      ...
      省略对输入的sample具体操作
      ...
      return sample

    # If necessary, override call methods such as Mixup, Gt2XXXTarget, etc
    # def __call__(self, sample, context=None):
    #   ...
    #   The specific operation on the input sample is omitted
    #   ...
    #   return sample
  ```
In most cases, you simply override the Apply method, such as the preprocessor in `transform/operators.py` in addition to Mixup and Cutmix. In the case of batch processing, it is generally necessary to override the call method, such as the preprocessing operator of `transform/batch_operators.py`.

### 4.Reader
The Reader class is defined in `reader.py`, where the `BaseDataLoader` class is defined. `BaseDataLoader` encapsulates a layer on the basis of `paddle.io.DataLoader`, which has all the functions of `paddle.io.DataLoader` and can realize the different needs of `DetDataset` for different models. For example, you can set Reader to control `DetDataset` to support Mixup, Cutmix and other operations. In addition, the Data preprocessing operators are combined into the `DetDataset` and `paddle.io.DataLoader` by the `Compose` and 'Batch Compose' classes, respectively. All Reader classes inherit from the `BaseDataLoader` class. See source code for details.

### 5.Configuration and Operation

#### 5.1 Configuration
The configuration files for modules related to data preprocessing contain the configuration files for Datasets common to all models and the configuration files for readers specific to different models.

##### 5.1.1 Dataset Configuration
The configuration file for the Dataset exists in the `configs/datasets` folder. For example, the COCO dataset configuration file is as follows:
```
metric: COCO # Currently supports COCO, VOC, OID, Wider Face and other evaluation standards
num_classes: 80 # num_classes: The number of classes in the dataset, excluding background classes

TrainDataset:
  !COCODataSet
    image_dir: train2017 # The path where the training set image resides relative to the dataset_dir
    anno_path: annotations/instances_train2017.json # Path to the annotation file of the training set relative to the dataset_dir
    dataset_dir: dataset/coco #The path where the dataset is located relative to the PaddleDetection path
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd'] # Controls the fields contained in the sample output of the dataset, note data_fields are unique to the TrainDataset and must be configured

EvalDataset:
  !COCODataSet
    image_dir: val2017 # The path where the images of the validation set reside relative to the dataset_dir
    anno_path: annotations/instances_val2017.json # The path to the annotation file of the validation set relative to the dataset_dir
    dataset_dir: dataset/coco # The path where the dataset is located relative to the PaddleDetection path
TestDataset:
  !ImageFolder
    anno_path: dataset/coco/annotations/instances_val2017.json # The path of the annotation file,  it is only used to read the category information of the dataset. JSON and TXT formats are supported
    dataset_dir: dataset/coco # The path of the dataset, note if this row is added, `anno_path` will be 'dataset_dir/anno_path`, if not set or removed, `anno_path` is `anno_path`
```
In the YML profile for Paddle Detection, use `!`directly serializes module instances (functions, instances, etc.). The above configuration files are serialized using Dataset.

**Note:**
Please carefully check the configuration path of the dataset before running. During training or verification, if the path of TrainDataset or EvalDataset is wrong, it will download the dataset automatically. When using a user-defined dataset, if the TestDataset path is incorrectly configured during inference, the category of the default COCO dataset will be used.


##### 5.1.2 Reader configuration
The Reader configuration files for yolov3 are defined in `configs/yolov3/_base_/yolov3_reader.yml`. An example Reader configuration is as follows:
```
worker_num: 2
TrainReader:
  sample_transforms:
    - Decode: {}
    ...
  batch_transforms:
    ...
  batch_size: 8
  shuffle: true
  drop_last: true
  use_shared_memory: true

EvalReader:
  sample_transforms:
    - Decode: {}
    ...
  batch_size: 1
  drop_empty: false

TestReader:
  inputs_def:
    image_shape: [3, 608, 608]
  sample_transforms:
    - Decode: {}
    ...
  batch_size: 1
```
You can define different preprocessing operators in Reader, batch_size per gpu, worker_num of Data Loader, etc.

#### 5.2run
In the Paddle Detection training, evaluation, and test runs, Reader iterators are created. The Reader is created in `ppdet/engine/trainer.py`. The following code shows how to create a training-time Reader
``` python
from ppdet.core.workspace import create
# build data loader
self.dataset = cfg['TrainDataset']
self.loader = create('TrainReader')(selfdataset, cfg.worker_num)
```
The Reader for prediction and evaluation is similar to `ppdet/engine/trainer.py`.

> About the data processing module, if you have other questions or suggestions, please send us an issue, we welcome your feedback.
