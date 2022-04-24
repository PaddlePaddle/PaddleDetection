# How to Prepare Training Data
## Directory
- [How to Prepare Training Data](#how-to-prepare-training-data)
  - [Directory](#directory)
    - [Description of Object Detection Data](#description-of-object-detection-data)
    - [Prepare Training Data](#prepare-training-data)
      - [VOC Data](#voc-data)
        - [VOC Dataset Download](#voc-dataset-download)
        - [Introduction to VOC Data Annotation File](#introduction-to-voc-data-annotation-file)
      - [COCO Data](#coco-data)
        - [COCO Data Download](#coco-data-download)
        - [Description of COCO Data Annotation](#description-of-coco-data-annotation)
      - [User Data](#user-data)
        - [Convert User Data to VOC Data](#convert-user-data-to-voc-data)
        - [Convert User Data to COCO Data](#convert-user-data-to-coco-data)
        - [Reader of User Define Data](#reader-of-user-define-data)
      - [Example of User Data Conversion](#example-of-user-data-conversion)

### Description of Object Detection Data
The data of object detection is more complex than classification. In an image, it is necessary to mark the position and category of each object.

The general object position is represented by a rectangular box, which is generally expressed in the following three ways

| Expression  |                                  Explanation                                   |
| :---------: | :----------------------------------------------------------------------------: |
| x1,y1,x2,y2 |    (x1,y1)is the top left coordinate, (x2,y2)is the bottom right coordonate    |
|  x1,y1,w,h  | (x1,y1)is the top left coordinate, w is width of object, h is height of object |
|  xc,yc,w,h  |    (xc,yc)is center of object, w is width of object, h is height of object     |

Common object detection datasets such as Pascal VOC, adopting `[x1,y1,x2,y2]` to express the bounding box of object. COCO uses `[x1,y1,w,h]` , [format](https://cocodataset.org/#format-data).

### Prepare Training Data
PaddleDetection is supported [COCO](http://cocodataset.org) and [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [WIDER-FACE](http://shuoyang1213.me/WIDERFACE/) datasets by default.

It also supports custom data sources including:

(1) Convert custom data to VOC format;  
(2) Convert custom data to COOC format;  
(3) Customize a new data source, and add custom reader;  

firstly, enter `PaddleDetection` root directory

```
cd PaddleDetection/
ppdet_root=$(pwd)
```

#### VOC Data

VOC data is used in [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) competition. Pascal VOC competition not only contains image classification task, but also contains object detection and object segmentation et al., the annotation file contains the ground truth of multiple tasks.
VOC dataset denotes the data of PAscal VOC competition. when customizeing VOC data, For non mandatory fields in the XML file, please select whether to label or use the default value according to the actual situation.

##### VOC Dataset Download  

- Download VOC datasets through code automation. The datasets are large and take a long time to download

    ```
    # Execute code to automatically download VOC dataset
    python dataset/voc/download_voc.py
    ```

    After code execution, the VOC dataset file organization structure is：
    ```
    >>cd dataset/voc/
    >>tree
    ├── create_list.py
    ├── download_voc.py
    ├── generic_det_label_list.txt
    ├── generic_det_label_list_zh.txt
    ├── label_list.txt
    ├── VOCdevkit/VOC2007
    │   ├── annotations
    │       ├── 001789.xml
    │       |   ...
    │   ├── JPEGImages
    │       ├── 001789.jpg
    │       |   ...
    │   ├── ImageSets
    │       |   ...
    ├── VOCdevkit/VOC2012
    │   ├── Annotations
    │       ├── 2011_003876.xml
    │       |   ...
    │   ├── JPEGImages
    │       ├── 2011_003876.jpg
    │       |   ...
    │   ├── ImageSets
    │       |   ...
    |   ...
    ```

    Description of each document
    ```
    # label_list.txt is list of classes name，filename must be label_list.txt. If using VOC dataset, when `use_default_label=true` in config file, this file is not required.

    >>cat label_list.txt
    aeroplane
    bicycle
    ...

    # trainval.txt is file list of trainset
    >>cat trainval.txt
    VOCdevkit/VOC2007/JPEGImages/007276.jpg VOCdevkit/VOC2007/Annotations/007276.xml
    VOCdevkit/VOC2012/JPEGImages/2011_002612.jpg VOCdevkit/VOC2012/Annotations/2011_002612.xml
    ...

    # test.txt is file list of testset
    >>cat test.txt
    VOCdevkit/VOC2007/JPEGImages/000001.jpg VOCdevkit/VOC2007/Annotations/000001.xml
    ...

    # label_list.txt voc list of classes name
    >>cat label_list.txt

    aeroplane
    bicycle
    ...
    ```
- If the VOC dataset has been downloaded
    You can organize files according to the above data file organization structure.

##### Introduction to VOC Data Annotation File

In VOC dataset, Each image file corresponds to an XML file with the same name, the coordinates and categories of the marked object frame in the XML file, such as `2007_002055.jpg`:
![](../images/2007_002055.jpg)

The XML file corresponding to the image contains the basic information of the corresponding image, such as file name, source, image size, object area information and category information contained in the image.

The XML file contains the following fields：
- filename, indicating the image name.
- size, indicating the image size, including: image width, image height and image depth
    ```
    <size>
        <width>500</width>
        <height>375</height>
        <depth>3</depth>
    </size>
    ```
- object field, indict each object, including:

    |      Label       |                                                        Explanation                                                         |
    | :--------------: | :------------------------------------------------------------------------------------------------------------------------: |
    |       name       |                                                    name of object class                                                    |
    |       pose       |                               attitude description of the target object (non required field)                               |
    |    truncated     | If the occlusion of the object exceeds 15-20% and is outside the bounding box，mark it as `truncated` (non required field) |
    |    difficult     |                   objects that are difficult to recognize are marked as`difficult` (non required field)                    |
    | bndbox son laebl |                            (xmin,ymin) top left coordinate, (xmax,ymax) bottom right coordinate                            |


#### COCO Data
COOC data is used in [COCO](http://cocodataset.org) competition. alike, Coco competition also contains multiple competition tasks, and its annotation file contains the annotation contents of multiple tasks.
The coco dataset refers to the data used in the coco competition. Customizing coco data, some fields in JSON file, please select whether to label or use the default value according to the actual situation.


##### COCO Data Download
- The coco dataset is downloaded automatically through the code. The dataset is large and takes a long time to download

    ```
    # automatically download coco datasets by executing code
    python dataset/coco/download_coco.py
    ```

    after code execution, the organization structure of coco dataset file is：
    ```
    >>cd dataset/coco/
    >>tree
    ├── annotations
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   |   ...
    ├── train2017
    │   ├── 000000000009.jpg
    │   ├── 000000580008.jpg
    │   |   ...
    ├── val2017
    │   ├── 000000000139.jpg
    │   ├── 000000000285.jpg
    │   |   ...
    |   ...
    ```
- If the coco dataset has been downloaded  
    The files can be organized according to the above data file organization structure.

##### Description of COCO Data Annotation  
Coco data annotation is to store the annotations of all training images in a JSON file. Data is stored in the form of nested dictionaries.

The JSON file contains the following keys:  
- info，indicating the annotation file info。
- licenses, indicating the label file licenses。
- images, indicating the list of image information in the annotation file, and each element is the information of an image. The following is the information of one of the images:
    ```
    {
        'license': 3,                       # license
        'file_name': '000000391895.jpg',    # file_name
         # coco_url
        'coco_url': 'http://images.cocodataset.org/train2017/000000391895.jpg',
        'height': 360,                      # image height
        'width': 640,                       # image width
        'date_captured': '2013-11-14 11:18:45', # date_captured
        # flickr_url
        'flickr_url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg',
        'id': 391895                        # image id
    }
    ```
- annotations: indicating the annotation information list of the target object in the annotation file. Each element is the annotation information of a target object. The following is the annotation information of one of the target objects:
    ```
    {

        'segmentation':             # object segmentation annotation
        'area': 2765.1486500000005, # object area
        'iscrowd': 0,               # iscrowd
        'image_id': 558840,         # image id
        'bbox': [199.84, 200.46, 77.71, 70.88], # bbox [x1,y1,w,h]
        'category_id': 58,          # category_id
        'id': 156                   # image id
    }
    ```

    ```
    # Viewing coco annotation files
    import json
    coco_anno = json.load(open('./annotations/instances_train2017.json'))

    # coco_anno.keys
    print('\nkeys:', coco_anno.keys())

    # Viewing categories information
    print('\ncategories:', coco_anno['categories'])

    # Viewing the number of images
    print('\nthe number of images：', len(coco_anno['images']))

    # Viewing the number of obejcts
    print('\nthe number of annotation：', len(coco_anno['annotations']))

    # View object annotation information
    print('\nobject annotation information: ', coco_anno['annotations'][0])
    ```

    Coco data is prepared as follows.
    `dataset/coco/`Initial document organization
    ```
    >>cd dataset/coco/
    >>tree
    ├── download_coco.py
    ```

#### User Data
There are three processing methods for user data:  
  (1) Convert user data into VOC data (only include labels necessary for object detection as required)  
  (2) Convert user data into coco data (only include labels necessary for object detection as required)  
  (3) Customize a reader for user data (for complex data, you need to customize the reader)  

##### Convert User Data to VOC Data
After the user dataset is converted to VOC data, the directory structure is as follows (note that the path name and file name in the dataset should not use Chinese as far as possible to avoid errors caused by Chinese coding problems):

```
dataset/xxx/
├── annotations
│   ├── xxx1.xml
│   ├── xxx2.xml
│   ├── xxx3.xml
│   |   ...
├── images
│   ├── xxx1.jpg
│   ├── xxx2.jpg
│   ├── xxx3.jpg
│   |   ...
├── label_list.txt (Must be provided and the file name must be label_list.txt )
├── train.txt (list of trainset ./images/xxx1.jpg ./annotations/xxx1.xml)
└── valid.txt (list of valid file)
```

Description of each document
```
# label_list.txt is a list of category names. The file name must be this
>>cat label_list.txt
classname1
classname2
...

# train.txt is list of trainset
>>cat train.txt
./images/xxx1.jpg ./annotations/xxx1.xml
./images/xxx2.jpg ./annotations/xxx2.xml
...

# valid.txt is list of validset
>>cat valid.txt
./images/xxx3.jpg ./annotations/xxx3.xml
...
```

##### Convert User Data to COCO Data
`x2coco.py` is provided in `./tools/` to convert VOC dataset, labelme labeled dataset or cityscape dataset into coco data, for example:

（1）Conversion of labelme data to coco data:
```bash
python tools/x2coco.py \
                --dataset_type labelme \
                --json_input_dir ./labelme_annos/ \
                --image_input_dir ./labelme_imgs/ \
                --output_dir ./cocome/ \
                --train_proportion 0.8 \
                --val_proportion 0.2 \
                --test_proportion 0.0
```
（2）Convert VOC data to coco data:
```bash
python tools/x2coco.py \
        --dataset_type voc \
        --voc_anno_dir path/to/VOCdevkit/VOC2007/Annotations/ \
        --voc_anno_list path/to/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt \
        --voc_label_list dataset/voc/label_list.txt \
        --voc_out_name voc_train.json
```

After the user dataset is converted to coco data, the directory structure is as follows (note that the path name and file name in the dataset should not use Chinese as far as possible to avoid errors caused by Chinese coding problems):
```
dataset/xxx/
├── annotations
│   ├── train.json  # Annotation file of coco data
│   ├── valid.json  # Annotation file of coco data
├── images
│   ├── xxx1.jpg
│   ├── xxx2.jpg
│   ├── xxx3.jpg
│   |   ...
...
```

##### Reader of User Define Data  
  If new data in the dataset needs to be added to paddedetection, you can refer to the [add new data source] (../advanced_tutorials/READER.md#2.3_Customizing_Dataset) document section in the data processing document to develop corresponding code to complete the new data source support. At the same time, you can read the [data processing document] (../advanced_tutorials/READER.md) for specific code analysis of data processing

The configuration file for the Dataset exists in the `configs/datasets` folder. For example, the COCO dataset configuration file is as follows:
```
metric: COCO # Currently supports COCO, VOC, OID, Wider Face and other evaluation standards
num_classes: 80 # num_classes: The number of classes in the dataset, excluding background classes

TrainDataset:
  !COCODataSet
    image_dir: train2017 # The path where the training set image resides relative to the dataset_dir
    anno_path: annotations/instances_train2017.json # Path to the annotation file of the training set relative to the dataset_dir
    dataset_dir: dataset/coco #The path where the dataset is located relative to the PaddleDetection path
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd'] # Controls the fields contained in the sample output of the dataset, note data_fields are unique to the trainreader and must be configured

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


#### Example of User Data Conversion
  Take [Kaggle Dataset](https://www.kaggle.com/andrewmvd/road-sign-detection) competition data as an example to illustrate how to prepare custom data. The dataset of Kaggle [road-sign-detection](https://www.kaggle.com/andrewmvd/road-sign-detection) competition contains 877 images, four categories：crosswalk，speedlimit，stop，trafficlight. Available for download from kaggle, also available from [link](https://paddlemodels.bj.bcebos.com/object_detection/roadsign_voc.tar).
  Example diagram of road sign dataset:  
  ![](../images/road554.png)

```
# Downing and unziping data
  >>cd $(ppdet_root)/dataset
# Download and unzip the kaggle dataset. The current file organization is as follows

├── annotations
│   ├── road0.xml
│   ├── road1.xml
│   ├── road10.xml
│   |   ...
├── images
│   ├── road0.jpg
│   ├── road1.jpg
│   ├── road2.jpg
│   |   ...
```

The data is divided into training set and test set
```
# Generating label_list.txt
>>echo -e "speedlimit\ncrosswalk\ntrafficlight\nstop" > label_list.txt

# Generating train.txt, valid.txt and test.txt
>>ls images/*.png | shuf > all_image_list.txt
>>awk -F"/" '{print $2}' all_image_list.txt | awk -F".png" '{print $1}'  | awk -F"\t" '{print "images/"$1".png annotations/"$1".xml"}' > all_list.txt

# The proportion of training set, verification set and test set is about 80%, 10% and 10% respectively.
>>head -n 88 all_list.txt > test.txt
>>head -n 176 all_list.txt | tail -n 88 > valid.txt
>>tail -n 701 all_list.txt > train.txt

# Deleting unused files
>>rm -rf all_image_list.txt all_list.txt

The organization structure of the final dataset file is:

├── annotations
│   ├── road0.xml
│   ├── road1.xml
│   ├── road10.xml
│   |   ...
├── images
│   ├── road0.jpg
│   ├── road1.jpg
│   ├── road2.jpg
│   |   ...
├── label_list.txt
├── test.txt
├── train.txt
└── valid.txt

# label_list.txt is list of file name, file name must be label_list.txt
>>cat label_list.txt
crosswalk
speedlimit
stop
trafficlight

# train.txt is the list of training dataset files, and each line is an image path and the corresponding annotation file path, separated by spaces. Note that the path here is a relative path within the dataset folder.
>>cat train.txt
./images/road839.png ./annotations/road839.xml
./images/road363.png ./annotations/road363.xml
...

# valid.txt is the list of validation dataset files. Each line is an image path and the corresponding annotation file path, separated by spaces. Note that the path here is a relative path within the dataset folder.
>>cat valid.txt
./images/road218.png ./annotations/road218.xml
./images/road681.png ./annotations/road681.xml
```

You can also download [the prepared data](https://paddlemodels.bj.bcebos.com/object_detection/roadsign_voc.tar), unzip to `dataset/roadsign_voc/`  
After preparing the data, we should generally understand the data, such as image quantity, image size, number of target areas of each type, target area size, etc. If necessary, clean the data.

Roadsign dataset statistics:

| data  | number of images |
| :---: | :--------------: |
| train |       701        |
| valid |       176        |

**Explanation:**  
  (1) For user data, it is recommended to carefully check the data before training to avoid crash during training due to wrong data annotation format or incomplete image data  
  (2) If the image size is too large, it will occupy more memory without limiting the read data size, which will cause memory / video memory overflow. Please set batch reasonably_ Size, you can try from small to large
