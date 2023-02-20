[简体中文](pphuman_attribute.md) | English

# Customized Pedestrian Attribute Recognition

## Data Preparation

### Data format

We use the PA100K attribute annotation format, with a total of 26 attributes.

The names, locations, and the number of these 26 attributes are shown in the table below.

| Attribute                                                                       | index                  | length |
|:------------------------------------------------------------------------------- |:---------------------- |:------ |
| 'Hat','Glasses'                                                                 | [0, 1]                 | 2      |
| 'ShortSleeve','LongSleeve','UpperStride','UpperLogo','UpperPlaid','UpperSplice' | [2, 3, 4, 5, 6, 7]     | 6      |
| 'LowerStripe','LowerPattern','LongCoat','Trousers','Shorts','Skirt&Dress'       | [8, 9, 10, 11, 12, 13] | 6      |
| 'boots'                                                                         | [14, ]                 | 1      |
| 'HandBag','ShoulderBag','Backpack','HoldObjectsInFront'                         | [15, 16, 17, 18]       | 4      |
| 'AgeOver60', 'Age18-60', 'AgeLess18'                                            | [19, 20, 21]           | 3      |
| 'Female'                                                                        | [22, ]                 | 1      |
| 'Front','Side','Back'                                                           | [23, 24, 25]           | 3      |

Examples:

[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

The first group: position [0, 1] values are [0, 1], which means'no hat', 'has glasses'.

The second group: position [22, ] values are [0, ], indicating that the gender attribute is 'male', otherwise it is 'female'.

The third group: position [23, 24, 25] values are [0, 1, 0], indicating that the direction attribute is 'side'.

Other groups follow in this order



### Data Annotation

After knowing the purpose of the above `attribute annotation` format, we can start to annotate data. The essence is that each single-person image creates a set of 26 annotation items, corresponding to the attribute values at 26 positions.

Examples:

For an original image:

1) Using bounding boxes to annotate the position of each person in the picture.

2) Each detection box (corresponding to each person) contains 26 attribute values which are represented by 0 or 1. It corresponds to the above 26 attributes. For example, if the picture is 'Female', then the 22nd bit of the array is 0. If the person is between 'Age18-60', then the corresponding value at position [19, 20, 21] is [0, 1, 0], or if the person matches 'AgeOver60', then the corresponding value is [1, 0, 0].

After the annotation is completed, the model will use the detection box to intercept each person into a single-person picture, and its picture establishes a corresponding relationship with the 26 attribute annotation. It is also possible to cut into a single-person image first and then annotate it. The results are the same.



## Model Training

Once the data is annotated, it can be used for model training to complete the optimization of the customized model.

There are two main steps: 1) Organize the data and annotated data into the training format. 2) Modify the configuration file to start training.

### Training data format

The training data includes the images used for training and a training list called train.txt. Its location is specified in the training configuration, with the following example:

```
Attribute/
|-- data Training images folder
|-- 00001.jpg
|-- 00002.jpg
| `-- 0000x.jpg
train.txt List of training data
```

train.txt file contains the names of all training images (file path relative to the root path) + 26 annotation values

Each line of it represents a person's image and annotation result. The format is as follows:

```
00001.jpg    0,0,1,0,....
```

Note 1) The images are separated by Tab[\t], 2) The annotated values are separated by commas [,]. If the format is wrong, the parsing will fail.



### Modify the configuration to start training

First run the following command to download the training code (for more environmental issues, please refer to [Install_PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/en/installation/ install_paddleclas_en.md)):

```
git clone https://github.com/PaddlePaddle/PaddleClas
```

You need to modify the following configuration in the configuration file `PaddleClas/blob/develop/ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml`

```
DataLoader:
  Train:
    Train: dataset:
      name: MultiLabelDataset
      image_root: "dataset/pa100k/" #Specify the root path of training image
      cls_label_path: "dataset/pa100k/train_list.txt" #Specify the location of the training list file
      label_ratio: True
      transform_ops:

  Eval:
    dataset:
      name: MultiLabelDataset
      image_root: "dataset/pa100k/" #Specify the root path of evaluated image
      cls_label_path: "dataset/pa100k/val_list.txt" #Specify the location of the evaluation list file
      label_ratio: True
      transform_ops:
```

Note:

1. here image_root path and the relative path of the image in train.txt, corresponding to the full path of the image.
2. If you modify the number of attributes, the number of attribute types in the content configuration item should also be modified accordingly.

```
# model architecture
Arch:
name: "PPLCNet_x1_0"
pretrained: True
use_ssld: True
class_num: 26           #Attribute classes and numbers
```

Then run the following command to start training:

```
#Multi-card training
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml

#Single card training
python3 tools/train.py \
        -c ./ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml
```

You can run the following commands for performance evaluation after the training is completed:

```
#Multi-card evaluation
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/eval.py \
        -c ./ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml \
        -o Global.pretrained_model=./output/PPLCNet_x1_0/best_model

#Single card evaluation
python3 tools/eval.py \
        -c ./ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml \
        -o Global.pretrained_model=./output/PPLCNet_x1_0/best_model
```

### Model Export

Use the following command to export the trained model as an inference deployment model.

```
python3 tools/export_model.py \
    -c ./ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_person_attribute_infer
```

After exporting the model, you need to download the [infer_cfg.yml](https://bj.bcebos.com/v1/paddledet/models/pipeline/infer_cfg.yml) file and put it into the exported model folder `PPLCNet_x1_0_person_ attribute_infer` .

When you use the model, you need to modify the new model path `model_dir` entry and set `enable: True` in the configuration file of PP-Human `. /deploy/pipeline/config/infer_cfg_pphuman.yml` .

```
ATTR:
  model_dir: [YOUR_DEPLOY_MODEL_DIR]/PPLCNet_x1_0_person_attribute_infer/   #The exported model location
  enable: True                                                              #Whether to enable the function
```



Now, the model is ready for you.

 To this point,  a new attribute category recognition task is completed.



## Adding or deleting attributes

The above is the annotation and training process with 26 attributes.

If the attributes need to be added or deleted, you need to

1) New attribute category information needs to be added or deleted when annotating the data.

2) Modify the number and name of attributes used in train.txt corresponding to the training.

3) Modify the training configuration, for example, the number of attributes in the ``PaddleClas/blob/develop/ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml`` file, for details, please see the ``Modify configuration to start training`` section above.

Example of adding attributes.

1. Continue to add new attribute annotation values after 26 values when annotating the data.
2. Add new attribute values to the annotated values in the train.txt file as well.
3. The above is the annotation and training process with 26 attributes.

   If the attributes need to be added or deleted, you need to
   1) New attribute category information needs to be added or deleted when annotating the data.

   2) Modify the number and name of attributes used in train.txt corresponding to the training.

   3) Modify the training configuration, for example, the number of attributes in the ``PaddleClas/blob/develop/ppcls/configs/PULC/person_attribute/PPLCNet_x1_0.yaml`` file, for details, please see the ``Modify configuration to start training`` section above.

   Example of adding attributes.

   1. Continue to add new attribute annotation values after 26 values when annotating the data.
   2. Add new attribute values to the annotated values in the train.txt file as well.
   3. Note that the correlation of attribute types and values in train.txt needs to be fixed, for example, the [19, 20, 21] position indicates age, and all images should use the [19, 20, 21] position to indicate age.



   The same applies to the deletion of attributes.
   For example, if the age attribute is not needed, the values in positions [19, 20, 21] can be removed. You can simply remove all the values in positions 19-21 from the 26 numbers marked in train.txt, and you no longer need to annotate these 3 attribute values.
