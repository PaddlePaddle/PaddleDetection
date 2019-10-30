English | [简体中文](DATA_cn.md)

# Data Pipline

## Introduction

The data pipeline is responsible for loading and converting data. Each
resulting data sample is a tuple of np.ndarrays.
For example, Faster R-CNN training uses samples of this format: `[(im,
im_info, im_id, gt_bbox, gt_class, is_crowd), (...)]`.

### Implementation

The data pipeline consists of four sub-systems: data parsing, image
pre-processing, data conversion and data feeding APIs.

Data samples are collected to form `data.Dataset`s, usually 3 sets are
needed for training, validation, and testing respectively.

First, `data.source` loads the data files into memory, then
`data.transform` processes them, and lastly, the batched samples
are fetched by `data.Reader`.

Sub-systems details:
1. Data parsing
Parses various data sources and creates `data.Dataset` instances. Currently,
following data sources are supported:

- COCO data source

Loads `COCO` type datasets with directory structures like this:

  ```
  dataset/coco/
  ├── annotations
  │   ├── instances_train2014.json
  │   ├── instances_train2017.json
  │   ├── instances_val2014.json
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

- Pascal VOC data source

Loads `Pascal VOC` like datasets with directory structure like this:

  ```
  dataset/voc/
  ├── train.txt
  ├── val.txt
  ├── test.txt
  ├── label_list.txt (optional)
  ├── VOCdevkit/VOC2007
  │   ├── Annotations
  │       ├── 001789.xml
  │       |   ...
  │   ├── JPEGImages
  │       ├── 001789.xml
  │       |   ...
  │   ├── ImageSets
  │       |   ...
  ├── VOCdevkit/VOC2012
  │   ├── Annotations
  │       ├── 003876.xml
  │       |   ...
  │   ├── JPEGImages
  │       ├── 003876.xml
  │       |   ...
  │   ├── ImageSets
  │       |   ...
  |   ...
  ```

**NOTE:** If you set `use_default_label=False` in yaml configs, the `label_list.txt`
of Pascal VOC dataset will be read, otherwise, `label_list.txt` is unnecessary and
the default Pascal VOC label list which defined in
[voc\_loader.py](../ppdet/data/source/voc_loader.py) will be used.

- Roidb data source
A generalized data source serialized as pickle files, which have the following
structure:
```python
(records, cname2id)
# `cname2id` is a `dict` which maps category name to class IDs
# and `records` is a list of dict of this structure:
{
    'im_file': im_fname,    # image file name
    'im_id': im_id,         # image ID
    'h': im_h,              # height of image
    'w': im_w,              # width of image
    'is_crowd': is_crowd,   # crowd marker
    'gt_class': gt_class,   # ground truth class
    'gt_bbox': gt_bbox,     # ground truth bounding box
    'gt_poly': gt_poly,     # ground truth segmentation
}
```

We provide a tool to generate roidb data sources. To convert `COCO` or `VOC`
like dataset, run this command:
```sh
# --type: the type of original data (xml or json)
# --annotation: the path of file, which contains the name of annotation files
# --save-dir: the save path
# --samples: the number of samples (default is -1, which mean all datas in dataset)
python ./ppdet/data/tools/generate_data_for_training.py
            --type=json \
            --annotation=./annotations/instances_val2017.json \
            --save-dir=./roidb \
            --samples=-1
```

 2. Image preprocessing
the `data.transform.operator` module provides operations such as image
decoding, expanding, cropping, etc. Multiple operators are combined to form
larger processing pipelines.

 3. Data transformer
Transform a `data.Dataset` to achieve various desired effects, Notably: the
`data.transform.paralle_map` transformer accelerates image processing with
multi-threads or multi-processes. More transformers can be found in
`data.transform.transformer`.

 4. Data feeding apis
To facilitate data pipeline building, we combine multiple `data.Dataset` to
form a `data.Reader` which can provide data for training, validation and
testing respectively. Users can simply call `Reader.[train|eval|infer]` to get
the corresponding data stream. Many aspect of the `Reader`, such as storage
location, preprocessing pipeline, acceleration mode can be configured with yaml
files.

### APIs

The main APIs are as follows:

1. Data parsing

 - `source/coco_loader.py`: COCO dataset parser. [source](../ppdet/data/source/coco_loader.py)
 - `source/voc_loader.py`: Pascal VOC dataset parser. [source](../ppdet/data/source/voc_loader.py)
 [Note] To use a non-default label list for VOC datasets, a `label_list.txt`
 file is needed, one can use the provided label list
 (`data/pascalvoc/ImageSets/Main/label_list.txt`) or generate a custom one (with `tools/generate_data_for_training.py`). Also, `use_default_label` option should
 be set to `false` in the configuration file
 - `source/loader.py`: Roidb dataset parser. [source](../ppdet/data/source/loader.py)

2. Operator
 `transform/operators.py`: Contains a variety of data augmentation methods, including:
- `DecodeImage`: Read images in RGB format.
- `RandomFlipImage`: Horizontal flip.
- `RandomDistort`: Distort brightness, contrast, saturation, and hue.
- `ResizeImage`: Resize image with interpolation.
- `RandomInterpImage`: Use a random interpolation method to resize the image.
- `CropImage`: Crop image with respect to different scale, aspect ratio, and overlap.
- `ExpandImage`: Pad image to a larger size, padding filled with mean image value.
- `NormalizeImage`: Normalize image pixel values.
- `NormalizeBox`: Normalize the bounding box.
- `Permute`: Arrange the channels of the image and optionally convert image to BGR format.
- `MixupImage`: Mixup two images with given fraction<sup>[1](#mix)</sup>.

<a name="mix">[1]</a> Please refer to [this paper](https://arxiv.org/pdf/1710.09412.pdf)。

`transform/arrange_sample.py`: Assemble the data samples needed by different models.
3. Transformer
`transform/post_map.py`: Transformations that operates on whole batches, mainly for:
- Padding whole batch to given stride values
- Resize images to Multi-scales
- Randomly adjust the image size of the batch data
`transform/transformer.py`: Data filtering batching.
`transform/parallel_map.py`: Accelerate data processing with multi-threads/multi-processes.
4. Reader
`reader.py`: Combine source and transforms, return batch data according to `max_iter`.
`data_feed.py`: Configure default parameters for `reader.py`.


### Usage

#### Canned Datasets

Preset for common datasets, e.g., `COCO` and `Pascal Voc` are included. In
most cases, user can simply use these canned dataset as is. Moreover, the
whole data pipeline is fully customizable through the yaml configuration files.

#### Custom Datasets

- Option 1: Convert the dataset to COCO format.
```sh
 # a small utility (`tools/x2coco.py`) is provided to convert
 # Labelme-annotated dataset or cityscape dataset to COCO format.
 python ./ppdet/data/tools/x2coco.py --dataset_type labelme
                                --json_input_dir ./labelme_annos/
                                --image_input_dir ./labelme_imgs/
                                --output_dir ./cocome/
                                --train_proportion 0.8
                                --val_proportion 0.2
                                --test_proportion 0.0
 # --dataset_type: The data format which is need to be converted. Currently supported are: 'labelme' and 'cityscape'
 # --json_input_dir：The path of json files which are annotated by Labelme.
 # --image_input_dir：The path of images.
 # --output_dir：The path of coverted COCO dataset.
 # --train_proportion：The train proportion of annatation data.
 # --val_proportion：The validation proportion of annatation data.
 # --test_proportion: The inference proportion of annatation data.
```

- Option 2:

1. Add `source/XX_loader.py` and implement the `load` function, following the
   example of `source/coco_loader.py` and `source/voc_loader.py`.
2. Modify the `load` function in `source/loader.py` to make use of the newly
   added data loader.
3. Modify `/source/__init__.py` accordingly.
```python
if data_cf['type'] in ['VOCSource', 'COCOSource', 'RoiDbSource']:
    source_type = 'RoiDbSource'
# Replace the above code with the following code:
if data_cf['type'] in ['VOCSource', 'COCOSource', 'RoiDbSource', 'XXSource']:
    source_type = 'RoiDbSource'
```
4. In the configure file, define the `type` of `dataset` as `XXSource`.

#### How to add data pre-processing？

- To add pre-processing operation for a single image, refer to the classes in
  `transform/operators.py`, and implement the desired transformation with a new
  class.
- To add pre-processing for a batch, one needs to modify the `build_post_map`
  function in `transform/post_map.py`.
