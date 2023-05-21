[简体中文](./pphuman_mtmct.md) | English

# Customized Multi-Target Multi-Camera Tracking Module of PP-Human

## Data Preparation

### Data Format



Multi-target multi-camera tracking, or mtmct is achieved by the pedestrian REID technique. It is trained with a multiclassification model and uses the features before the head of the classification softmax as the retrieval feature vector.

Therefore its format is the same as the multi-classification task. Each pedestrian is assigned an exclusive id, which is different for different pedestrians while the same pedestrian has the same id in different images.

For example, images 0001.jpg, 0003.jpg are the same person, 0002.jpg, 0004.jpg are different pedestrians. Then the labeled ids are.

```
0001.jpg    00001
0002.jpg    00002
0003.jpg    00001
0004.jpg    00003
...
```

### Data Annotation

After understanding the meaning of the `annotation` format above, we can work on the data annotation. The essence of data annotation is that each single person diagram creates an annotation item that corresponds to the id assigned to that pedestrian.

For example:

For an original picture

1) Use bouding boxes to annotate the position of each person in the picture.

2) Each bouding box (corresponding to each person) contains an int id attribute. For example, the person in 0001.jpg in the above example corresponds to id: 1.

After the annotation is completed, use the detection box to intercept each person into a single picture, the picture and id attribute annotation will establish a corresponding relationship. You can also first cut into a single image and then annotate, the result is the same.



## Model Training

Once the data is annotated, it can be used for model training to complete the optimization of the customized model.

There are two main steps to implement: 1) organize the data and annotated data into a training format. 2) modify the configuration file to start training.

### Training data format

The training data consists of the images used for training and a training list bounding_box_train.txt, the location of which is specified in the training configuration, with the following example placement.


```
REID/
|-- data Training image folder
|-- 00001.jpg
|-- 00002.jpg
|-- 0000x.jpg
`-- bounding_box_train.txt List of training data
```

bounding_box_train.txt file contains the names of all training images (file path relative to the root path) + 1 id annotation value

Each line represents a person's image and id annotation result. The format is as follows:

```
0001.jpg    00001
0002.jpg    00002
0003.jpg    00001
0004.jpg    00003
```

Note: The images are separated from the annotated values by a Tab[\t] symbol. This format must be correct, otherwise, the parsing will fail.



### Modify the configuration to start training

First, execute the following command to download the training code (for more environment issues, please refer to [Install_PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/en/installation/ install_paddleclas_en.md):

```
git clone https://github.com/PaddlePaddle/PaddleClas
```

You need to change the following configuration items in the configuration file [softmax_triplet_with_center.yaml](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/configs/reid/strong_ baseline/softmax_triplet_with_center.yaml):

```
  Head:
    name: "FC"
    embedding_size: *feat_dim
    class_num: &class_num 751 #Total number of pedestrian ids

DataLoader:
  Train:
    dataset:
        name: "Market1501"
        image_root: ". /dataset/" #training image root path
        cls_label_path: "bounding_box_train" #training_file_list


  Eval:
    Query:
      dataset:
        name: "Market1501"
        image_root: ". /dataset/" #Evaluated image root path
        cls_label_path: "query" #List of evaluation files
```

Note:

1. Here the image_root path + the relative path of the image in the bounding_box_train.txt corresponds to the full path where the image is stored.

Then run the following command to start the training.

```
#Multi-card training
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml

#Single card training
python3 tools/train.py \
    -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml
```

After the training is completed, you may run the following commands for performance evaluation:

```
#Multi-card evaluation
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/eval.py \
        -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml \
        -o Global.pretrained_model=./output/strong_baseline/best_model

#Single card evaluation
python3 tools/eval.py \
        -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml \
        -o Global.pretrained_model=./output/strong_baseline/best_model
```

### Model Export

Use the following command to export the trained model as an inference deployment model.

```
python3 tools/export_model.py \
    -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml \
    -o Global.pretrained_model=./output/strong_baseline/best_model \
    -o Global.save_inference_dir=deploy/models/strong_baseline_inference
```

After exporting the model, download the [infer_cfg.yml](https://bj.bcebos.com/v1/paddledet/models/pipeline/REID/infer_cfg.yml) file to the newly exported model folder 'strong_baseline_ inference'.

Change the model path `model_dir` in the configuration file `infer_cfg_pphuman.yml` in PP-Human and set `enable`.

```
REID:
 model_dir: [YOUR_DEPLOY_MODEL_DIR]/strong_baseline_inference/
 enable: True
```

Now, the model is ready.
