English | [简体中文](GETTING_STARTED_cn.md)

# Getting Started

For setting up the running environment, please refer to [installation
instructions](INSTALL.md).


## Training/Evaluation/Inference

PaddleDetection provides scripots for training, evalution and inference with various features according to different configure.

```bash
# set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.
# training in single-GPU and multi-GPU. specify different GPU numbers by CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tools/train.py -c configs/faster_rcnn_r50_1x.yml
# GPU evalution
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c configs/faster_rcnn_r50_1x.yml
# Inference
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml --infer_img=demo/000000570688.jpg
```

### Optional argument list

list below can be viewed by `--help`

|         FLAG             |  script supported  |    description    |     default     |      remark      |
| :----------------------: | :------------: | :---------------: | :--------------: | :-----------------: |
|          -c              |      ALL       |  Select config file  |  None  |  **The whole description of configure can refer to [config_example](config_example)** |
|          -o              |      ALL       |  Set parameters in configure file  |  None  |  `-o` has higher priority to file configured by `-c`. Such as `-o use_gpu=False max_iter=10000`  |  
|   -r/--resume_checkpoint |     train      |  Checkpoint path for resuming training  |  None  |  `-r output/faster_rcnn_r50_1x/10000`  |
|        --eval            |     train      |  Whether to perform evaluation in training  |  False  |    |
|      --output_eval       |     train/eval |  json path in evalution  |  current path  |  `--output_eval ./json_result`  |
|   -d/--dataset_dir       |   train/eval   |  path for dataset, same as dataset_dir in configs  |  None  |  `-d dataset/coco`  |
|       --fp16             |     train      |  Whether to enable mixed precision training  |  False  |  GPU training is required  |
|       --loss_scale       |     train      |  Loss scaling factor for mixed precision training  |  8.0  |  enable when `--fp16` is True  |  
|       --json_eval        |       eval     |  Whether to evaluate with already existed bbox.json or mask.json  |  False  |  json path is set in `--output_eval`  |
|       --output_dir       |      infer     |  Directory for storing the output visualization files  |  `./output`  |  `--output_dir output`  |
|    --draw_threshold      |      infer     |  Threshold to reserve the result for visualization  |  0.5  |  `--draw_threshold 0.7`  |
|      --infer_dir         |       infer     |  Directory for images to perform inference on  |  None  |    |
|      --infer_img         |       infer     |  Image path  |  None  |  higher priority over --infer_dir  |
|        --use_tb          |   train/infer   |  Whether to record the data with [tb-paddle](https://github.com/linshuliang/tb-paddle), so as to display in Tensorboard  |  False  |      |
|        --tb\_log_dir     |   train/infer   |  tb-paddle logging directory for image  |  train:`tb_log_dir/scalar` infer: `tb_log_dir/image`  |     |


## Examples

### Training

- Perform evaluation in training

  ```bash
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python -u tools/train.py -c configs/faster_rcnn_r50_1x.yml --eval
  ```

  Perform training and evalution alternatively and evaluate at each snapshot_iter. Meanwhile, the best model with highest MAP is saved at each `snapshot_iter` which has the same path as `model_final`.

  If evaluation dataset is large, we suggest decreasing evaluation times or evaluating after training.

- Fine-tune other task

  When using pre-trained model to fine-tune other task, two methods can be used:

  1. The excluded pre-trained parameters can be set by `finetune_exclude_pretrained_params` in YAML config
  2. Set -o finetune\_exclude\_pretrained_params in the arguments.

  ```bash
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python -u tools/train.py -c configs/faster_rcnn_r50_1x.yml \
                           -o pretrain_weights=output/faster_rcnn_r50_1x/model_final/ \
                              finetune_exclude_pretrained_params = ['cls_score','bbox_pred']
  ```

##### NOTES

- `CUDA_VISIBLE_DEVICES` can specify different gpu numbers. Such as: `export CUDA_VISIBLE_DEVICES=0,1,2,3`. GPU calculation rules can refer [FAQ](#faq)
- Dataset will be downloaded automatically and cached in `~/.cache/paddle/dataset` if not be found locally.
- Pretrained model is downloaded automatically and cached in `~/.cache/paddle/weights`.
- Checkpoints are saved in `output` by default, and can be revised from save_dir in configure files.
- RCNN models training on CPU is not supported on PaddlePaddle<=1.5.1 and will be fixed on later version.


### Mixed Precision Training

Mixed precision training can be enabled with `--fp16` flag. Currently Faster-FPN, Mask-FPN and Yolov3 have been verified to be working with little to no loss of precision (less than 0.2 mAP)

To speed up mixed precision training, it is recommended to train in multi-process mode, for example

```bash
python -m paddle.distributed.launch --selected_gpus 0,1,2,3,4,5,6,7 tools/train.py --fp16 -c configs/faster_rcnn_r50_fpn_1x.yml
```

If loss becomes `NaN` during training, try tweak the `--loss_scale` value. Please refer to the Nvidia [documentation](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#mptrain) on mixed precision training for details.

Also, please note mixed precision training currently requires changing `norm_type` from `affine_channel` to `bn`.



### Evaluation

- Evaluate by specified weights path and dataset path

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python -u tools/eval.py -c configs/faster_rcnn_r50_1x.yml \
                          -o weights=https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar \
                          -d dataset/coco
  ```

  The path of model to be evaluted can be both local path and link in [MODEL_ZOO](MODEL_ZOO_cn.md).

- Evaluate with json

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python tools/eval.py -c configs/faster_rcnn_r50_1x.yml \
             --json_eval \
             -f evaluation/
  ```

  The json file must be named bbox.json or mask.json, placed in the `evaluation/` directory.

#### NOTES

- Multi-GPU evaluation for R-CNN and SSD models is not supported at the
moment, but it is a planned feature


### Inference

- Output specified directory && Set up threshold

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python tools/infer.py -c configs/faster_rcnn_r50_1x.yml \
                      --infer_img=demo/000000570688.jpg \
                      --output_dir=infer_output/ \
                      --draw_threshold=0.5 \
                      -o weights=output/faster_rcnn_r50_1x/model_final \
                      --use_tb=Ture
  ```

  `--draw_threshold` is an optional argument. Default is 0.5.
  Different thresholds will produce different results depending on the calculation of [NMS](https://ieeexplore.ieee.org/document/1699659).


- Export model

  ```bash
  python tools/export_model.py -c configs/faster_rcnn_r50_1x.yml \
                      --output_dir=inference_model \
                      -o weights=output/faster_rcnn_r50_1x/model_final \
                         FasterRCNNTestFeed.image_shape=[3,800,1333]
  ```

  Save inference model `tools/export_model.py`, which can be loaded by PaddlePaddle predict library.

## FAQ

**Q:**  Why do I get `NaN` loss values during single GPU training? </br>
**A:**  The default learning rate is tuned to multi-GPU training (8x GPUs), it must
be adapted for single GPU training accordingly (e.g., divide by 8).
The calculation rules are as follows，they are equivalent: </br>


| GPU number  | Learning rate  | Max_iters | Milestones       |
| :---------: | :------------: | :-------: | :--------------: |
| 2           | 0.0025         | 720000    | [480000, 640000] |
| 4           | 0.005          | 360000    | [240000, 320000] |
| 8           | 0.01           | 180000    | [120000, 160000] |


**Q:**  How to reduce GPU memory usage? </br>
**A:**  Setting environment variable FLAGS_conv_workspace_size_limit to a smaller
number can reduce GPU memory footprint without affecting training speed.
Take Mask-RCNN (R50) as example, by setting `export FLAGS_conv_workspace_size_limit=512`,
batch size could reach 4 per GPU (Tesla V100 16GB).


**Q:**  How to change data preprocessing? </br>
**A:**  Set `sample_transform` in configuration. Note that **the whole transforms** need to be added in configuration.
For example, `DecodeImage`, `NormalizeImage` and `Permute` in RCNN models. For detail description, please refer
to [config_example](config_example).
