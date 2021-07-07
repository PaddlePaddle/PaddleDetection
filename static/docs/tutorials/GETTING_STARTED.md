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
|          -c              |      ALL       |  Select config file  |  None  |  **The description of configure can refer to [CONFIG.md](../advanced_tutorials/config_doc/CONFIG.md)** |
|          -o              |      ALL       |  Set parameters in configure file  |  None  |  `-o` has higher priority to file configured by `-c`. Such as `-o use_gpu=False max_iter=10000`  |  
|   -r/--resume_checkpoint |     train      |  Checkpoint path for resuming training  |  None  |  `-r output/faster_rcnn_r50_1x/10000`  |
|        --eval            |     train      |  Whether to perform evaluation in training  |  False  |    |
|      --output_eval       |     train/eval |  json path in evalution  |  current path  |  `--output_eval ./json_result`  |
|       --fp16             |     train      |  Whether to enable mixed precision training  |  False  |  GPU training is required  |
|       --loss_scale       |     train      |  Loss scaling factor for mixed precision training  |  8.0  |  enable when `--fp16` is True  |  
|       --json_eval        |       eval     |  Whether to evaluate with already existed bbox.json or mask.json  |  False  |  json path is set in `--output_eval`  |
|       --output_dir       |      infer     |  Directory for storing the output visualization files  |  `./output`  |  `--output_dir output`  |
|    --draw_threshold      |      infer     |  Threshold to reserve the result for visualization  |  0.5  |  `--draw_threshold 0.7`  |
|      --infer_dir         |       infer     |  Directory for images to perform inference on  |  None  |    |
|      --infer_img         |       infer     |  Image path  |  None  |  higher priority over --infer_dir  |
|        --use_vdl          |   train/infer   |  Whether to record the data with [VisualDL](https://github.com/paddlepaddle/visualdl), so as to display in VisualDL  |  False  |  VisualDL requires Python>=3.5   |
|        --vdl\_log_dir     |   train/infer   |  VisualDL logging directory for image  |  train:`vdl_log_dir/scalar` infer: `vdl_log_dir/image`  |  VisualDL requires Python>=3.5   |


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

  When using pre-trained model to fine-tune other task, pretrain\_weights can be used directly. The parameters with different shape will be ignored automatically. For example:


  ```bash
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  # If the shape of parameters in program is different from pretrain_weights,
  # then PaddleDetection will not use such parameters.
  python -u tools/train.py -c configs/faster_rcnn_r50_1x.yml \
                           -o pretrain_weights=output/faster_rcnn_r50_1x/model_final \
  ```

  Besides, the name of parameters which need to ignore can be specified explicitly as well. Two methods can be used:

  1. The excluded pre-trained parameters can be set by `finetune_exclude_pretrained_params` in YAML config
  2. Set -o finetune\_exclude\_pretrained_params in the arguments.

  ```bash
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python -u tools/train.py -c configs/faster_rcnn_r50_1x.yml \
                           -o pretrain_weights=output/faster_rcnn_r50_1x/model_final \
                              finetune_exclude_pretrained_params = ['cls_score','bbox_pred']
  ```

- Training YOLOv3 with fine grained YOLOv3 loss built by Paddle OPs in python

  In order to facilitate the redesign of YOLOv3 loss function, we also provide fine grained YOLOv3 loss function building in python code by common Paddle OPs instead of using `fluid.layers.yolov3_loss`,
  training YOLOv3 with python loss function as follows:

  ```bash
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python -u tools/train.py -c configs/yolov3_darknet.yml \
                           -o use_fine_grained_loss=true
  ```

  Fine grained YOLOv3 loss code is defined in `ppdet/modeling/losses/yolo_loss.py`.

##### NOTES

- `CUDA_VISIBLE_DEVICES` can specify different gpu numbers. Such as: `export CUDA_VISIBLE_DEVICES=0,1,2,3`. GPU calculation rules can refer [FAQ](../FAQ.md)
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
  ```

  The path of model to be evaluted can be both local path and link in [MODEL_ZOO](../MODEL_ZOO_cn.md).

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
                      --use_vdl=Ture
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
