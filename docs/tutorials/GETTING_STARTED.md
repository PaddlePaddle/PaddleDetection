English | [简体中文](GETTING_STARTED_cn.md)

# Getting Started

## Installation

For setting up the running environment, please refer to [installation
instructions](INSTALL_cn.md).



## Data preparation

- Please refer to [PrepareDataSet](PrepareDataSet.md) for data preparation
- Please set the data path for data configuration file in ```configs/datasets```


## Training & Evaluation & Inference

PaddleDetection provides scripts for training, evalution and inference with various features according to different configure.

```bash
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml
# GPU evaluation
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams
# Inference
python tools/infer.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml --infer_img=demo/000000570688.jpg -o weights=https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams
```

### Other argument list

list below can be viewed by `--help`

|         FLAG             |  script supported  |    description    |     default     |      remark      |
| :----------------------: | :------------: | :---------------: | :--------------: | :-----------------: |
|          -c              |      ALL       |  Select config file  |  None  |  **required**, such as `-c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml` |
|          -o              |      ALL       |  Set parameters in configure file  |  None  |  `-o` has higher priority to file configured by `-c`. Such as `-o use_gpu=False`  |  
|        --eval            |     train      |  Whether to perform evaluation in training  |  False  |  set `--eval` if needed  |
|   -r/--resume_checkpoint |     train      |  Checkpoint path for resuming training  |  None  |  such as `-r output/faster_rcnn_r50_1x_coco/10000`  |
|      --slim_config     |     ALL |  Configure file of slim method  |  None  |  such as `--slim_config configs/slim/prune/yolov3_prune_l1_norm.yml`  |
|        --use_vdl          |   train/infer   |  Whether to record the data with [VisualDL](https://github.com/paddlepaddle/visualdl), so as to display in VisualDL  |  False  |  VisualDL requires Python>=3.5   |
|        --vdl\_log_dir     |   train/infer   |  VisualDL logging directory for image  |  train:`vdl_log_dir/scalar` infer: `vdl_log_dir/image`  |  VisualDL requires Python>=3.5   |
|      --output_eval       |   eval |  Directory for storing the evaluation output  | None  |   such as `--output_eval=eval_output`, default is current directory  |
|       --json_eval        |       eval     |  Whether to evaluate with already existed bbox.json or mask.json  |  False  |  set `--json_eval` if needed and json path is set in `--output_eval`  |
|      --classwise         |       eval     |  Whether to eval AP for each class and draw PR curve  |  False  |  set `--classwise` if needed  |
|       --output_dir       |      infer     |  Directory for storing the output visualization files  |  `./output`  |  such as `--output_dir output`  |
|    --draw_threshold      |      infer     |  Threshold to reserve the result for visualization  |  0.5  |   such as `--draw_threshold 0.7`  |
|      --infer_dir         |       infer     |  Directory for images to perform inference on  |  None  | One of `infer_dir` and `infer_img` is requied  |
|      --infer_img         |       infer     |  Image path  |  None  | One of `infer_dir` and `infer_img` is requied, `infer_img` has higher priority over `infer_dir`  |




## Examples

### Training

- Perform evaluation in training

  ```bash
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml --eval
  ```

  Perform training and evalution alternatively and evaluate at each end of epoch. Meanwhile, the best model with highest MAP is saved at each epoch which has the same path as `model_final`.

  If evaluation dataset is large, we suggest modifing `snapshot_epoch` in `configs/runtime.yml` to decrease evaluation times or evaluating after training.

- Fine-tune other task

  When using pre-trained model to fine-tune other task, pretrain\_weights can be used directly. The parameters with different shape will be ignored automatically. For example:


  ```bash
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  # If the shape of parameters in program is different from pretrain_weights,
  # then PaddleDetection will not use such parameters.
  python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
                           -o pretrain_weights=output/faster_rcnn_r50_1x_coco/model_final \
  ```

##### NOTES

- `CUDA_VISIBLE_DEVICES` can specify different gpu numbers. Such as: `export CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Dataset will be downloaded automatically and cached in `~/.cache/paddle/dataset` if not be found locally.
- Pretrained model is downloaded automatically and cached in `~/.cache/paddle/weights`.
- Checkpoints are saved in `output` by default, and can be revised from `save_dir` in `configs/runtime.yml`.


### Evaluation

- Evaluate by specified weights path and dataset path

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python -u tools/eval.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
                          -o weights=https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams
  ```

  The path of model to be evaluted can be both local path and link in [MODEL_ZOO](../MODEL_ZOO_cn.md).

- Evaluate with json

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python tools/eval.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
             --json_eval \
             -output_eval evaluation/
  ```

  The json file must be named bbox.json or mask.json, placed in the `evaluation/` directory.


### Inference

- Output specified directory && Set up threshold

  ```bash
  export CUDA_VISIBLE_DEVICES=0
  python tools/infer.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
                      --infer_img=demo/000000570688.jpg \
                      --output_dir=infer_output/ \
                      --draw_threshold=0.5 \
                      -o weights=output/faster_rcnn_r50_fpn_1x_coco/model_final \
                      --use_vdl=Ture
  ```

  `--draw_threshold` is an optional argument. Default is 0.5.
  Different thresholds will produce different results depending on the calculation of [NMS](https://ieeexplore.ieee.org/document/1699659).


## Deployment

Please refer to [depolyment](../../deploy/README_en.md)

## Model Compression

Please refer to [slim](../../configs/slim/README_en.md)
