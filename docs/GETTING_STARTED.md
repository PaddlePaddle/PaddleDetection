# Getting Started

For setting up the running environment, please refer to [installation
instructions](INSTALL.md).


## Training

#### Single-GPU Training


```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
python tools/train.py -c configs/faster_rcnn_r50_1x.yml
```

#### Multi-GPU Training

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:.
python tools/train.py -c configs/faster_rcnn_r50_1x.yml
```

#### CPU Training

```bash
export CPU_NUM=8
export PYTHONPATH=$PYTHONPATH:.
python tools/train.py -c configs/faster_rcnn_r50_1x.yml -o use_gpu=false
```

##### Optional arguments

- `-r` or `--resume_checkpoint`: Checkpoint path for resuming training. Such as: `-r output/faster_rcnn_r50_1x/10000`
- `--eval`: Whether to perform evaluation in training, default is `False`
- `--output_eval`: If perform evaluation in training, this edits evaluation directory, default is current directory.
- `-d` or `--dataset_dir`: Dataset path, same as `dataset_dir` of configs. Such as: `-d dataset/coco`
- `-o`: Set configuration options in config file. Such as: `-o max_iters=180000`


##### Examples

- Perform evaluation in training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:.
python -u tools/train.py -c configs/faster_rcnn_r50_1x.yml --eval
```

Alternating between training epoch and evaluation run is possible, simply pass
in `--eval` to do so and evaluate at each snapshot_iter. It can be modified at `snapshot_iter` of the configuration file. If evaluation dataset is large and
causes time-consuming in training, we suggest decreasing evaluation times or evaluating after training. When perform evaluation in training,
the best model with highest MAP is saved at each `snapshot_iter`. `best_model` has the same path as `model_final`.


- configuration options and assign Dataset path
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:.
python -u tools/train.py -c configs/faster_rcnn_r50_1x.yml \
                         -d dataset/coco
```


##### NOTES

- `CUDA_VISIBLE_DEVICES` can specify different gpu numbers. Such as: `export CUDA_VISIBLE_DEVICES=0,1,2,3`. GPU calculation rules can refer [FAQ](#faq)
- Dataset is stored in `dataset/coco` by default (configurable).
- Dataset will be downloaded automatically and cached in `~/.cache/paddle/dataset` if not be found locally.
- Pretrained model is downloaded automatically and cached in `~/.cache/paddle/weights`.
- Model checkpoints are saved in `output` by default (configurable).
- To check out hyper parameters used, please refer to the [configs](../configs).
- RCNN models training on CPU is not supported on PaddlePaddle<=1.5.1 and will be fixed on later version.



## Evaluation


```bash
# run on GPU with:
export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c configs/faster_rcnn_r50_1x.yml
```

#### Optional arguments

- `-d` or `--dataset_dir`: Dataset path, same as dataset_dir of configs. Such as: `-d dataset/coco`
- `--output_eval`: Evaluation directory, default is current directory.
- `-o`: Set configuration options in config file. Such as: `-o weights=output/faster_rcnn_r50_1x/model_final`
- `--json_eval`: Whether to eval with already existed bbox.json or mask.json. Default is `False`. Json file directory is assigned by `-f` argument.

#### Examples

- configuration options && assign Dataset path
```bash
# run on GPU with:
export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=0
python -u tools/eval.py -c configs/faster_rcnn_r50_1x.yml \
                        -o weights=output/faster_rcnn_r50_1x/model_final \
                        -d dataset/coco
```

- Evaluation with json
```bash
# run on GPU with:
export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c configs/faster_rcnn_r50_1x.yml \
             --json_eval \
             -f evaluation/
```

The json file must be named bbox.json or mask.json, placed in the `evaluation/` directory. Or without the `-f` parameter, default is the current directory.

#### NOTES

- Checkpoint is loaded from `output` by default (configurable)
- Multi-GPU evaluation for R-CNN and SSD models is not supported at the
moment, but it is a planned feature


## Inference


- Run inference on a single image:

```bash
# run on GPU with:
export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml --infer_img=demo/000000570688.jpg
```

- Multi-image inference:

```bash
# run on GPU with:
export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml --infer_dir=demo
```

#### Optional arguments

- `--output_dir`: Directory for storing the output visualization files.
- `--draw_threshold`: Threshold to reserve the result for visualization. Default is 0.5.
- `--save_inference_model`: Save inference model in output_dir if True.

#### Examples

- Output specified directory && Set up threshold
```bash
# run on GPU with:
export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml \
                      --infer_img=demo/000000570688.jpg \
                      --output_dir=infer_output/ \
                      --draw_threshold=0.5 \
                      -o weights=output/faster_rcnn_r50_1x/model_final
```
The visualization files are saved in `output` by default, to specify a different
path, simply add a `--output_dir=` flag.  
`--draw_threshold` is an optional argument. Default is 0.5. Different thresholds will produce different results depending on the calculation of [NMS](https://ieeexplore.ieee.org/document/1699659). If users want to infer according to customized model path, `-o weights` can be set for specified path.

- Save inference model

```bash
# run on GPU with:
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
python tools/infer.py -c configs/faster_rcnn_r50_1x.yml \
                      --infer_img=demo/000000570688.jpg \
                      --save_inference_model
```

Save inference model by set `--save_inference_model`, which can be loaded by PaddlePaddle predict library.


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
