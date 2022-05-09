# Logging

This document talks about how to track metrics and visualize model performance during training. The library currently supports [VisualDL](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/03_VisualDL/visualdl_usage_en.html) and [Weights & Biases](https://docs.wandb.ai).

## VisualDL
Logging to VisualDL is supported only in python >= 3.5. To install VisualDL

```
pip install visualdl
```

PaddleDetection uses a callback to log the training metrics at the end of every step and metrics from the validation step at the end of every epoch. To use VisualDL for visualization, add the `--use_vdl` flag to the training command and `--vdl_log_dir <logs>` to set the directory which stores the records.

For example

```
python tools/train -c config.yml --use_vdl --vdl_log_dir ./logs
```

Another possible way to do this is to add the aforementioned flags to the `config.yml` file.

## Weights & Biases
W&B is a MLOps tool that can be used for experiment tracking, dataset/model versioning, visualizing results and collaborating with colleagues. A W&B logger is integrated directly into PaddleDetection and to use it, first you need to install the wandb sdk and login to your wandb account.

```
pip install wandb
wandb login
```

To use wandb to log metrics while training add the `--use_wandb` flag to the training command and any other arguments for the W&B logger can be provided like this - 

```
python tools/train -c config.yml --use_wandb -o wandb-project=MyDetector wandb-entity=MyTeam wandb-save_dir=./logs
```

The arguments to the W&B logger must be proceeded by `-o` and each invidiual argument must contain the prefix "wandb-".

If this is too tedious, an alternative way is to add the arguments to the `config.yml` file under the `wandb` header. For example

```
use_wandb: True
wandb:
    project: MyProject
    entity: MyTeam
    save_dir: ./logs
```
