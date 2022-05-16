# Sample and Computation Redistribution for Efficient Face Detection

## Introduction

We reproduce `Sample and Computation Redistribution for Efficient Face Detection`.

## Model Zoo

| Network structure | size | images/GPUs | Learning rate strategy | Easy/Medium/Hard Set  | Prediction delay（SD855）| pretraind Model size(MB) | Download | Configuration File |
|:------------:|:--------:|:----:|:-------:|:-------:|:---------:|:----------:|:---------:|:--------:|
| scrfd2.5g  | 640  |    4    | 640e     | 0.9361 / 0.9229 / 0.7717 | - | 2.6 |[link](https://paddledet.bj.bcebos.com/models/scrfd2_5g.pdparams) | [Configuration File](./scrfd_r50_pafpn_2_5.yml) |

**Notes:**

- All above models are trained on WIDERFace with 4 GPUs.

- Evaluation on WIDERFace can be found [here](../face_detection/README.md).

- Dataset: The retinaface version ofWIDERFace is used, which can be downloaded [here](https://drive.google.com/file/d/1UW3KoApOhusyqSHX96yEDRYiNkd3Iv3Z/view?usp=sharing).

## Citation

```latex
@article{guo2021sample,
  title={Sample and computation redistribution for efficient face detection},
  author={Guo, Jia and Deng, Jiankang and Lattas, Alexandros and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:2105.04714},
  year={2021}
}
```
