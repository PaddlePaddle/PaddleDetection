# Multi-Target Multi-Camera Tracking Module of PP-Human

Multi-target multi-camera tracking, or MTMCT, matches the identity of a person in different cameras based on the single-camera tracking. MTMCT is usually applied to the security system and the smart retailing.
The MTMCT module of PP-Human aims to provide a multi-target multi-camera pipleline which is simple, and efficient.

## How to Use

1. Download [REID model](https://bj.bcebos.com/v1/paddledet/models/pipeline/reid_model.zip) and unzip it to ```./output_inference```. For the MOT model, please refer to [mot description](./mot.md).

2. In the MTMCT mode, input videos are required to be put in the same directory. The command line is:
```python
python3 deploy/pphuman/pipeline.py --config deploy/pphuman/config/infer_cfg.yml --video_dir=[your_video_file_directory] --device=gpu
```

3. Configuration can be modified in `./deploy/pphuman/config/infer_cfg.yml`.

```python
python3 deploy/pphuman/pipeline.py
        --config deploy/pphuman/config/infer_cfg.yml
        --video_dir=[your_video_file_directory]
        --device=gpu
        --model_dir reid=reid_best/
```

## Intorduction to the Solution

MTMCT module consists of the multi-target multi-camera tracking pipeline and the REID model.

1. Multi-Target Multi-Camera Tracking Pipeline

```

single-camera tracking[id+bbox]
        │
capture the target in the original image according to bbox——│
        │            │
    REID model      quality assessment (covered or not, complete or not, brightness, etc.)
        │            │
    [feature]        [quality]
        │            │
   datacollector—————│
        │
      sort out and filter features
        │
 calculate the similarity of IDs in the videos
        │
  make the IDs cluster together and rearrange them
```

2. The model solution is [reid-centroids](https://github.com/mikwieczorek/centroids-reid), with ResNet50 as the backbone. It is worth noting that the solution employs different features of the same ID to enhance the similarity.

Under the above circumstances, the REID model used in MTMCT integrates open-source datasets and compresses model features to 128-dimensional features to optimize the generalization. In this way, the actual generalization result becomes much better.

### Other Suggestions

- The provided REID model is obtained from open-source dataset training. It is recommended to add your own data to get a more powerful REID model, notably improving the MTMCT effect.
- The quality assessment is based on simple logic +OpenCV, whose effect is limited. If possible, it is advisable to conduct specific training on the quality assessment model.


### Example

- camera 1:
<div width="1080" align="center">
  <img src="./images/c1.gif"/>
</div>

- camera 2:
<div width="1080" align="center">
  <img src="./images/c2.gif"/>
</div>


## Reference
```
@article{Wieczorek2021OnTU,
  title={On the Unreasonable Effectiveness of Centroids in Image Retrieval},
  author={Mikolaj Wieczorek and Barbara Rychalska and Jacek Dabrowski},
  journal={ArXiv},
  year={2021},
  volume={abs/2104.13643}
}
```
