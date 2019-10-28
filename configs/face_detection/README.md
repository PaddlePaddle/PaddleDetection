English | [简体中文](README_cn.md)

# FaceDetection
The goal of FaceDetection is to provide efficient and high-speed face detection solutions,
including cutting-edge and classic models.


<div align="center">
  <img src="../../demo/output/12_Group_Group_12_Group_Group_12_935.jpg" />
</div>

## Data Pipline
We use the [WIDER FACE dataset](http://shuoyang1213.me/WIDERFACE/) to carry out the training
and testing of the model, the official website gives detailed data introduction.
- WIDER Face data source:  
Loads `wider_face` type dataset with directory structures like this:

  ```
  dataset/wider_face/
  ├── wider_face_split
  │   ├── wider_face_train_bbx_gt.txt
  │   ├── wider_face_val_bbx_gt.txt
  ├── WIDER_train
  │   ├── images
  │   │   ├── 0--Parade
  │   │   │   ├── 0_Parade_marchingband_1_100.jpg
  │   │   │   ├── 0_Parade_marchingband_1_381.jpg
  │   │   │   │   ...
  │   │   ├── 10--People_Marching
  │   │   │   ...
  ├── WIDER_val
  │   ├── images
  │   │   ├── 0--Parade
  │   │   │   ├── 0_Parade_marchingband_1_1004.jpg
  │   │   │   ├── 0_Parade_marchingband_1_1045.jpg
  │   │   │   │   ...
  │   │   ├── 10--People_Marching
  │   │   │   ...
  ```

- Download dataset manually:  
To download the WIDER FACE dataset, run the following commands:
```
cd dataset/wider_face && ./download.sh
```

- Download dataset automatically:
If a training session is started but the dataset is not setup properly
(e.g, not found in dataset/wider_face), PaddleDetection can automatically
download them from [WIDER FACE dataset](http://shuoyang1213.me/WIDERFACE/),
the decompressed datasets will be cached in ~/.cache/paddle/dataset/ and can be discovered
automatically subsequently.

### Data Augmentation

- **Data-anchor-sampling:** Randomly transform the scale of the image to a certain range of scales,
greatly enhancing the scale change of the face. The specific operation is to obtain $v=\sqrt{width * height}$
according to the randomly selected face height and width, and judge the value of `v` in which interval of
 `[16,32,64,128]`. Assuming `v=45` && `32<v<64`, and any value of `[16,32,64]` is selected with a probability
 of uniform distribution. If `64` is selected, the face's interval is selected in `[64 / 2, min(v * 2, 64 * 2)]`.

- **Other methods:** Including `RandomDistort`,`ExpandImage`,`RandomInterpImage`,`RandomFlipImage` etc.
Please refer to [DATA.md](../../docs/DATA.md#APIs) for details.


##  Benchmark and Model Zoo
Supported architectures is shown in the below table, please refer to
[Algorithm Description](#Algorithm-Description) for details of the algorithm.

|                          | Original | Lite <sup>[1](#lite)</sup> | NAS <sup>[2](#nas)</sup> |
|:------------------------:|:--------:|:--------------------------:|:------------------------:|
| [BlazeFace](#BlazeFace)  | ✓        |                          ✓ | ✓                        |
| [FaceBoxes](#FaceBoxes)  | ✓        |                          ✓ | x                        |

<a name="lite">[1]</a> `Lite` edition means reduces the number of network layers and channels.  
<a name="nas">[2]</a> `NAS` edition means use `Neural Architecture Search` algorithm to
optimized network structure.

**Todo List:**
- [ ] HamBox
- [ ] Pyramidbox

### Model Zoo

#### mAP in WIDER FACE

| Architecture | Type     | Size | Img/gpu | Lr schd | Easy Set  | Medium Set | Hard Set  | Download |
|:------------:|:--------:|:----:|:-------:|:-------:|:---------:|:----------:|:---------:|:--------:|
| BlazeFace    | Original | 640  |    8    | 32w     | **0.915** | **0.892**  | **0.797** | [model](https://paddlemodels.bj.bcebos.com/object_detection/blazeface_original.tar) |
| BlazeFace    | Lite     | 640  |    8    | 32w     | 0.909     | 0.885      | 0.781     | [model](https://paddlemodels.bj.bcebos.com/object_detection/blazeface_lite.tar) |
| BlazeFace    | NAS      | 640  |    8    | 32w     | 0.837     | 0.807      | 0.658     | [model](https://paddlemodels.bj.bcebos.com/object_detection/blazeface_nas.tar) |
| FaceBoxes    | Original | 640  |    8    | 32w     | 0.875     | 0.848      | 0.568     | [model](https://paddlemodels.bj.bcebos.com/object_detection/faceboxes_original.tar) |
| FaceBoxes    | Lite     | 640  |    8    | 32w     | 0.898     | 0.872      | 0.752     | [model](https://paddlemodels.bj.bcebos.com/object_detection/faceboxes_lite.tar) |

**NOTES:**  
- Get mAP in `Easy/Medium/Hard Set` by multi-scale evaluation in `tools/face_eval.py`.
For details can refer to [Evaluation](#Evaluate-on-the-WIDER-FACE).
- BlazeFace-Lite Training and Testing ues [blazeface.yml](../../configs/face_detection/blazeface.yml)
configs file and set `lite_edition: true`.

#### mAP in FDDB

| Architecture | Type     | Size | DistROC | ContROC |
|:------------:|:--------:|:----:|:-------:|:-------:|
| BlazeFace    | Original | 640  | **0.992**   | **0.762**   |
| BlazeFace    | Lite     | 640  | 0.990   | 0.756   |
| BlazeFace    | NAS      | 640  | 0.981   | 0.741   |
| FaceBoxes    | Original | 640  | 0.985   | 0.731   |
| FaceBoxes    | Lite     | 640  | 0.987   | 0.741   |

**NOTES:**  
- Get mAP by multi-scale evaluation on the FDDB dataset.
For details can refer to [Evaluation](#Evaluate-on-the-FDDB).

#### Infer Time and Model Size comparison  

| Architecture | Type     | Size | P4 (ms)   | CPU (ms) | ARM (ms)   | File size (MB) | Flops     |
|:------------:|:--------:|:----:|:---------:|:--------:|:----------:|:--------------:|:---------:|
| BlazeFace    | Original | 128  | -         | -        | -          | -              | -         |
| BlazeFace    | Lite     | 128  | -         | -        | -          | -              | -         |
| BlazeFace    | NAS      | 128  | -         | -        | -          | -              | -         |
| FaceBoxes    | Original | 128  | -         | -        | -          | -              | -         |
| FaceBoxes    | Lite     | 128  | -         | -        | -          | -              | -         |
| BlazeFace    | Original | 320  | -         | -        | -          | -              | -         |
| BlazeFace    | Lite     | 320  | -         | -        | -          | -              | -         |
| BlazeFace    | NAS      | 320  | -         | -        | -          | -              | -         |
| FaceBoxes    | Original | 320  | -         | -        | -          | -              | -         |
| FaceBoxes    | Lite     | 320  | -         | -        | -          | -              | -         |
| BlazeFace    | Original | 640  | -         | -        | -          | -              | -         |
| BlazeFace    | Lite     | 640  | -         | -        | -          | -              | -         |
| BlazeFace    | NAS      | 640  | -         | -        | -          | -              | -         |
| FaceBoxes    | Original | 640  | -         | -        | -          | -              | -         |
| FaceBoxes    | Lite     | 640  | -         | -        | -          | -              | -         |


**NOTES:**  
- CPU: i5-7360U @ 2.30GHz. Single core and single thread.



## Get Started
`Training` and `Inference` please refer to [GETTING_STARTED.md](../../docs/GETTING_STARTED.md)
- **NOTES:**     
- `BlazeFace` and `FaceBoxes` is trained in 4 GPU with `batch_size=8` per gpu (total batch size as 32) 
and trained 320000 iters.(If your GPU count is not 4, please refer to the rule of training parameters 
in the table of [calculation rules](../../docs/GETTING_STARTED.md#faq))
- Currently we do not support evaluation in training.

### Evaluation
```
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
python tools/face_eval.py -c configs/face_detection/blazeface.yml
```
- Optional arguments
- `-d` or `--dataset_dir`: Dataset path, same as dataset_dir of configs. Such as: `-d dataset/wider_face`.
- `-f` or `--output_eval`: Evaluation file directory, default is `output/pred`.
- `-e` or `--eval_mode`: Evaluation mode, include `widerface` and `fddb`, default is `widerface`.
- `--multi_scale`: If you add this action button in the command, it will select `multi_scale` evaluation. 
Default is `False`, it will select `single-scale` evaluation.

After the evaluation is completed, the test result in txt format will be generated in `output/pred`,
and then mAP will be calculated according to different data sets. If you set `--eval_mode=widerface`,
it will [Evaluate on the WIDER FACE](#Evaluate-on-the-WIDER-FACE).If you set `--eval_mode=fddb`,
it will [Evaluate on the FDDB](#Evaluate-on-the-FDDB).

#### Evaluate on the WIDER FACE
- Download the official evaluation script to evaluate the AP metrics:
```
wget http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip
unzip eval_tools.zip && rm -f eval_tools.zip
```
- Modify the result path and the name of the curve to be drawn in `eval_tools/wider_eval.m`:
```
# Modify the folder name where the result is stored.
pred_dir = './pred';  
# Modify the name of the curve to be drawn
legend_name = 'Fluid-BlazeFace';
```
- `wider_eval.m` is the main execution program of the evaluation module. The run command is as follows:
```
matlab -nodesktop -nosplash -nojvm -r "run wider_eval.m;quit;"
```

#### Evaluate on the FDDB
[FDDB dataset](http://vis-www.cs.umass.edu/fddb/) details can refer to FDDB's official website.   
- Download the official dataset and evaluation script to evaluate the ROC metrics:
```
#external link to the Faces in the Wild data set
wget http://tamaraberg.com/faceDataset/originalPics.tar.gz
#The annotations are split into ten folds. See README for details.
wget http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz
#information on directory structure and file formats
wget http://vis-www.cs.umass.edu/fddb/README.txt
```
- Install OpenCV: Requires [OpenCV library](http://sourceforge.net/projects/opencvlibrary/)  
If the utility 'pkg-config' is not available for your operating system,
edit the Makefile to manually specify the OpenCV flags as following:
```
INCS = -I/usr/local/include/opencv
LIBS = -L/usr/local/lib -lcxcore -lcv -lhighgui -lcvaux -lml
```

- Compile FDDB evaluation code: execute `make` in evaluation folder.

- Generate full image path list and groundtruth in FDDB-folds. The run command is as follows:
```
cat `ls|grep -v"ellipse"` > filePath.txt` and `cat *ellipse* > fddb_annotFile.txt`
```
- Evaluation
Finally evaluation command is:
```
./evaluate -a ./FDDB/FDDB-folds/fddb_annotFile.txt \
           -d DETECTION_RESULT.txt -f 0 \
           -i ./FDDB -l ./FDDB/FDDB-folds/filePath.txt \
           -r ./OUTPUT_DIR -z .jpg
```
**NOTES:** The interpretation of the argument can be performed by `./evaluate --help`.

## Algorithm Description

### BlazeFace
**Introduction:**  
[BlazeFace](https://arxiv.org/abs/1907.05047) is Google Research published face detection model.
It's lightweight but good performance, and tailored for mobile GPU inference. It runs at a speed
of 200-1000+ FPS on flagship devices.

**Particularity:**  
- Anchor scheme stops at 8×8(input 128x128), 6 anchors per pixel at that resolution.
- 5 single, and 6 double BlazeBlocks: 5×5 depthwise convs, same accuracy with fewer layers.
- Replace the non-maximum suppression algorithm with a blending strategy that estimates the
regression parameters of a bounding box as a weighted mean between the overlapping predictions.

**Edition information:**
- Original: Reference original paper reproduction.
- Lite: Replace 5x5 conv with 3x3 conv, fewer network layers and conv channels.
- NAS: use `Neural Architecture Search` algorithm to optimized network structure,
less network layer and conv channel number than `Lite`.

### FaceBoxes
**Introduction:**     
[FaceBoxes](https://arxiv.org/abs/1708.05234) which named A CPU Real-time Face Detector
with High Accuracy is face detector proposed by Shifeng Zhang, with high performance on
both speed and accuracy. This paper is published by IJCB(2017).

**Particularity:**
- Anchor scheme stops at 20x20, 10x10, 5x5, which network input size is 640x640,
including 3, 1, 1 anchors per pixel at each resolution. The corresponding densities
are 1, 2, 4(20x20), 4(10x10) and 4(5x5).
- 2 convs with CReLU, 2 poolings, 3 inceptions and 2 convs with ReLU.
- Use density prior box to improve detection accuracy.

**Edition information:**
- Original: Reference original paper reproduction.
- Lite: 2 convs with CReLU, 1 pooling, 2 convs with ReLU, 3 inceptions and 2 convs with ReLU.
Anchor scheme stops at 80x80 and 40x40, including 3, 1 anchors per pixel at each resolution.
The corresponding densities are 1, 2, 4(80x80) and 4(40x40), using less conv channel number than lite.


## Contributing
Contributions are highly welcomed and we would really appreciate your feedback!!
