# 人脸检测模型

## 简介
`face_detection`中提供高效、高速的人脸检测解决方案，包括最先进的模型和经典模型。

![](../../docs/images/12_Group_Group_12_Group_Group_12_935.jpg)

## 模型库

#### WIDER-FACE数据集上的mAP

| 网络结构 | 输入尺寸 | 图片个数/GPU | 学习率策略 | Easy/Medium/Hard Set  | 预测时延（SD855）| 模型大小(MB) | 下载 | 配置文件 |
|:------------:|:--------:|:----:|:-------:|:-------:|:---------:|:----------:|:---------:|:--------:|
| BlazeFace  | 640  |    8    | 1000e     | 0.889 / 0.859 / 0.740 | - | 0.472 |[下载链接](https://paddledet.bj.bcebos.com/models/blazeface_1000e.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/dygraph/configs/face_detection/blazeface_1000e.yml) |

**注意:**  
- 我们使用多尺度评估策略得到`Easy/Medium/Hard Set`里的mAP。具体细节请参考[在WIDER-FACE数据集上评估](#在WIDER-FACE数据集上评估)。

## 快速开始

### 数据准备
我们使用[WIDER-FACE数据集](http://shuoyang1213.me/WIDERFACE/)进行训练和模型测试，官方网站提供了详细的数据介绍。
- WIDER-Face数据源:  
使用如下目录结构加载`wider_face`类型的数据集：

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

- 手动下载数据集：
要下载WIDER-FACE数据集，请运行以下命令：
```
cd dataset/wider_face && ./download_wider_face.sh
```

### 训练与评估
训练流程与评估流程方法与其他算法一致，请参考[GETTING_STARTED_cn.md](../../docs/tutorials/GETTING_STARTED_cn.md)。  
**注意:**
- 人脸检测模型目前不支持边训练边评估。

#### 在WIDER-FACE数据集上评估
评估并生成结果文件：
```shell
python -u tools/eval.py -c configs/face_detection/blazeface_1000e.yml \
       -o weights=output/blazeface_1000e/model_final \
       multi_scale=True
```
设置`multi_scale=True`进行多尺度评估，评估完成后，将在`output/pred`中生成txt格式的测试结果。

- 下载官方评估脚本来评估AP指标：
```
wget http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip
unzip eval_tools.zip && rm -f eval_tools.zip
```
- 在`eval_tools/wider_eval.m`中修改保存结果路径和绘制曲线的名称：
```
# Modify the folder name where the result is stored.
pred_dir = './pred';  
# Modify the name of the curve to be drawn
legend_name = 'Fluid-BlazeFace';
```
- `wider_eval.m` 是评估模块的主要执行程序。运行命令如下：
```
matlab -nodesktop -nosplash -nojvm -r "run wider_eval.m;quit;"
```


## Citations

```
@article{bazarevsky2019blazeface,
      title={BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs},
      author={Valentin Bazarevsky and Yury Kartynnik and Andrey Vakunov and Karthik Raveendran and Matthias Grundmann},
      year={2019},
      eprint={1907.05047},
      archivePrefix={arXiv},
```
