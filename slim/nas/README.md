>运行该示例前请安装Paddle1.6或更高版本

# 神经网络搜索(NAS)教程

## 概述

我们选取人脸检测的BlazeFace模型作为神经网络搜索示例，该示例使用[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)
辅助完成神经网络搜索实验，具体技术细节，请您参考[神经网络搜索策略](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/quick_start/nas_tutorial.md)。<br>
基于PaddleSlim进行搜索实验过程中，搜索限制条件可以选择是浮点运算数(FLOPs)限制还是硬件延时(latency)限制，硬件延时限制需要提供延时表。本示例提供一份基于blazeface搜索空间的硬件延时表，名称是latency_855.txt(基于PaddleLite在骁龙855上测试的延时)，可以直接用该表进行blazeface的硬件延时搜索实验。<br>
硬件延时表每个字段的含义可以参考：[硬件延时表说明](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/table_latency.md)


## 定义搜索空间
在BlazeFace模型的搜索实验中，我们采用了SANAS的方式进行搜索，本次实验会对网络模型中的通道数和卷积核尺寸进行搜索。
所以我们定义了如下搜索空间：
- 初始化通道模块`blaze_filter_num1`：定义了BlazeFace第一个模块中通道数变化区间，人为定义了较小的通道数区间；
- 单blaze模块`blaze_filter_num2`: 定义了BlazeFace单blaze模块中通道数变化区间，人为定义了适中的通道数区间；
- 过渡blaze模块`mid_filter_num`：定义了BlazeFace由单blaze模块到双blaze模块的过渡区间；
- 双blaze模块`double_filter_num`：定义了BlazeFace双blaze模块中通道数变化区间，人为定义了较大的通道数区间；
- 卷积核尺寸`use_5x5kernel`：定义了BlazeFace中卷积和尺寸大小是3x3或者5x5。由于提供的延时表中只统计了3x3卷积的延时，所以启动硬件延时搜索实验时，需要把卷积核尺寸固定为3x3。

根据定义的搜索空间各个区间，我们的搜索空间tokens共9位，变化区间在([0, 0, 0, 0, 0, 0, 0, 0, 0], [7, 9, 12, 12, 6, 6, 6, 6, 2])范围内。硬件延时搜索实验时，token的变化区间在([0, 0, 0, 0, 0, 0, 0, 0, 0], [7, 9, 12, 12, 6, 6, 6, 6, 1])范围内。

9位tokens分别表示：

- tokens[0]：初始化通道数 = blaze_filter_num1[tokens[0]]
- tokens[1]：单blaze模块通道数 = blaze_filter_num2[tokens[1]]
- tokens[2]-tokens[3]：双blaze模块起始通道数 = double_filter_num[tokens[2/3]]
- tokens[4]-tokens[7]：过渡blaze模块通道数 = [tokens[4/5/6/7]]
- tokens[8]：卷积核尺寸使用5x5 = True if use_5x5kernel[tokens[8]] else False

我们人为定义三个单blaze模块与4个双blaze模块，定义规则如下：
```
blaze_filters = [[self.blaze_filter_num1[tokens[0]], self.blaze_filter_num1[tokens[0]]],
                [self.blaze_filter_num1[tokens[0]], self.blaze_filter_num2[tokens[1]], 2],
                [self.blaze_filter_num2[tokens[1]], self.blaze_filter_num2[tokens[1]]]]

double_blaze_filters = [
       [self.blaze_filter_num2[tokens[1]], self.mid_filter_num[tokens[4]], self.double_filter_num[tokens[2]], 2],
       [self.double_filter_num[tokens[2]], self.mid_filter_num[tokens[5]], self.double_filter_num[tokens[2]]],
       [self.double_filter_num[tokens[2]], self.mid_filter_num[tokens[6]], self.double_filter_num[tokens[3]], 2],
       [self.double_filter_num[tokens[3]], self.mid_filter_num[tokens[7]], self.double_filter_num[tokens[3]]]]
```
blaze_filters与double_blaze_filters字段请参考[blazenet.py](https://github.com/PaddlePaddle/PaddleDetection/blob/master/ppdet/modeling/backbones/blazenet.py)中定义。

初始化tokens为：[2, 1, 3, 8, 2, 1, 2, 1, 1]。

## 开始搜索
首先需要安装PaddleSlim，请参考[安装教程](https://paddlepaddle.github.io/PaddleSlim/#_2)。

然后进入 `slim/nas`目录中，修改blazeface.yml配置，配置文件中搜索配置字段含义请参考[NAS-API文档](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/nas_api.rst)<br>

配置文件blazeface.yml中的`Constraint`字段表示当前搜索实验的搜索限制条件实例: <br>
- `ctype`：具体的限制条件，可以设置为flops或者latency，分别表示浮点运算数限制和硬件延时限制。
- `max_constraint`：限制条件的最大值。
- `min_constraint`：限制条件的最小值。
- `table_file`：读取的硬件延时表文件的路径，这个字段只有在硬件延时搜索实验中才会用到。

然后开始搜索实验：
```
cd slim/nas
python -u train_nas.py -c blazeface.yml
```
**注意:**

搜索过程中为了加速，在`blazeface.yml`中去掉了数据预处理`CropImageWithDataAchorSampling`的操作。

训练完成后会获得最佳tokens，以及对应的`BlazeFace-NAS`的网络结构：
```
------------->>> BlazeFace-NAS structure start: <<<----------------
BlazeNet:
  blaze_filters: XXX
  double_blaze_filters: XXX
  use_5x5kernel: XXX
  with_extra_blocks: XXX
  lite_edition: XXX
-------------->>> BlazeFace-NAS structure end! <<<-----------------
```

## 训练、评估与预测
- （1）修改配置文件：

根据搜索得到的`BlazeFace-NAS`的网络结构修改`blazeface.yml`中的`BlazeNet`模块。

- （2）训练、评估与预测：

启动完整的训练评估实验，可参考PaddleDetection的[训练、评估与预测流程](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/tutorials/GETTING_STARTED_cn.md)

## 实验结果
请参考[人脸检测模型库](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/featured_model/FACE_DETECTION.md)中BlazeFace-NAS的实验结果。

## FAQ
- 运行报错：`socket.error: [Errno 98] Address already in use`。

解决方法：当前端口被占用，请修改blazeface.yml中的`server_port`端口。

- 运行报错：`not enough space for reason[failed to malloc 601 pages...`

解决方法：当前reader的共享存储队列空间不足，请增大blazeface.yml中的`memsize`。
