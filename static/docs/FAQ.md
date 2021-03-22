## FAQ(常见问题)

**Q:**  为什么我使用单GPU训练loss会出`NaN`? </br>
**A:**  默认学习率是适配多GPU训练(8x GPU)，若使用单GPU训练，须对应调整学习率（例如，除以8）。
计算规则表如下所示，它们是等价的，表中变化节点即为`piecewise decay`里的`boundaries`: </br>


| GPU数  | 学习率  | 最大轮数 | 变化节点       |
| :---------: | :------------: | :-------: | :--------------: |
| 2           | 0.0025         | 720000    | [480000, 640000] |
| 4           | 0.005          | 360000    | [240000, 320000] |
| 8           | 0.01           | 180000    | [120000, 160000] |


**Q:**  如何减少GPU显存使用率? </br>
**A:**  可通过设置环境变量`FLAGS_conv_workspace_size_limit`为较小的值来减少显存消耗，并且不
会影响训练速度。以Mask-RCNN（R50）为例，设置`export FLAGS_conv_workspace_size_limit = 512`，
batch size可以达到每GPU 4 (Tesla V100 16GB)。


**Q:**  哪些参数会影响内存使用量? </br>
**A:**  会影响内存使用量的参数有：`是否使用多进程use_process、 batch_size、reader中的bufsize、reader中的memsize、数据预处理中的RandomExpand ratio参数、以及图像本身大小`等。


**Q:**  如何修改数据预处理? </br>
**A:**  可在配置文件中设置 `sample_transform`。注意需要在配置文件中加入**完整预处理**
例如RCNN模型中`DecodeImage`, `NormalizeImage` and `Permute`。

**Q:** affine_channel和batch norm是什么关系? </br>
**A:** 在RCNN系列模型加载预训练模型初始化，有时候会固定住batch norm的参数, 使用预训练模型中的全局均值和方式，并且batch norm的scale和bias参数不更新，已发布的大多ResNet系列的RCNN模型采用这种方式。这种情况下可以在config中设置norm_type为bn或affine_channel, freeze_norm为true (默认为true)，两种方式等价。affne_channel的计算方式为`scale * x + bias`。只不过设置affine_channel时，内部对batch norm的参数自动做了融合。如果训练使用的affine_channel，用保存的模型做初始化，训练其他任务时，既可使用affine_channel, 也可使用batch norm, 参数均可正确加载。

**Q:** 某些配置项会在多个模块中用到(如 `num_classes`)，如何避免在配置文件中多次重复设置？ </br>
**A:** 框架提供了 `__shared__` 标记来实现配置的共享，用户可以标记参数，如 `__shared__ = ['num_classes']` ，配置数值作用规则如下：

1.  如果模块配置中提供了 `num_classes` ，会优先使用其数值。
2.  如果模块配置中未提供 `num_classes` ，但配置文件中存在全局键值，那么会使用全局键值。
3.  两者均为配置的情况下，将使用默认值(`81`)。

**Q:** 在配置文件中设置use_process=True，并且运行报错：`not enough space for reason[failed to malloc 601 pages...` </br>
**A:** 当前Reader的共享存储队列空间不足，请增大配置文件`xxx.yml`中的`memsize`,如`memsize: 3G`->`memsize: 6G`。或者配置文件中设置`use_process=False`。
