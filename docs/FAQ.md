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

**Q:** 在配置文件中设置use_process=True，并且运行报错：`not enough space for reason[failed to malloc 601 pages...` </br>
**A:** 当前Reader的共享存储队列空间不足，请增大配置文件`xxx.yml`中的`memsize`,如`memsize: 3G`->`memsize: 6G`。或者配置文件中设置`use_process=False`。
