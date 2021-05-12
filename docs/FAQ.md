## FAQ(常见问题)

**Q:**  为什么我使用单GPU训练loss会出`NaN`? </br>
**A:**  默认学习率是适配多GPU训练(8x GPU)，若使用单GPU训练，须对应调整学习率（例如，除以8）。
计算规则表如下所示: </br>

设`base_lr`为配置文件中默认学习率

设`base_batch_size`为配置文件中默认batch_size

您需要设置的学习率=`训练时使用GPU卡数/8 * 训练时设置的batch_size/base_batch_size * base_lr`


**Q:**  内存溢出怎么办? </br>
**A:**  会影响内存使用量的参数有：`batch_size、worker_num`等。可以先将`worker_num=0, batch_size=1`，学习率按照上面公式进行设置，然后看内存是否还溢出。在内存不溢出前提下，尝试增大`batch_size`或者增大`worker_num`来加速训练。

