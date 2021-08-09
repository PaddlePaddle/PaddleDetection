# FAQ：第一期

**Q：**SOLOv2训练mAP值宽幅震荡，无上升趋势，检测效果不好，检测置信度超过了1的原因是？

**A：** SOLOv2训练不收敛的话，先更新PaddleDetection到release/2.2或者develop分支尝试。



**Q：** Optimizer中优化器支持哪几种？

**A：** Paddle中支持的优化器[Optimizer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html )在PaddleDetection中均支持，需要手动修改下配置文件即可。



**Q：** 在tools/infer.py加入如下函数，得到FLOPs值为-1,请问原因？

**A：** 更新PaddleDetection到release/2.2或者develop分支，`print_flops`设为True即可打印FLOPs。



**Q：** 使用官方的ReID模块时遇到了模块未注册的问题

**A：** 请尝试`pip uninstall paddledet`并重新安装，或者`python setup.py install`。



**Q：** 大规模实用目标检测模型有动态图版本吗，或者可以转换为动态图版本吗？

**A：** 大规模实用模型的动态图版本正在整理，我们正在开发更大规模的通用预训练模型，预计在2.3版本中发布。



**Q：** Develop分支下FairMot预测视频问题：预测视频时不会完全运行完毕。比如用一个300frame的视频，代码会保存预测结果的每一帧图片，但只保存到299张就没了，并且也没有预测好的视频文件生成，该如何解决？

**A：** 已经支持自己设置帧率infer视频，请使用develop分支或release/2.2分支，命令如下：

```
CUDA_VISIBLE_DEVICES=0 python tools/infer_mot.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams --video_file={your video name}.mp4 --frame_rate=20 --save_videos
```



**Q：** 使用YOLOv3模型如何通过yml文件修改输入图片尺寸？

**A：** 模型预测部署需要用到指定的尺寸时，首先在训练前需要修改`configs/_base_/yolov3_reader.yml`中的`TrainReader`的`BatchRandomResize`中`target_size`包含指定的尺寸，训练完成后，在评估或者预测时，需要将`EvalReader`和`TestReader`中的`Resize`的`target_size`修改成对应的尺寸，如果是需要模型导出(export_model)，则需要将`TestReader`中的`image_shape`修改为对应的图片输入尺寸 。



**Q：** 以前的模型都是用静态图训练的，现在想用动态图训练，但想加载原来静态图的模型作为预训练模型，可以直接用加载静态图保存的模型断点吗？如不行，有其它方法吗？

**A：** 静态图和动态图模型的权重的key做下映射一一对应转过去是可以的，可以参考[这个代码](https://github.com/nemonameless/weights_st2dy )。但是不保证所有静态图的权重的key映射都能对应上，静态图是把背景也训练了，动态图去背景类训的，而且现有动态图模型训出来的一般都比以前静态图更高，资源时间够的情况下建议还是直接训动态图版本。



**Q：** TTFNet训练过程中hm_loss异常

**A：** 如果是单卡的话学习率需要对应降低8倍。另外ttfnet模型因为自身设置的学习率比较大，可能会出现其他数据集训练出现不稳定的情况。建议pretrain_weights加载官方release出的coco数据集上训练好的模型，然后将学习率再调低一些。
