# 自定义OP编译
旋转框IOU计算OP是参考[自定义外部算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_custom_op.html) 。

## 1. 环境依赖
- Paddle >= 2.0.1
- gcc 8.2

## 2. 调用方式
```
import numpy as np
import paddle
custom_ops = load(
        name="custom_jit_ops",
        sources=["ppdet/ext_op/rbox_iou_op.cc", "ppdet/ext_op/rbox_iou_op.cu"])

paddle.set_device('gpu:0')
paddle.disable_static()

rbox1 = np.random.rand(13000, 5)
rbox2 = np.random.rand(7, 5)

pd_rbox1 = paddle.to_tensor(rbox1)
pd_rbox2 = paddle.to_tensor(rbox2)

iou = custom_ops.rbox_iou(pd_rbox1, pd_rbox2)
print('iou', iou)
```

## 3. 单元测试
单元测试`test.py`文件中，通过对比python实现的结果和测试自定义op结果。

由于python计算细节与cpp计算细节略有区别，误差区间设置为0.02。
```
python3.7 test.py

# result
paddle time: 9.059906005859375e-06
iou is [13000, 7]
intersection  all sp_time 36.13534760475159
rbox time 36.680736780166626
(13000, 7)
sum of abs diff 0.0012188556971352176
rbox_iou OP compute right!
```


