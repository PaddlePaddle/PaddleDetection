# 自定义OP编译
旋转框IOU计算OP是参考[自定义外部算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_custom_op.html) 。

## 1. 环境依赖
- Paddle >= 2.0.1
- gcc 8.2

## 2. 安装
```
python3.7 setup.py install
```

按照如下方式使用
```
# 引入自定义op
from rbox_iou_ops import rbox_iou

paddle.set_device('gpu:0')
paddle.disable_static()

rbox1 = np.random.rand(13000, 5)
rbox2 = np.random.rand(7, 5)

pd_rbox1 = paddle.to_tensor(rbox1)
pd_rbox2 = paddle.to_tensor(rbox2)

iou = rbox_iou(pd_rbox1, pd_rbox2)
print('iou', iou)
```

## 3. 单元测试
单元测试`test.py`文件中，通过对比python实现的结果和测试自定义op结果。

由于python计算细节与cpp计算细节略有区别，误差区间设置为0.02。
```
python3.7 test.py
```
提示`rbox_iou OP compute right!`说明OP测试通过。
