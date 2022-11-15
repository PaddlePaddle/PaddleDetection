# 自定义OP编译
旋转框IOU计算OP是参考[自定义外部算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html) 。

## 1. 环境依赖
- Paddle >= 2.0.1
- gcc 8.2

## 2. 安装
```
python setup.py install
```

编译完成后即可使用，以下为`rbox_iou`的使用示例
```
# 引入自定义op
from ext_op import rbox_iou

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
可以通过执行单元测试来确认自定义算子功能的正确性，执行单元测试的示例如下所示：
```
python unittest/test_matched_rbox_iou.py
```
