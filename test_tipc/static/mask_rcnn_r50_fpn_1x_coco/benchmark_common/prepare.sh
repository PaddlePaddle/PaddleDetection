#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
echo `pip --version`
pip install Cython -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install paddlepaddle-gpu==2.2.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

################################# 准备训练数据 如:
wget -nc -P static/data/coco/ https://paddledet.bj.bcebos.com/data/coco_benchmark.tar
cd ./static/data/coco/ && tar -xf coco_benchmark.tar && mv -u coco_benchmark/* .
rm -rf coco_benchmark/ && cd ../../../
echo "*******prepare benchmark end***********"
