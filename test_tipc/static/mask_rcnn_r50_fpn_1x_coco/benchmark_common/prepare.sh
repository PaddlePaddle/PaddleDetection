#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`
pip install Cython
pip install -r requirements.txt

################################# 准备训练数据 如:
wget -nc -P static/dataset/coco/ https://paddledet.bj.bcebos.com/data/coco_benchmark.tar
cd ./static/dataset/coco/ && tar -xf coco_benchmark.tar && mv -u coco_benchmark/* .
rm -rf coco_benchmark/ && cd ../../../
echo "*******prepare benchmark end***********"
