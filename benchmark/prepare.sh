#!/usr/bin/env bash

pip3.7 install -U pip Cython
pip3.7 install -r requirements.txt
mv ./dataset/coco/download_coco.py . && rm -rf ./dataset/coco/* && mv ./download_coco.py ./dataset/coco/
# prepare lite train data
wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/coco_benchmark.tar
cd ./dataset/coco/ && tar -xvf coco_benchmark.tar && mv -u coco_benchmark/* .
rm -rf coco_benchmark/
