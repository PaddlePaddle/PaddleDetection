#!/usr/bin/env bash

pip3.7 install -U pip Cython
pip3.7 install -r requirements.txt

mv ./dataset/coco/download_coco.py . && rm -rf ./dataset/coco/* && mv ./download_coco.py ./dataset/coco/
# prepare lite train data
wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/coco_benchmark.tar
cd ./dataset/coco/ && tar -xvf coco_benchmark.tar && mv -u coco_benchmark/* .
rm -rf coco_benchmark/

rm -rf ./dataset/mot/*
# prepare mot mini train data
wget -nc -P ./dataset/mot/ https://paddledet.bj.bcebos.com/data/mot_benchmark.tar
cd ./dataset/mot/ && tar -xvf mot_benchmark.tar && mv -u mot_benchmark/* .
rm -rf mot_benchmark/
