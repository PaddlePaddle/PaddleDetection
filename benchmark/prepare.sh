#!/usr/bin/env bash

script_dir=$(dirname "${BASH_SOURCE[0]}")
chmod +x ${script_dir}/../paddle_log
${script_dir}/../paddle_log

pip install -U pip Cython
pip install -r requirements.txt

mv ./dataset/coco/download_coco.py . && rm -rf ./dataset/coco/* && mv ./download_coco.py ./dataset/coco/
# prepare lite train data
wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/coco_benchmark.tar
cd ./dataset/coco/ && tar -xvf coco_benchmark.tar && mv -u coco_benchmark/* .
rm -rf coco_benchmark/

cd ../../
rm -rf ./dataset/mot/*
# prepare mot mini train data
wget -nc -P ./dataset/mot/ https://paddledet.bj.bcebos.com/data/mot_benchmark.tar
cd ./dataset/mot/ && tar -xvf mot_benchmark.tar && mv -u mot_benchmark/* .
rm -rf mot_benchmark/
