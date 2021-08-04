#!/bin/bash
FILENAME=$1

# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer']
MODE=$2

# prepare dataset 
if [ ${MODE} = "lite_train_infer" ];then
    # pretrain lite train data
    cd dataset/coco
    wget https://paddledet.bj.bcebos.com/data/coco_ce.tar
    tar -xvf coco_ce.tar
    mv coco_ce/* .
    rm -rf coco_ce*
else
    # pretrain lite train data
    cd dataset/coco
    wget https://paddledet.bj.bcebos.com/data/coco_ce.tar
    tar -xvf coco_ce.tar
    mv coco_ce/* .
    rm -rf coco_ce*
fi

