#!/bin/bash
FILENAME=$1
# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer', 'infer']
MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})
function func_parser_key(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}
function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}
IFS=$'\n'
# The training params
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")
trainer_list=$(func_parser_value "${lines[14]}")

# MODE be one of ['lite_train_infer' 'whole_infer' 'whole_train_infer']
MODE=$2

if [ ${MODE} = "lite_train_infer" ];then
    mv ./dataset/coco/download_coco.py . && rm -rf ./dataset/coco/* && mv ./download_coco.py ./dataset/coco/
    # prepare lite train data
    wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/coco_ce.tar
    cd ./dataset/coco/ && tar -xvf coco_ce.tar && mv -u coco_ce/* .
    rm -rf coco_ce/
    # prepare lite train infer_img_dir
    cd ../../ && mkdir -p ./tests/demo/
    cp -u ./demo/road554.png ./demo/orange_71.jpg ./tests/demo/
    if [[ ${model_name} = "ppyolov2_r50vd_dcn_365e_coco" ]]; then
        wget -nc -P ./tests/weights/ https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams
    fi
elif [ ${MODE} = "whole_train_infer" ];then
    mv ./dataset/coco/download_coco.py . && rm -rf ./dataset/coco/* && mv ./download_coco.py ./dataset/coco/
    # prepare whole train data
    eval "${python} ./dataset/coco/download_coco.py"
    # prepare whole train infer_img_dir
    mkdir -p ./tests/demo/
    cp -u dataset/coco/val2017/* ./tests/demo/
elif [ ${MODE} = "whole_infer" ];then
    mv ./dataset/coco/download_coco.py . && rm -rf ./dataset/coco/* && mv ./download_coco.py ./dataset/coco/
    # prepare whole infer data
    wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/coco_ce.tar
    cd ./dataset/coco/ && tar -xvf coco_ce.tar && mv -u coco_ce/* .
    rm -rf coco_ce/
    # prepare whole infer infer_img_dir
    cd ../../ && mkdir -p ./tests/demo/
    cp -u dataset/coco/val2017/* ./tests/demo/
else
    mv ./dataset/coco/download_coco.py . && rm -rf ./dataset/coco/* && mv ./download_coco.py ./dataset/coco/
    # prepare infer data
    wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/coco_ce.tar
    cd ./dataset/coco/ && tar -xvf coco_ce.tar && mv -u coco_ce/* .
    rm -rf coco_ce/ && cd ../../
    if [[ ${model_name} = "yolov3_darknet53_270e_coco" ]]; then
        wget -nc -P ./tests/weights/ https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams
        wget -nc -P ./tests/weights/ https://paddledet.bj.bcebos.com/models/slim/yolov3_darknet_coco_qat.pdparams
        wget -nc -P ./tests/weights/ https://paddledet.bj.bcebos.com/models/slim/yolov3_darknet_prune_fpgm.pdparams
    elif [[ ${model_name} = "ppyolo_r50vd_dcn_1x_coco" ]]; then
        wget -nc -P ./tests/weights/ https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams
        wget -nc -P ./tests/weights/ https://paddledet.bj.bcebos.com/models/slim/ppyolo_r50vd_qat_pact.pdparams
        wget -nc -P ./tests/weights/ https://paddledet.bj.bcebos.com/models/slim/ppyolo_r50vd_prune_fpgm.pdparams
    elif [[ ${model_name} = "ppyolo_mbv3_large_coco" ]]; then
        wget -nc -P ./tests/weights/ https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_large_coco.pdparams
        wget -nc -P ./tests/weights/ https://paddledet.bj.bcebos.com/models/slim/ppyolo_mbv3_large_qat.pdparams
        wget -nc -P ./tests/weights/ https://paddledet.bj.bcebos.com/models/slim/ppyolo_mbv3_large_prune_fpgm.pdparams
    elif [[ ${model_name} = "ppyolov2_r50vd_dcn_365e_coco" ]]; then
        wget -nc -P ./tests/weights/ https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams
    elif [[ ${model_name} = "mask_rcnn_r50_fpn_1x_coco" ]]; then
        wget -nc -P ./tests/weights/ https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_fpn_1x_coco.pdparams
    elif [[ ${model_name} = "solov2_r50_fpn_1x_coco" ]]; then
        wget -nc -P ./tests/weights/ https://paddledet.bj.bcebos.com/models/solov2_r50_fpn_1x_coco.pdparams
    else
        sleep 2
    fi
fi
