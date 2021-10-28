#!/bin/bash
source test_tipc/utils_func.sh

FILENAME=$1
# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer'
#                 'whole_train_whole_infer', 'whole_infer', 'klquant_whole_infer',
#                 'cpp_infer', 'serving_infer',  'lite_infer']
MODE=$2

# parse params
dataline=$(cat ${FILENAME})
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")

if [ ${MODE} = "lite_train_lite_infer" ] || [ ${MODE} = "lite_train_whole_infer" ];then
    mv ./dataset/coco/download_coco.py . && rm -rf ./dataset/coco/* && mv ./download_coco.py ./dataset/coco/
    # download data
    wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/coco_tipc.tar
    cd ./dataset/coco/ && tar -xvf coco_tipc.tar && mv -u coco_tipc/* .
    rm -rf coco_tipc/ && cd ../../
elif [ ${MODE} = "whole_train_whole_infer" ];then
    mv ./dataset/coco/download_coco.py . && rm -rf ./dataset/coco/* && mv ./download_coco.py ./dataset/coco/
    # prepare whole training data
    eval "${python} ./dataset/coco/download_coco.py"
elif [ ${MODE} = "whole_infer" ] || [ ${MODE} = "klquant_whole_infer" ];then
    mv ./dataset/coco/download_coco.py . && rm -rf ./dataset/coco/* && mv ./download_coco.py ./dataset/coco/
    # download data
    wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/coco_tipc.tar
    cd ./dataset/coco/ && tar -xvf coco_tipc.tar && mv -u coco_tipc/* .
    rm -rf coco_tipc/ && cd ../../
elif [ ${MODE} = "cpp_infer" ];then
    opencv_dir=$(func_parser_value "${lines[15]}")
    # prepare opencv
    cd ./deploy/cpp
    if [ ${opencv_dir} = "default" ] || [ ${opencv_dir} = "null" ]; then
        if [ -d "deps/opencv-3.4.16_gcc8.2_ffmpeg/" ]; then
            echo "################### Opencv already exists, skip downloading. ###################"
        else
            mkdir -p $(pwd)/deps && cd $(pwd)/deps
            wget -c https://paddledet.bj.bcebos.com/data/opencv-3.4.16_gcc8.2_ffmpeg.tar.gz
            tar -xvf opencv-3.4.16_gcc8.2_ffmpeg.tar.gz && cd ../
            echo "################### Finish downloading opencv. ###################"
        fi
    fi
    cd ../../
else
    mv ./dataset/coco/download_coco.py . && rm -rf ./dataset/coco/* && mv ./download_coco.py ./dataset/coco/
    # prepare infer data
    wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/coco_tipc.tar
    cd ./dataset/coco/ && tar -xvf coco_tipc.tar && mv -u coco_tipc/* .
    rm -rf coco_tipc/ && cd ../../
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
        sleep 1
    fi
fi
