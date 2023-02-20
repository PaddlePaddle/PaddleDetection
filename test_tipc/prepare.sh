#!/bin/bash
source test_tipc/utils_func.sh

FILENAME=$1
# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer'
#                 'whole_train_whole_infer', 'whole_infer', 'klquant_whole_infer',
#                 'cpp_infer', 'serving_infer',  'lite_infer', 'paddle2onnx_infer']
MODE=$2

# parse params
dataline=$(cat ${FILENAME})
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")

if [ ${MODE} = "whole_train_whole_infer" ];then
    mv ./dataset/coco/download_coco.py . && rm -rf ./dataset/coco/* && mv ./download_coco.py ./dataset/coco/
    # prepare whole training data
    eval "${python} ./dataset/coco/download_coco.py"
elif [ ${MODE} = "cpp_infer" ];then
    # download coco lite data
    wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/tipc/coco_tipc.tar --no-check-certificate
    cd ./dataset/coco/ && tar -xvf coco_tipc.tar && mv -n coco_tipc/* .
    rm -rf coco_tipc/ && cd ../../
    # download wider_face lite data
    wget -nc -P ./dataset/wider_face/ https://paddledet.bj.bcebos.com/data/tipc/wider_tipc.tar --no-check-certificate
    cd ./dataset/wider_face/ && tar -xvf wider_tipc.tar && mv -n wider_tipc/* .
    rm -rf wider_tipc/ && cd ../../
    # download spine lite data
    wget -nc -P ./dataset/spine_coco/ https://paddledet.bj.bcebos.com/data/tipc/spine_tipc.tar --no-check-certificate
    cd ./dataset/spine_coco/ && tar -xvf spine_tipc.tar && mv -n spine_tipc/* .
    rm -rf spine_tipc/ && cd ../../
    if [[ ${model_name} =~ "s2anet" ]]; then
        cd ./ppdet/ext_op && eval "${python} setup.py install"
        cd ../../
    elif [[ ${model_name} =~ "tinypose" ]]; then
        wget -nc -P ./output_inference/ https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_320_pedestrian.tar --no-check-certificate
        cd ./output_inference/ && tar -xvf picodet_s_320_pedestrian.tar
        cd ../
    fi
    # download KL model
    if [[ ${model_name} = "picodet_lcnet_1_5x_416_coco_KL" ]]; then
        wget -nc -P ./output_inference/picodet_lcnet_1_5x_416_coco_KL/ https://bj.bcebos.com/v1/paddledet/data/tipc/models/picodet_lcnet_1_5x_416_coco_ptq.tar --no-check-certificate
        cd ./output_inference/picodet_lcnet_1_5x_416_coco_KL/ && tar -xvf picodet_lcnet_1_5x_416_coco_ptq.tar && mv -n picodet_lcnet_1_5x_416_coco_ptq/* .
        cd ../../
    elif [[ ${model_name} = "ppyoloe_crn_s_300e_coco_KL" ]]; then
        wget -nc -P ./output_inference/ppyoloe_crn_s_300e_coco_KL/ https://bj.bcebos.com/v1/paddledet/data/tipc/models/ppyoloe_crn_s_300e_coco_ptq.tar --no-check-certificate
        cd ./output_inference/ppyoloe_crn_s_300e_coco_KL/ && tar -xvf ppyoloe_crn_s_300e_coco_ptq.tar && mv -n ppyoloe_crn_s_300e_coco_ptq/* .
        cd ../../
    elif [[ ${model_name} = "ppyolo_mbv3_large_coco_KL" ]]; then
        wget -nc -P ./output_inference/ppyolo_mbv3_large_coco_KL/ https://bj.bcebos.com/v1/paddledet/data/tipc/models/ppyolo_mbv3_large_ptq.tar --no-check-certificate
        cd ./output_inference/ppyolo_mbv3_large_coco_KL/ && tar -xvf ppyolo_mbv3_large_ptq.tar && mv -n ppyolo_mbv3_large_ptq/* .
        cd ../../
    elif [[ ${model_name} = "mask_rcnn_r50_fpn_1x_coco_KL" ]]; then
        wget -nc -P ./output_inference/mask_rcnn_r50_fpn_1x_coco_KL/ https://bj.bcebos.com/v1/paddledet/data/tipc/models/mask_rcnn_r50_fpn_1x_coco_ptq.tar --no-check-certificate
        cd ./output_inference/mask_rcnn_r50_fpn_1x_coco_KL/ && tar -xvf mask_rcnn_r50_fpn_1x_coco_ptq.tar && mv -n mask_rcnn_r50_fpn_1x_coco_ptq/* .
        cd ../../
    elif [[ ${model_name} = "tinypose_128x96_KL" ]]; then
        wget -nc -P ./output_inference/tinypose_128x96_KL/ https://bj.bcebos.com/v1/paddledet/data/tipc/models/tinypose_128x96_ptq.tar --no-check-certificate
        cd ./output_inference/tinypose_128x96_KL/ && tar -xvf tinypose_128x96_ptq.tar && mv -n tinypose_128x96_ptq/* .
        cd ../../
    fi
    # download mot lite data
    wget -nc -P ./dataset/mot/ https://paddledet.bj.bcebos.com/data/tipc/mot_tipc.tar --no-check-certificate
    cd ./dataset/mot/ && tar -xvf mot_tipc.tar && mv -n mot_tipc/* .
    rm -rf mot_tipc/ && cd ../../

    opencv_dir=$(func_parser_value "${lines[15]}")
    # prepare opencv
    cd ./deploy/cpp
    if [ ${opencv_dir} = "default" ] || [ ${opencv_dir} = "null" ]; then
        if [ -d "deps/opencv-3.4.16_gcc8.2_ffmpeg/" ]; then
            echo "################### Opencv already exists, skip downloading. ###################"
        else
            mkdir -p $(pwd)/deps && cd $(pwd)/deps
            wget -c https://paddledet.bj.bcebos.com/data/opencv-3.4.16_gcc8.2_ffmpeg.tar.gz --no-check-certificate
            tar -xvf opencv-3.4.16_gcc8.2_ffmpeg.tar.gz && cd ../
            echo "################### Finish downloading opencv. ###################"
        fi
    fi
    cd ../../
elif [ ${MODE} = "benchmark_train" ];then
    pip install -U pip
    pip install Cython
    pip install -r requirements.txt
    if [[ ${model_name} =~ "higherhrnet" ]] || [[ ${model_name} =~ "hrnet" ]] || [[ ${model_name} =~ "tinypose" ]];then
        wget -nc -P ./dataset/ https://bj.bcebos.com/v1/paddledet/data/coco.tar --no-check-certificate
        cd ./dataset/ && tar -xf coco.tar
        ls ./coco/
        cd ../
    elif [[ ${model_name} =~ "ppyoloe_r_crn_s_3x_spine_coco" ]];then
        wget -nc -P ./dataset/spine_coco/ https://paddledet.bj.bcebos.com/data/tipc/spine_coco_tipc.tar --no-check-certificate
        cd ./dataset/spine_coco/ && tar -xvf spine_coco_tipc.tar && mv -n spine_coco_tipc/* .
        rm -rf spine_coco_tipc/ && cd ../../
        cd ./ppdet/ext_op && eval "${python} setup.py install"
        cd ../../
    else
        # prepare lite benchmark coco data
        wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/coco_benchmark.tar --no-check-certificate
        cd ./dataset/coco/ && tar -xf coco_benchmark.tar
        mv -u coco_benchmark/* ./
        ls ./
        cd ../../
        # prepare lite benchmark mot data
        wget -nc -P ./dataset/mot/ https://paddledet.bj.bcebos.com/data/mot_benchmark.tar --no-check-certificate
        cd ./dataset/mot/ && tar -xf mot_benchmark.tar
        mv -u mot_benchmark/* ./
        ls ./
        cd ../../
    fi
elif [ ${MODE} = "paddle2onnx_infer" ];then
    # install paddle2onnx
    ${python} -m pip install paddle2onnx
    ${python} -m pip install onnx onnxruntime
elif [ ${MODE} = "serving_infer" ];then
    unset https_proxy http_proxy
    # download coco lite data
    wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/tipc/coco_tipc.tar --no-check-certificate
    cd ./dataset/coco/ && tar -xvf coco_tipc.tar && mv -n coco_tipc/* .
    rm -rf coco_tipc/ && cd ../../
    # download KL model
    if [[ ${model_name} = "picodet_lcnet_1_5x_416_coco_KL" ]]; then
        wget -nc -P ./output_inference/picodet_lcnet_1_5x_416_coco_KL/ https://bj.bcebos.com/v1/paddledet/data/tipc/models/picodet_lcnet_1_5x_416_coco_ptq.tar --no-check-certificate
        cd ./output_inference/picodet_lcnet_1_5x_416_coco_KL/ && tar -xvf picodet_lcnet_1_5x_416_coco_ptq.tar && mv -n picodet_lcnet_1_5x_416_coco_ptq/* .
        cd ../../
        eval "${python} -m paddle_serving_client.convert --dirname output_inference/picodet_lcnet_1_5x_416_coco_KL/ --model_filename model.pdmodel --params_filename model.pdiparams --serving_server output_inference/picodet_lcnet_1_5x_416_coco_KL/serving_server --serving_client output_inference/picodet_lcnet_1_5x_416_coco_KL/serving_client"
    elif [[ ${model_name} = "ppyoloe_crn_s_300e_coco_KL" ]]; then
        wget -nc -P ./output_inference/ppyoloe_crn_s_300e_coco_KL/ https://bj.bcebos.com/v1/paddledet/data/tipc/models/ppyoloe_crn_s_300e_coco_ptq.tar --no-check-certificate
        cd ./output_inference/ppyoloe_crn_s_300e_coco_KL/ && tar -xvf ppyoloe_crn_s_300e_coco_ptq.tar && mv -n ppyoloe_crn_s_300e_coco_ptq/* .
        cd ../../
        eval "${python} -m paddle_serving_client.convert --dirname output_inference/ppyoloe_crn_s_300e_coco_KL/ --model_filename model.pdmodel --params_filename model.pdiparams --serving_server output_inference/ppyoloe_crn_s_300e_coco_KL/serving_server --serving_client output_inference/ppyoloe_crn_s_300e_coco_KL/serving_client"
    elif [[ ${model_name} = "ppyolo_mbv3_large_coco_KL" ]]; then
        wget -nc -P ./output_inference/ppyolo_mbv3_large_coco_KL/ https://bj.bcebos.com/v1/paddledet/data/tipc/models/ppyolo_mbv3_large_ptq.tar --no-check-certificate
        cd ./output_inference/ppyolo_mbv3_large_coco_KL/ && tar -xvf ppyolo_mbv3_large_ptq.tar && mv -n ppyolo_mbv3_large_ptq/* .
        cd ../../
        eval "${python} -m paddle_serving_client.convert --dirname output_inference/ppyolo_mbv3_large_coco_KL/ --model_filename model.pdmodel --params_filename model.pdiparams --serving_server output_inference/ppyolo_mbv3_large_coco_KL/serving_server --serving_client output_inference/ppyolo_mbv3_large_coco_KL/serving_client"
    elif [[ ${model_name} = "mask_rcnn_r50_fpn_1x_coco_KL" ]]; then
        wget -nc -P ./output_inference/mask_rcnn_r50_fpn_1x_coco_KL/ https://bj.bcebos.com/v1/paddledet/data/tipc/models/mask_rcnn_r50_fpn_1x_coco_ptq.tar --no-check-certificate
        cd ./output_inference/mask_rcnn_r50_fpn_1x_coco_KL/ && tar -xvf mask_rcnn_r50_fpn_1x_coco_ptq.tar && mv -n mask_rcnn_r50_fpn_1x_coco_ptq/* .
        cd ../../
        eval "${python} -m paddle_serving_client.convert --dirname output_inference/mask_rcnn_r50_fpn_1x_coco_KL/ --model_filename model.pdmodel --params_filename model.pdiparams --serving_server output_inference/mask_rcnn_r50_fpn_1x_coco_KL/serving_server --serving_client output_inference/mask_rcnn_r50_fpn_1x_coco_KL/serving_client"
    elif [[ ${model_name} = "tinypose_128x96_KL" ]]; then
        wget -nc -P ./output_inference/tinypose_128x96_KL/ https://bj.bcebos.com/v1/paddledet/data/tipc/models/tinypose_128x96_ptq.tar --no-check-certificate
        cd ./output_inference/tinypose_128x96_KL/ && tar -xvf tinypose_128x96_ptq.tar && mv -n tinypose_128x96_ptq/* .
        cd ../../
        eval "${python} -m paddle_serving_client.convert --dirname output_inference/tinypose_128x96_KL/ --model_filename model.pdmodel --params_filename model.pdiparams --serving_server output_inference/tinypose_128x96_KL/serving_server --serving_client output_inference/tinypose_128x96_KL/serving_client"
    fi
else
    # download coco lite data
    wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/tipc/coco_tipc.tar --no-check-certificate
    cd ./dataset/coco/ && tar -xvf coco_tipc.tar && mv -n coco_tipc/* .
    rm -rf coco_tipc/ && cd ../../
    # download wider_face lite data
    wget -nc -P ./dataset/wider_face/ https://paddledet.bj.bcebos.com/data/tipc/wider_tipc.tar --no-check-certificate
    cd ./dataset/wider_face/ && tar -xvf wider_tipc.tar && mv -n wider_tipc/* .
    rm -rf wider_tipc/ && cd ../../
    # download spine_coco lite data
    wget -nc -P ./dataset/spine_coco/ https://paddledet.bj.bcebos.com/data/tipc/spine_coco_tipc.tar --no-check-certificate
    cd ./dataset/spine_coco/ && tar -xvf spine_coco_tipc.tar && mv -n spine_coco_tipc/* .
    rm -rf spine_coco_tipc/ && cd ../../
    if [[ ${model_name} =~ "s2anet" ]]; then
        cd ./ppdet/ext_op && eval "${python} setup.py install"
        cd ../../
    elif [[ ${model_name} =~ "ppyoloe_r_crn_s_3x_spine_coco" ]]; then
        cd ./ppdet/ext_op && eval "${python} setup.py install"
        cd ../../
    elif [[ ${model_name} =~ "fcosr_x50_3x_spine_coco" ]]; then
        cd ./ppdet/ext_op && eval "${python} setup.py install"
        cd ../../
    fi
    # download mot lite data
    wget -nc -P ./dataset/mot/ https://paddledet.bj.bcebos.com/data/tipc/mot_tipc.tar --no-check-certificate
    cd ./dataset/mot/ && tar -xvf mot_tipc.tar && mv -n mot_tipc/* .
    rm -rf mot_tipc/ && cd ../../
fi
