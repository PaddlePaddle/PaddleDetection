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
    wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/tipc/coco_tipc.tar
    cd ./dataset/coco/ && tar -xvf coco_tipc.tar && mv -n coco_tipc/* .
    rm -rf coco_tipc/ && cd ../../
    # download wider_face lite data
    wget -nc -P ./dataset/wider_face/ https://paddledet.bj.bcebos.com/data/tipc/wider_tipc.tar
    cd ./dataset/wider_face/ && tar -xvf wider_tipc.tar && mv -n wider_tipc/* .
    rm -rf wider_tipc/ && cd ../../
    # download spine lite data
    wget -nc -P ./dataset/spine_coco/ https://paddledet.bj.bcebos.com/data/tipc/spine_tipc.tar
    cd ./dataset/spine_coco/ && tar -xvf spine_tipc.tar && mv -n spine_tipc/* .
    rm -rf spine_tipc/ && cd ../../
    if [[ ${model_name} =~ "s2anet" ]]; then
        cd ./ppdet/ext_op && eval "${python} setup.py install"
        cd ../../
    fi
    # download mot lite data
    wget -nc -P ./dataset/mot/ https://paddledet.bj.bcebos.com/data/tipc/mot_tipc.tar
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
            wget -c https://paddledet.bj.bcebos.com/data/opencv-3.4.16_gcc8.2_ffmpeg.tar.gz
            tar -xvf opencv-3.4.16_gcc8.2_ffmpeg.tar.gz && cd ../
            echo "################### Finish downloading opencv. ###################"
        fi
    fi
    cd ../../
elif [ ${MODE} = "benchmark_train" ];then
    pip install -U pip Cython
    pip install -r requirements.txt
    # prepare lite benchmark coco data
    wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/coco_benchmark.tar
    cd ./dataset/coco/ && tar -xvf coco_benchmark.tar 
    mv -u coco_benchmark/* ./
    ls ./
    cd ../../
    # prepare lite benchmark mot data
    wget -nc -P ./dataset/mot/ https://paddledet.bj.bcebos.com/data/mot_benchmark.tar
    cd ./dataset/mot/ && tar -xvf mot_benchmark.tar
    mv -u mot_benchmark/* ./
    ls ./
    cd ../../
elif [ ${MODE} = "paddle2onnx_infer" ];then
    # set paddle2onnx_infer enve
    ${python} -m pip install install paddle2onnx
    ${python} -m pip install onnxruntime==1.10.0
elif [ ${MODE} = "serving_infer" ];then
    git clone https://github.com/PaddlePaddle/Serving
    bash Serving/tools/paddle_env_install.sh
    cd Serving
    pip install -r python/requirements.txt
    cd ..
    pip install paddle-serving-client==0.8.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install paddle-serving-app==0.8.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install paddle-serving-server-gpu==0.8.3.post101 -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install paddlepaddle-gpu==2.2.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
else
    # download coco lite data
    wget -nc -P ./dataset/coco/ https://paddledet.bj.bcebos.com/data/tipc/coco_tipc.tar
    cd ./dataset/coco/ && tar -xvf coco_tipc.tar && mv -n coco_tipc/* .
    rm -rf coco_tipc/ && cd ../../
    # download wider_face lite data
    wget -nc -P ./dataset/wider_face/ https://paddledet.bj.bcebos.com/data/tipc/wider_tipc.tar
    cd ./dataset/wider_face/ && tar -xvf wider_tipc.tar && mv -n wider_tipc/* .
    rm -rf wider_tipc/ && cd ../../
    # download spine_coco lite data
    wget -nc -P ./dataset/spine_coco/ https://paddledet.bj.bcebos.com/data/tipc/spine_tipc.tar
    cd ./dataset/spine_coco/ && tar -xvf spine_tipc.tar && mv -n spine_tipc/* .
    rm -rf spine_tipc/ && cd ../../
    if [[ ${model_name} =~ "s2anet" ]]; then
        cd ./ppdet/ext_op && eval "${python} setup.py install"
        cd ../../
    fi
    # download mot lite data
    wget -nc -P ./dataset/mot/ https://paddledet.bj.bcebos.com/data/tipc/mot_tipc.tar
    cd ./dataset/mot/ && tar -xvf mot_tipc.tar && mv -n mot_tipc/* .
    rm -rf mot_tipc/ && cd ../../
fi
