# test with:
# docker run -it -v `pwd`:/WORKIR -v ~/.cache/paddle/weights:/root/.cache/paddle/weights -v /THE_DIR_WITH_THE_FILES:/DS paddledet python3 tools/infer.py -c configs/pedestrian/pedestrian_yolov3_darknet.yml -o use_gpu=true weights=https://paddledet.bj.bcebos.com/models/pedestrian_yolov3_darknet.pdparams --infer_img=/DS/output2.jpg


FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

WORKDIR /WORKIR

RUN apt-get update && \
      apt-get install -y apt-utils && \
      apt-get install -y \
            wget \
            curl \
            git \
            vim \
            ffmpeg libsm6 libxext6 \
            python3-pip && \
      rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN python3 -m pip install --upgrade pip && pip3 install paddlepaddle-gpu==2.1.0.post101 -f https://paddlepaddle.org.cn/whl/mkl/stable.html
RUN pip3 install pandas paddledet==2.1.0 -i https://mirror.baidu.com/pypi/simple
