#!/usr/bin/env bash

# download pretrain model
root_url="https://paddlemodels.bj.bcebos.com/object_detection"
yolov3_r34_voc="yolov3_r34_voc.tar"
pretrain_dir='./pretrain'

if [ ! -d ${pretrain_dir} ]; then
  mkdir ${pretrain_dir}
fi

cd ${pretrain_dir}

if [ ! -f ${yolov3_r34_voc} ]; then
    wget ${root_url}/${yolov3_r34_voc}
    tar xf ${yolov3_r34_voc}
fi
cd -

# enable GC strategy
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0

# for distillation
#-----------------
export CUDA_VISIBLE_DEVICES=0,1,2,3


# Fixing name conflicts in distillation
cd ${pretrain_dir}/yolov3_r34_voc
for files in $(ls teacher_*)
    do mv $files ${files#*_}
done
for files in $(ls *)
    do mv $files "teacher_"$files
done
cd -

python -u compress.py \
-c ../../configs/yolov3_mobilenet_v1_voc.yml \
-t yolov3_resnet34.yml \
-s yolov3_mobilenet_v1_yolov3_resnet34_distillation.yml \
-o YoloTrainFeed.batch_size=64 \
-d ../../dataset/voc \
--teacher_pretrained ./pretrain/yolov3_r34_voc \
> yolov3_distallation.log 2>&1 &
tailf yolov3_distallation.log
