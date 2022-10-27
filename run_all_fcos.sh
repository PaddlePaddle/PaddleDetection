#!/bin/bash

batch_size_list=(8)
precision_list=(fp16_o1)
opt_level_list=(0)

for batch_size in ${batch_size_list[@]}; do
    for precision in ${precision_list[@]}; do
        for opt_level in ${opt_level_list[@]}; do
            bash run_fcos_r50_fpn_1x_coco.sh ${batch_size} ${precision} ${opt_level}
            sleep 60
        done
    done
done

