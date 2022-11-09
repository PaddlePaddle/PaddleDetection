batch_size_list=(8)
precision_list=(fp16_o2) #fp32 tf32 fp16_o1 fp16_o2)
opt_level_list=(4) # 2 3 4)

for batch_size in ${batch_size_list[@]}; do
    for precision in ${precision_list[@]}; do
        for opt_level in ${opt_level_list[@]}; do
            bash run_yolov3_darknet53_270e_coco.sh ${batch_size} ${precision} ${opt_level}
            sleep 60
        done
    done
done


