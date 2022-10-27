batch_size_list=(6)
precision_list=(fp16_o2) # fp16_o1 tf32 fp32)
opt_level_list=(5)
data_format=(NCHW)
# 1: base 2: num= 8 3:auto_tune 4: workspace  5:bn  6:old_prof
for batch_size in ${batch_size_list[@]}; do
    for precision in ${precision_list[@]}; do
        for opt_level in ${opt_level_list[@]}; do
            bash run_fairmot.sh ${batch_size} ${precision} ${opt_level} ${data_format}
            sleep 60
        done
    done
done


