# All rights `PaddleDetection` reserved
#!/bin/bash
model_dir=$1
model_name=$2

export img_dir="demo"
export log_path="output_pipeline"


echo "model_dir : ${model_dir}"
echo "img_dir: ${img_dir}"

# TODO: support batch size>1
for run_mode in "trt_int8"; do
    echo "${model_name}  ${model_dir}, run_mode: ${run_mode}"
    python deploy/python/infer.py \
	--model_dir=${model_dir} \
	--run_benchmark=True \
	--device=GPU \
	--run_mode=${run_mode} \
	--image_dir=${img_dir}  2>&1 | tee ${log_path}/${model_name}_gpu_runmode_${run_mode}_bs1_infer.log
done

