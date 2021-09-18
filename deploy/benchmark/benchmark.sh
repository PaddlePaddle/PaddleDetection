# All rights `PaddleDetection` reserved
#!/bin/bash
model_dir=$1
model_name=$2

export img_dir="demo"
export log_path="output_pipeline"


echo "model_dir : ${model_dir}"
echo "img_dir: ${img_dir}"

# TODO: support batch size>1
for use_mkldnn in "True" "False"; do
    for threads in "1" "6"; do
            echo "${model_name}  ${model_dir}, use_mkldnn: ${use_mkldnn}   threads: ${threads}"
            python deploy/python/infer.py \
		 --model_dir=${model_dir} \
		 --run_benchmark True \
		 --enable_mkldnn=${use_mkldnn} \
		 --device=CPU \
		 --cpu_threads=${threads} \
		 --image_dir=${img_dir}  2>&1 | tee ${log_path}/${model_name}_cpu_usemkldnn_${use_mkldnn}_cputhreads_${threads}_bs1_infer.log
    done
done

for run_mode in "fluid" "trt_fp32" "trt_fp16"; do
    echo "${model_name}  ${model_dir}, run_mode: ${run_mode}"
    python deploy/python/infer.py \
	--model_dir=${model_dir} \
	--run_benchmark=True \
	--device=GPU \
	--run_mode=${run_mode} \
	--image_dir=${img_dir}  2>&1 | tee ${log_path}/${model_name}_gpu_runmode_${run_mode}_bs1_infer.log
done

