python3.7 deploy/python/infer.py \
    --model_dir=output_inference/yolov3_darknet53_270e_coco/ \
    --image_dir demo/ \
    --run_benchmark True \
    --use_gpu True \
    --enable_mkldnn True \
    --cpu_threads 6
