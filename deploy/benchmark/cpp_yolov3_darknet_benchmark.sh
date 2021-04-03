./deploy/cpp/build/main \
    --model_dir=output_inference/yolov3_darknet53_270e_coco \
    --image_dir=demo/ \
    --use_gpu True \
    --use_mkldnn=true \
    --cpu_threads=6