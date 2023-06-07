#!/bin/bash
HOST_SPACE=${PWD}
echo ${HOST_SPACE}
WORK_SPACE=/data/local/tmp/test

# The first parameter represents the demo name
DEMO_NAME=image_classification_demo
if [ -n "$1" ]; then
  DEMO_NAME=$1
fi

# The second parameter represents the model name
MODEL_NAME=mobilenet_v1_fp32_224
if [ -n "$2" ]; then
  MODEL_NAME=$2
fi

# The third parameter indicates the name of the image to be tested
IMAGE_NAME=0001.jpg
if [ -n "$3" ]; then
  IMAGE_NAME=$3
fi

# The fourth parameter represents the ID of the device
ADB_DEVICE_NAME=
if [ -n "$4" ]; then
  ADB_DEVICE_NAME="-s $4"
fi

# Set the environment variables required during the running process
EXPORT_ENVIRONMENT_VARIABLES="export GLOG_v=5; export SUBGRAPH_ONLINE_MODE=true; export RKNPU_LOGLEVEL=5; export RKNN_LOG_LEVEL=5; ulimit -c unlimited; export VIV_VX_ENABLE_GRAPH_TRANSFORM=-pcq:1; export VIV_VX_SET_PER_CHANNEL_ENTROPY=100; export TIMVX_BATCHNORM_FUSION_MAX_ALLOWED_QUANT_SCALE_DEVIATION=300000; export VSI_NN_LOG_LEVEL=5;"

EXPORT_ENVIRONMENT_VARIABLES="${EXPORT_ENVIRONMENT_VARIABLES}export LD_LIBRARY_PATH=${WORK_SPACE}/lib:\$LD_LIBRARY_PATH;"

# Please install adb, and DON'T run this in the docker.
set -e
adb $ADB_DEVICE_NAME shell "rm -rf $WORK_SPACE"
adb $ADB_DEVICE_NAME shell "mkdir -p $WORK_SPACE"

# Upload the demo, librarys, model and test images to the device
adb $ADB_DEVICE_NAME push ${HOST_SPACE}/lib $WORK_SPACE
adb $ADB_DEVICE_NAME push ${HOST_SPACE}/${DEMO_NAME} $WORK_SPACE
adb $ADB_DEVICE_NAME push models $WORK_SPACE
adb $ADB_DEVICE_NAME push images $WORK_SPACE

# Execute the deployment demo
adb $ADB_DEVICE_NAME shell "cd $WORK_SPACE; ${EXPORT_ENVIRONMENT_VARIABLES} chmod +x ./${DEMO_NAME}; ./${DEMO_NAME} ./models/${MODEL_NAME} ./images/$IMAGE_NAME"
