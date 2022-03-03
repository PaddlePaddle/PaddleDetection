# TinyPose OpenVINO Demo

This fold provides TinyPose inference code using
[Intel's OpenVINO Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html). Most of the implements in this fold are same as *demo_ncnn*.  
**Recommand** to use the xxx.tar.gz file to install instead of github method, [link](https://registrationcenter-download.intel.com/akdlm/irc_nas/18096/l_openvino_toolkit_p_2021.4.689.tgz).


## Install OpenVINO Toolkit

Go to [OpenVINO HomePage](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)

Download a suitable version and install.

Follow the official Get Started Guides: https://docs.openvinotoolkit.org/latest/get_started_guides.html

## Set the Environment Variables

### Windows:

Run this command in cmd. (Every time before using OpenVINO)
```cmd
<INSTSLL_DIR>\openvino_2021\bin\setupvars.bat
```

Or set the system environment variables once for all:

Name                  |Value
:--------------------:|:--------:
INTEL_OPENVINO_DIR | <INSTSLL_DIR>\openvino_2021
INTEL_CVSDK_DIR | %INTEL_OPENVINO_DIR%
InferenceEngine_DIR | %INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\share
HDDL_INSTALL_DIR | %INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\external\hddl
ngraph_DIR | %INTEL_OPENVINO_DIR%\deployment_tools\ngraph\cmake

And add this to ```Path```
```
%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\Debug;%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\Release;%HDDL_INSTALL_DIR%\bin;%INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\external\tbb\bin;%INTEL_OPENVINO_DIR%\deployment_tools\ngraph\lib
```

### Linux

Run this command in shell. (Every time before using OpenVINO)

```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
```

Or edit .bashrc

```shell
vi ~/.bashrc
```

Add this line to the end of the file

```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
```

## Convert model

   Convert to OpenVINO

   ``` shell
   cd <INSTSLL_DIR>/openvino_2021/deployment_tools/model_optimizer
   ```

   Install requirements for convert tool

   ```shell
   cd ./install_prerequisites
   sudo install_prerequisites_onnx.sh

   ```

   Then convert model. Notice: mean_values and scale_values should be the same with your training settings in YAML config file.
   ```shell
   python3 mo_onnx.py --input_model <ONNX_MODEL> --mean_values [103.53,116.28,123.675] --scale_values [57.375,57.12,58.395]
   ```

## Build

### Windows

```cmd
<OPENVINO_INSTSLL_DIR>\openvino_2021\bin\setupvars.bat
mkdir -p build
cd build
cmake ..
msbuild tinypose_demo.vcxproj /p:configuration=release /p:platform=x64
```

### Linux
```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
mkdir build
cd build
cmake ..
make
```


## Run demo
Download PicoDet openvino model [PicoDet openvino model download link](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_m_416_openvino.zip).
Download TinyPose openvino model [TinyPose openvino model download link](https://paddledet.bj.bcebos.com/deploy/third_engine/tinypose_256_openvino.zip).

move picodet and tinypose openvino model files to the demo's weight folder. 

### Edit file
```
step1:
main.cpp
#define image_size 416
...
  cv::Mat image(256, 192, CV_8UC3, cv::Scalar(1, 1, 1));
  std::vector<float> center = {128, 96};
  std::vector<float> scale = {256, 192};
...
  auto detector = PicoDet("../weight/picodet_m_416.xml");
  auto kpts_detector = new KeyPointDetector("../weight/tinypose256.xml", -1, 256, 192);
...
step2:
picodet_openvino.h
#define image_size 416
```

### Run

Run command:
``` shell
./tinypose_demo [mode] [image_file]
```
|  param   | detail  |
|  ----  | ----  |
| --mode  | input mode，0:camera；1:image；2:video；3:benchmark |
| --image_file  | input image path |

#### Webcam

```shell
tinypose_demo 0 0
```

#### Inference images

```shell
tinypose_demo 1 IMAGE_FOLDER/*.jpg
```

#### Inference video

```shell
tinypose_demo 2 VIDEO_PATH
```

### Benchmark

```shell
tinypose_demo 3 0
```

Plateform: Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz x 24(核)
Model: [Tinypose256_Openvino](https://paddledet.bj.bcebos.com/deploy/third_engine/tinypose_256_openvino.zip)

| param         | Min   | Max   | Avg   |
| ------------- | ----- | ----- | ----- |
| infer time(s) | 0.018 | 0.062 | 0.028 |

