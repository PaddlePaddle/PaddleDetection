# PicoDet OpenVINO Demo

This fold provides PicoDet inference code using
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
msbuild picodet_demo.vcxproj /p:configuration=release /p:platform=x64
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

move picodet openvino model files to the demo's weight folder. Then run these commands:

### Edit file
```
step1:
main.cpp
#define image_size 416
...
auto detector = PicoDet("../weight/picodet_m_416.xml");
...
step2:
picodet_openvino.h
#define image_size 416
```

### Webcam

```shell
picodet_demo 0 0
```

### Inference images

```shell
picodet_demo 1 IMAGE_FOLDER/*.jpg
```

### Inference video

```shell
picodet_demo 2 VIDEO_PATH
```

### Benchmark

```shell
picodet_demo 3 0
```
