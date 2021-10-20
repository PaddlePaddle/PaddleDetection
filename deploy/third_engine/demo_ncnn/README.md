# PicoDet NCNN Demo

This project provides PicoDet image inference, webcam inference and benchmark using
[Tencent's NCNN framework](https://github.com/Tencent/ncnn).

# How to build

## Windows
### Step1.
Download and Install Visual Studio from https://visualstudio.microsoft.com/vs/community/

### Step2.
Download and install OpenCV from https://github.com/opencv/opencv/releases

### Step3(Optional).
Download and install Vulkan SDK from https://vulkan.lunarg.com/sdk/home

### Step4.
Clone NCNN repository

``` shell script
git clone --recursive https://github.com/Tencent/ncnn.git
```
Build NCNN following this tutorial: [Build for Windows x64 using VS2017](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)

### Step5.

Add `ncnn_DIR` = `YOUR_NCNN_PATH/build/install/lib/cmake/ncnn` to system environment variables.

Build project: Open x64 Native Tools Command Prompt for VS 2019 or 2017

``` cmd
cd <this-folder>
mkdir -p build
cd build
cmake ..
msbuild picodet_demo.vcxproj /p:configuration=release /p:platform=x64
```

## Linux

### Step1.
Build and install OpenCV from https://github.com/opencv/opencv

### Step2(Optional).
Download Vulkan SDK from https://vulkan.lunarg.com/sdk/home

### Step3.
Clone NCNN repository

``` shell script
git clone --recursive https://github.com/Tencent/ncnn.git
```

Build NCNN following this tutorial: [Build for Linux / NVIDIA Jetson / Raspberry Pi](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux)

### Step4.

Set environment variables. Run:

``` shell script
export ncnn_DIR=YOUR_NCNN_PATH/build/install/lib/cmake/ncnn
```

Build project

``` shell script
cd <this-folder>
mkdir build
cd build
cmake ..
make
```

# Run demo

Download PicoDet ncnn model.
* [PicoDet ncnn model download link](https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_m_416_ncnn.zip)


## Webcam

```shell script
picodet_demo 0 0
```

## Inference images

```shell script
picodet_demo 1 IMAGE_FOLDER/*.jpg
```

## Inference video

```shell script
picodet_demo 2 VIDEO_PATH
```

## Benchmark

```shell script
picodet_demo 3 0

result: picodet  min = 17.74  max = 22.71  avg = 18.16
```

****

Notice:

If benchmark speed is slow, try to limit omp thread num.

Linux:

```shell script
export OMP_THREAD_LIMIT=4
```
