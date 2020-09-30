# PaddleDetection安卓端demo

### 下载试用
可通过[下载链接](https://paddlemodels.bj.bcebos.com/object_detection/lite/paddledetection_app.apk)直接下载，或直接使用手机浏览器扫描二维码下载安装：

<div align="center">
  <img src="demo/ppdet_app.png" width='400'/>
</div>

### 环境搭建与代码运行
- 安装最新版本的Android Studio，可以从https://developer.android.com/studio 下载。本demo使用是4.0版本Android Studio编写。
- 下载NDK 20 以上版本，NDK 20版本以上均可以编译成功。可以用以下方式安装和测试NDK编译环境：点击 File -> New ->New Project，新建  "Native C++" project。
- 导入项目：点击 File->New->Import Project...， 跟随Android Studio的引导导入项目即可。
- 首先打开`app/build.gradle`文件，运行`downloadAndExtractArchives`函数，完成PaddleLite预测库与模型的下载与压缩。
- 连接并选择设备，编译app并运行。

### 效果展示
<div align="center">
  <img src="demo/ppdet_app_home.jpg" height="500px" ><img src="demo/ppdet_app_photo.jpg" height="500px" ><img src="demo/ppdet_app_camera.jpg" height="500px" >
</div>

### 获取更多支持
前往[端计算模型生成平台EasyEdge](https://ai.baidu.com/easyedge/app/open_source_demo?referrerUrl=paddlelite)，获得更多开发支持
