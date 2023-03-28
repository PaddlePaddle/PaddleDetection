English | [简体中文](README_CN.md)
# Target Detection PicoDet Android Demo Tutorial

Real-time target detection on Android. This Demo is simple to use for everyone. For example, you can run your own trained model in the Demo.

## Prepare the Environment

1. Install Android Studio in your local environment. Refer to [Android Studio Official Website](https://developer.android.com/studio) for detailed tutorial.
2. Prepare an Android phone and turn on the USB debug mode: `Settings -> Find developer options -> Open developer options and USB debug mode`

## Deployment Steps

1. The target detection PicoDet Demo is located in the `fastdeploy/examples/vision/detection/paddledetection/android` directory
2. Open paddledetection/android project with Android Studio
3. Connect the phone to the computer, turn on USB debug mode and file transfer mode, and connect your phone to Android Studio (allow the phone to install software from USB)

<p align="center">
<img width="1440" alt="image" src="https://user-images.githubusercontent.com/31974251/203257262-71b908ab-bb2b-47d3-9efb-67631687b774.png">
</p>

> **Attention：**
>> If you encounter an NDK configuration error during import, compilation or running, open ` File > Project Structure > SDK Location` and change the path of SDK configured by the `Andriod SDK location`.

4. Click the Run button to automatically compile the APP and install it to the phone. (The process will automatically download the pre-compiled FastDeploy Android library and model files. Internet is required). 
The final effect is as follows. Figure 1: Install the APP on the phone; Figure 2: The effect when opening the APP. It will automatically recognize and mark the objects in the image; Figure 3: APP setting option. Click setting in the upper right corner and modify your options.

| APP Icon | APP Effect | APP Settings
  | ---     | --- | --- |
  | <img width="300" height="500" alt="image" src="https://user-images.githubusercontent.com/31974251/203268599-c94018d8-3683-490a-a5c7-a8136a4fa284.jpg">  | <img width="300" height="500" alt="image" src="https://user-images.githubusercontent.com/31974251/203261763-a7513df7-e0ab-42e5-ad50-79ed7e8c8cd2.gif"> | <img width="300" height="500" alt="image" src="https://user-images.githubusercontent.com/31974251/197332983-afbfa6d5-4a3b-4c54-a528-4a3e58441be1.jpg"> |  


### PicoDet Java API Description 
- Model initialized API: The initialized API contains two ways: Firstly, initialize directly through the constructor. Secondly, initialize at the appropriate program node by calling the init function. PicoDet initialization parameters are as follows.
  - modelFile: String. Model file path in paddle format, such as model.pdmodel
  - paramFile: String. Parameter file path in paddle format, such as model.pdiparams  
  - configFile: String. Preprocessing file for model inference, such as infer_cfg.yml  
  - labelFile: String. This optional parameter indicates the path of the label file and is used for visualization, such as coco_label_list.txt, each line containing one label
  - option: RuntimeOption. Optional parameter for model initialization. Default runtime options if not passing the parameter.

```java
// Constructor: constructor w/o label file
public PicoDet(); // An empty constructor, which can be initialized by calling init
public PicoDet(String modelFile, String paramsFile, String configFile);
public PicoDet(String modelFile, String paramsFile, String configFile, String labelFile);
public PicoDet(String modelFile, String paramsFile, String configFile, RuntimeOption option);
public PicoDet(String modelFile, String paramsFile, String configFile, String labelFile, RuntimeOption option);
// Call init manually for initialization: call init manually w/o label file
public boolean init(String modelFile, String paramsFile, String configFile, RuntimeOption option);
public boolean init(String modelFile, String paramsFile, String configFile, String labelFile, RuntimeOption option);
```  
- Model Prediction API: The Model Prediction API contains an API for direct prediction and an API for visualization. In direct prediction, we do not save the image and render the result on Bitmap. Instead, we merely predict the inference result. For prediction and visualization, the results are both predicted and visualized, the visualized images are saved to the specified path, and the visualized results are rendered in Bitmap (Now Bitmap in ARGB8888 format is supported). Afterward, the Bitmap can be displayed on the camera.
```java
// Direct prediction: No image saving and no result rendering to Bitmap 
public DetectionResult predict(Bitmap ARGB8888Bitmap)；
// Prediction and visualization: Predict and visualize the results, save the visualized image to the specified path, and render the visualized results on Bitmap 
public DetectionResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float scoreThreshold);
public DetectionResult predict(Bitmap ARGB8888Bitmap, boolean rendering, float scoreThreshold); // Render without saving images
```
- Model resource release API: Call release() API to release model resources. Return true for successful release and false for failure; call initialized() to determine whether the model was initialized successfully, with true indicating successful initialization and false indicating failure. 
```java
public boolean release(); // Release native resources   
public boolean initialized(); // Check if the initialization is successful
```  

- RuntimeOption settings  
```java  
public void enableLiteFp16(); // Enable fp16 accuracy inference
public void disableLiteFP16(); // Disable fp16 accuracy inference
public void setCpuThreadNum(int threadNum); // Set thread numbers
public void setLitePowerMode(LitePowerMode mode);  // Set power mode
public void setLitePowerMode(String modeStr);  // Set power mode through character string
```

- Model DetectionResult
```java
public class DetectionResult {
  public float[][] mBoxes; // [n,4] Detection box (x1,y1,x2,y2)
  public float[] mScores;  // [n]   Score (confidence, probability)
  public int[] mLabelIds;  // [n]   Classification ID
  public boolean initialized(); // Whether the result is valid 
}
```  
Refer to [api/vision_results/detection_result.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/detection_result.md) for C++/Python DetectionResult

- Model Calling Example 1: Using Constructor and the default RuntimeOption
```java  
import java.nio.ByteBuffer;
import android.graphics.Bitmap;
import android.opengl.GLES20;

import com.baidu.paddle.fastdeploy.vision.DetectionResult;
import com.baidu.paddle.fastdeploy.vision.detection.PicoDet;

// Initialize the model
PicoDet model = new PicoDet("picodet_s_320_coco_lcnet/model.pdmodel",
                            "picodet_s_320_coco_lcnet/model.pdiparams",
                            "picodet_s_320_coco_lcnet/infer_cfg.yml");

// Read the image: The following is merely the pseudo code to read Bitmap
ByteBuffer pixelBuffer = ByteBuffer.allocate(width * height * 4);
GLES20.glReadPixels(0, 0, width, height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pixelBuffer);
Bitmap ARGB8888ImageBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
ARGB8888ImageBitmap.copyPixelsFromBuffer(pixelBuffer);

// Model inference
DetectionResult result = model.predict(ARGB8888ImageBitmap);  

// Release model resources   
model.release();
```  

- Model calling example 2: Manually call init at the appropriate program node and self-define RuntimeOption
```java  
// import is as the above...
import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.LitePowerMode;
import com.baidu.paddle.fastdeploy.vision.DetectionResult;
import com.baidu.paddle.fastdeploy.vision.detection.PicoDet;
// Create an empty model
PicoDet model = new PicoDet();  
// Model path
String modelFile = "picodet_s_320_coco_lcnet/model.pdmodel";
String paramFile = "picodet_s_320_coco_lcnet/model.pdiparams";
String configFile = "picodet_s_320_coco_lcnet/infer_cfg.yml";
// Specify RuntimeOption
RuntimeOption option = new RuntimeOption();
option.setCpuThreadNum(2);
option.setLitePowerMode(LitePowerMode.LITE_POWER_HIGH);
option.enableLiteFp16();
// Use init function for initialization 
model.init(modelFile, paramFile, configFile, option);
// Bitmap reading, model prediction, and resource release are as above...
```
Refer to [DetectionMainActivity](./app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/detection/DetectionMainActivity.java) for more information.

## Replace FastDeploy SDK and Models  
It’s simple to replace the FastDeploy prediction library and models. The prediction library is located at `app/libs/fastdeploy-android-sdk-xxx.aar`, where `xxx`  represents the version of your prediction library. The models are located at `app/src/main/assets/models/picodet_s_320_coco_lcnet`.
- Replace the FastDeploy Android SDK: Download or compile the latest FastDeploy Android SDK, unzip and place it in the `app/libs` directory; For detailed configuration, refer to  
     - [FastDeploy Java SDK  in Android](https://github.com/PaddlePaddle/FastDeploy/blob/develop/java/android/)

- Steps to replace PicoDet models:  
  - Put your PicoDet model in `app/src/main/assets/models`; 
  - Modify the default value of the model path in `app/src/main/res/values/strings.xml`. For example:  
```xml
<!-- Change this path to your model, such as models/picodet_l_320_coco_lcnet -->
<string name="DETECTION_MODEL_DIR_DEFAULT">models/picodet_s_320_coco_lcnet</string>  
<string name="DETECTION_LABEL_PATH_DEFAULT">labels/coco_label_list.txt</string>
```  

## More Reference Documents
For more FastDeploy Java API documentes and how to access FastDeploy C++ API via JNI, refer to: 
- [FastDeploy Java SDK in Android](https://github.com/PaddlePaddle/FastDeploy/blob/develop/java/android/)
- [FastDeploy C++ SDK in Android](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/faq/use_cpp_sdk_on_android.md)  
