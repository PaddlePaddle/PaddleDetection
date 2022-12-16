[简体中文](README_cn.md) | 繁體中文 | [English](README_en.md)

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/48054808/160532560-34cf7a1f-d950-435e-90d2-4b0a679e5119.png" align="middle" width = "800" />
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleDetection/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleDetection?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleDetection/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleDetection?color=ccf"></a>
</p>
</div>

## 🌈簡介

PaddleDetection是一個基於PaddlePaddle的目標檢測套件，在提供豐富的模型和測試基準的同時，也注重將此技術應用於產業之中，通過打造企業級特色模型與工具、公開業界範例等方式，幫助開發者從資料前處理、選擇模型、訓練到部署，一步步將專案開發。

主要模型效果範例如下（點擊標題可跳轉）：

|                                                  [**物件辨識**](#pp-yoloe-高精度物件辨識模型)                                                  |                                                [**小型物件辨識**](#pp-yoloe-sod-高精度小型物件辨識模型)                                                |                                                  [**旋轉框物件辨識**](#pp-yoloe-r-高效能旋轉框辨識模型)                                                  |                                            [**3D物件辨識**](https://github.com/PaddlePaddle/Paddle3D)                                            |
| :--------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src='https://user-images.githubusercontent.com/61035602/206095864-f174835d-4e9a-42f7-96b8-d684fc3a3687.png' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206095892-934be83a-f869-4a31-8e52-1074184149d1.jpg' height="126px" width="180px"> |  <img src='https://user-images.githubusercontent.com/61035602/206111796-d9a9702a-c1a0-4647-b8e9-3e1307e9d34c.png' height="126px" width="180px">  | <img src='https://user-images.githubusercontent.com/61035602/206095622-cf6dbd26-5515-472f-9451-b39bbef5b1bf.gif' height="126px" width="180px"> |
|                                                              [**人臉辨識**](#模型库)                                                               |                                                [**2D特徵點偵測**](#️pp-tinypose-人體骨骼特徵點辨識)                                                 |                                                  [**多目標追蹤**](#pp-tracking-实时多物件追蹤系统)                                                   |                                                              [**語義分割**](#模型库)                                                               |
| <img src='https://user-images.githubusercontent.com/61035602/206095684-72f42233-c9c7-4bd8-9195-e34859bd08bf.jpg' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206100220-ab01d347-9ff9-4f17-9718-290ec14d4205.gif' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206111753-836e7827-968e-4c80-92ef-7a78766892fc.gif' height="126px" width="180px"  > | <img src='https://user-images.githubusercontent.com/61035602/206095831-cc439557-1a23-4a99-b6b0-b6f2e97e8c57.jpg' height="126px" width="180px"> |
|                                               [**車輛分析——車牌辨識**](#️pp-vehicle-实时車輛分析工具)                                               |                                               [**車輛分析——車流統計**](#️pp-vehicle-实时車輛分析工具)                                               |                                                [**車輛分析——違規檢測**](#️pp-vehicle-实时車輛分析工具)                                                |                                               [**車輛分析——屬性分析**](#️pp-vehicle-实时車輛分析工具)                                               |
| <img src='https://user-images.githubusercontent.com/61035602/206099328-2a1559e0-3b48-4424-9bad-d68f9ba5ba65.gif' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206095918-d0e7ad87-7bbb-40f1-bcc1-37844e2271ff.gif' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206100295-7762e1ab-ffce-44fb-b69d-45fb93657fa0.gif' height="126px" width="180px"  > | <img src='https://user-images.githubusercontent.com/61035602/206095905-8255776a-d8e6-4af1-b6e9-8d9f97e5059d.gif' height="126px" width="180px"> |
|                                                [**行人分析——闖入分析**](#pp-human-实时行人分析工具)                                                |                                                [**行人分析——行為分析**](#pp-human-实时行人分析工具)                                                |                                                 [**行人分析——屬性分析**](#pp-human-实时行人分析工具)                                                 |                                                [**行人分析——人流統計**](#pp-human-实时行人分析工具)                                                |
| <img src='https://user-images.githubusercontent.com/61035602/206095792-ae0ac107-cd8e-492a-8baa-32118fc82b04.gif' height="126px" width="180px"> | <img src='https://user-images.githubusercontent.com/61035602/206095778-fdd73e5d-9f91-48c7-9d3d-6f2e02ec3f79.gif' height="126px" width="180px"> |  <img src='https://user-images.githubusercontent.com/61035602/206095709-2c3a209e-6626-45dd-be16-7f0bf4d48a14.gif' height="126px" width="180px">  | <img src="https://user-images.githubusercontent.com/61035602/206113351-cc59df79-8672-4d76-b521-a15acf69ae78.gif" height="126px" width="180px"> |

同時，PaddleDetection提供了模型的線上體驗功能，使用者可以選擇自己的資料進行線上測試。

`說明`：考慮到伺服器負載壓力，線上測試皆使用 CPU 進行運算，完整的模型開發案例以及業界部署程式案例請前往[🎗️業界特色模型|業界工具](#️業界特色模型業界工具-1)。

`傳送門`：[模型線上測試](https://www.paddlepaddle.org.cn/models)

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/61035602/206896755-bd0cd498-1149-4e94-ae30-da590ea78a7a.gif" align="middle"/>
</p>
</div>

## ✨主要特性

#### 🧩模組化設計
PaddleDetection將檢測模型拆分成不同的模組，通過自定義模組之間的組合，使用者可以簡單快速地完成辨識模型的搭建。 `傳送門`：[模組](#模組)。

#### 📱豐富的套件
PaddleDetection支持大量的最新主流的演算法以及預訓練模型，涵蓋2D/3D物件辨識、語義分割、人臉辨識、特徵點辨識、多目標追蹤、半監督學習等方向。 `傳送門`：[📱模型庫](#模型庫)、[⚖️模型效能對比](#️模型效能對比)。

#### 🎗️業界特色模型|業界工具
PaddleDetection打造企業級特色模型以及分析工具：PP-YOLOE+、PP-PicoDet、PP-TinyPose、PP-HumanV2、PP-Vehicle等，針對通用、常見垂類應用場景提供深度優化解決方案以及高度集成的分析工具，降低開發者的測試、選擇時間，針對業界場景快速應用開發。 `傳送門`：[🎗️業界特色模型|業界工具](#️業界特色模型業界工具-1)。

#### 💡🏆業界部屬案例
PaddleDetection整理工業、農業、林業、交通、醫療、金融、能源電力等AI應用範例，建構資料標註-模型訓練-模型調教-預測部署整體流程，持續降低物件辨識技術的應用門檻。 `傳送門`：[💡業界部屬範例](#業界部屬範例)、[🏆企業應用案例](#企業應用案例)。

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/61035602/206431371-912a14c8-ce1e-48ec-ae6f-7267016b308e.png" align="middle" width="1280"/>
</p>
</div>

## 📣最新进展

**💎穩定版本**

位於[`release/2.5`](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5)分支，最新的[**v2.5**](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5)版本已經在 2022.09.13 發布，版本更新詳細內容請參考[v2.5.0更新日誌](https://github.com/PaddlePaddle/PaddleDetection/releases/tag/v2.5.0)，重點更新：
- [🎗️業界特色模型|業界工具](#️業界特色模型業界工具-1)：
    - 發布[PP-YOLOE+](configs/ppyoloe)，最高精度提升2.4% mAP，達到54.9% mAP，模型訓練收斂速度提升3.75倍，端到端預測速度最高提升2.3倍；多個下游任務泛化性提升
    - 發布[PicoDet-NPU](configs/picodet)模型，支持模型全量化部署；新增[PicoDet](configs/picodet)版面分析模型
    - 發布[PP-TinyPose升級版](./configs/keypoint/tiny_pose/)增強版，在健身、舞蹈等場景精度提升9.1% AP，支持側身、臥躺、跳躍、高抬腿等非常規動作
    - 發布行人分析工具[PP-Human v2](./deploy/pipeline)，新增打架、打電話、抽煙、闖入四大行為辨識，底層演算法效能升級，覆蓋行人辨識、追蹤、屬性三類核心算法能力，提供保姆級全流程開發及模型優化方案，支持線上影像串流
    - 首次發布[PP-Vehicle](./deploy/pipeline)，提供車牌辨識、車輛屬性分析（顏色、車型）、車流量統計以及違規辨識四大功能，兼容圖片、線上影像串流、影片輸入，提供完善的二次開發文檔教案
- [📱模型庫](#模型庫)：
    - 所有[YOLO家族](https://github.com/PaddlePaddle/PaddleYOLO/tree/develop/docs/MODEL_ZOO_cn.md)經典與最新算法模型的程式庫[PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO): 包括YOLOv3，百度飛槳自研的最新高精度目標檢測模型PP-YOLOE，以及前沿檢測算法YOLOv4、YOLOv5、YOLOX，YOLOv6及YOLOv7
    - 新增基於[ViT](configs/vitdet)骨幹網絡高精度檢測模型，COCO資料集精度達到55.7% mAP；新增[OC-SORT](configs/mot/ocsort)多目標跟蹤模型；新增[ConvNeXt](configs/convnext)骨幹網絡
- [💡業界部屬範例](#業界部屬範例)：
    - 新增[智能健身](https://aistudio.baidu.com/aistudio/projectdetail/4385813)、[打架辨識](https://aistudio.baidu.com/aistudio/projectdetail/4086987?channelType=0&channel=0)、[來客分析](https://aistudio.baidu.com/aistudio/projectdetail/4230123?channelType=0&channel=0)

**🧬預覽版本**

位於[`develop`](https://github.com/PaddlePaddle/PaddleDetection/tree/develop)分支，體驗最新功能請切換到[該分支](https://github.com/PaddlePaddle/PaddleDetection/tree/develop)，最近更新：
- [📱模型庫](#模型庫)：
  - 新增[半監督檢測模型](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/semi_det);
- [🎗️業界特色模型|業界工具](#️業界特色模型|業界工具-1)：
  - 發布**旋轉框物件辨識**[PP-YOLOE-R](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rotate/ppyoloe_r)：Anchor-free旋轉框辨識SOTA模型，精度速度雙高、雲邊一體，s/m/l/x四個模型適配不用算力硬件、部署友好，避免使用特殊算子，能夠輕鬆使用TensorRT加速；
  - 發布**小型物件辨識**[PP-YOLOE-SOD](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/smalldet)：基於切圖的端到端辨識方案、基於原圖的辨識模型，精度達VisDrone開源最優；

## 👫開源社群

- **📑專案合作：** 如果您是企業開發者且有明確的物件辨識垂類應用需求，請掃描如下二維碼入群，並聯繫`群管理員AI`後可免費與官方團隊展開不同層次的合作。
- **🏅️社群貢獻：** PaddleDetection非常歡迎你加入到飛槳社群的開源專案中，參與貢獻方式可以參考[開源項目開發指南](docs/contribution/README.md)。
- **💻直播影片：** PaddleDetection會定期在飛槳直播間([B站:飛槳PaddlePaddle](https://space.bilibili.com/476867757)、[微信: 飛槳PaddlePaddle](https://mp.weixin.qq.com/s/6ji89VKqoXDY6SSGkxS8NQ))，針對發布的最新內容、以及企業範例、使用教學等進行直播分享。
- **🎁加入社群：** **微信掃描 QR code並填寫問卷之後，可以及時獲取如下資訊，包括：**
  - 社群最新文章、直播課等活動預告
  - 過去的直播錄播&PPT
  - 30+行人車輛等垂類高效能預訓練模型
  - 七大任務開源資料集下載連結
  - 40+前沿辨識領域頂級期刊演算法
  - 15+從零上手物件辨識理論與實作影片課程
  - 10+工業安防交通全流程專案（含程式碼）

<div align="center">
<img src="https://user-images.githubusercontent.com/22989727/202123813-1097e3f6-c784-4991-9b94-8cbcd972de82.png"  width = "150" height = "150",caption='' />
<p>PaddleDetection官方交流群 QR code</p>
</div>

- **🎈社群近期活動**

  - **⚽️2022卡達世界盃專題**
    - `文章傳送門`：[世界杯决赛号角吹响！趁周末来搭一套足球3D+AI量化分析系统吧！](https://mp.weixin.qq.com/s/koJxjWDPBOlqgI-98UsfKQ)
  
    <div align="center">
    <img src="https://user-images.githubusercontent.com/61035602/208036574-f151a7ff-a5f1-4495-9316-a47218a6576b.gif"  height = "250" caption='' />
    <p></p>
    </div>

  - **🔍旋轉框辨識專題**
    - `文章傳送門`：[Yes, PP-YOLOE！80.73mAP、38.5mAP，旋转框、小型物件辨識能力双SOTA！](https://mp.weixin.qq.com/s/6ji89VKqoXDY6SSGkxS8NQ)
  
    <div align="center">
    <img src="https://user-images.githubusercontent.com/61035602/208037368-5b9f01f7-afd9-46d8-bc80-271ccb5db7bb.png"  height = "220" caption='' />
    <p></p>
    </div>
    
  - **🎊YOLO Vision世界學術交流大會**
    - **PaddleDetection**受邀参与首个以**YOLO为主题**的**YOLO-VISION**世界大会，与全球AI领先开发者学习交流。
    - `活动連結傳送門`：[YOLO-VISION](https://ultralytics.com/yolo-vision)

    <div  align="center">
    <img src="https://user-images.githubusercontent.com/48054808/192301374-940cf2fa-9661-419b-9c46-18a4570df381.jpeg" width="400"/>
    </div>

- **🏅️社群貢獻**
  - `活動連結傳送門`：[Yes, PP-YOLOE! 基於PP-YOLOE的算法开发](https://github.com/PaddlePaddle/PaddleDetection/issues/7345)


## 🍱安裝

參考[安裝說明](docs/tutorials/INSTALL_cn.md)進行安裝。

## 🔥教學

**深度學習入門課程**

- [零基礎入門深度學習](https://www.paddlepaddle.org.cn/tutorials/projectdetail/4676538)
- [零基礎入門目標檢測](https://aistudio.baidu.com/aistudio/education/group/info/1617)

**快速開始**

- [快速體驗](docs/tutorials/QUICK_STARTED_cn.md)
- [範例：30分鐘快速開發交通標誌辨識模型](docs/tutorials/GETTING_STARTED_cn.md)

**資料準備**
- [資料準備](docs/tutorials/data/README.md)
- [資料處理](docs/advanced_tutorials/READER.md)

**配置文件說明**
- [RCNN參數說明](docs/tutorials/config_annotation/faster_rcnn_r50_fpn_1x_coco_annotation.md)
- [PP-YOLO參數說明](docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation.md)

**模型開發**

- [新增辨識模型](docs/advanced_tutorials/MODEL_TECHNICAL.md)
- 二次開發
  - [物件辨識](docs/advanced_tutorials/customization/detection.md)
  - [特徵點偵測](docs/advanced_tutorials/customization/keypoint_detection.md)
  - [多目標追蹤](docs/advanced_tutorials/customization/pphuman_mot.md)
  - [行為辨識](docs/advanced_tutorials/customization/action_recognotion/)
  - [屬性辨識](docs/advanced_tutorials/customization/pphuman_attribute.md)

**模型部屬**

- [模型儲存教學](deploy/EXPORT_MODEL.md)
- [模型壓縮](https://github.com/PaddlePaddle/PaddleSlim)
  - [剪枝/量化/蒸餾教學](configs/slim)
- [Paddle Inference部署](deploy/README.md)
  - [Python 部署](deploy/python)
  - [C++ 部署](deploy/cpp)
- [Paddle Lite 部署](deploy/lite)
- [Paddle Serving 部署](deploy/serving)
- [ONNX 模型儲存](deploy/EXPORT_ONNX_MODEL.md)
- [推理 benchmark](deploy/BENCHMARK_INFER.md)

## 🔑FAQ
- [FAQ/常見問題彙總](docs/tutorials/FAQ)

## 🧩模組

<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Necks</b>
      </td>
      <td>
        <b>Loss</b>
      </td>
      <td>
        <b>Common</b>
      </td>
      <td>
      <b>Data Augmentation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
          <li><a href="ppdet/modeling/backbones/resnet.py">ResNet</a></li>
          <li><a href="ppdet/modeling/backbones/res2net.py">CSPResNet</a></li>
          <li><a href="ppdet/modeling/backbones/senet.py">SENet</a></li>
          <li><a href="ppdet/modeling/backbones/res2net.py">Res2Net</a></li>
          <li><a href="ppdet/modeling/backbones/hrnet.py">HRNet</a></li>
          <li><a href="ppdet/modeling/backbones/lite_hrnet.py">Lite-HRNet</a></li>
          <li><a href="ppdet/modeling/backbones/darknet.py">DarkNet</a></li>
          <li><a href="ppdet/modeling/backbones/csp_darknet.py">CSPDarkNet</a></li>
          <li><a href="ppdet/modeling/backbones/mobilenet_v1.py">MobileNetV1</a></li>
          <li><a href="ppdet/modeling/backbones/mobilenet_v3.py">MobileNetV1</a></li>  
          <li><a href="ppdet/modeling/backbones/shufflenet_v2.py">ShuffleNetV2</a></li>
          <li><a href="ppdet/modeling/backbones/ghostnet.py">GhostNet</a></li>
          <li><a href="ppdet/modeling/backbones/blazenet.py">BlazeNet</a></li>
          <li><a href="ppdet/modeling/backbones/dla.py">DLA</a></li>
          <li><a href="ppdet/modeling/backbones/hardnet.py">HardNet</a></li>
          <li><a href="ppdet/modeling/backbones/lcnet.py">LCNet</a></li>  
          <li><a href="ppdet/modeling/backbones/esnet.py">ESNet</a></li>  
          <li><a href="ppdet/modeling/backbones/swin_transformer.py">Swin-Transformer</a></li>
          <li><a href="ppdet/modeling/backbones/convnext.py">ConvNeXt</a></li>
          <li><a href="ppdet/modeling/backbones/vgg.py">VGG</a></li>
          <li><a href="ppdet/modeling/backbones/vision_transformer.py">Vision Transformer</a></li>
          <li><a href="configs/convnext">ConvNext</a></li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="ppdet/modeling/necks/bifpn.py">BiFPN</a></li>
        <li><a href="ppdet/modeling/necks/blazeface_fpn.py">BlazeFace-FPN</a></li>
        <li><a href="ppdet/modeling/necks/centernet_fpn.py">CenterNet-FPN</a></li>
        <li><a href="ppdet/modeling/necks/csp_pan.py">CSP-PAN</a></li>
        <li><a href="ppdet/modeling/necks/custom_pan.py">Custom-PAN</a></li>
        <li><a href="ppdet/modeling/necks/fpn.py">FPN</a></li>
        <li><a href="ppdet/modeling/necks/es_pan.py">ES-PAN</a></li>
        <li><a href="ppdet/modeling/necks/hrfpn.py">HRFPN</a></li>
        <li><a href="ppdet/modeling/necks/lc_pan.py">LC-PAN</a></li>
        <li><a href="ppdet/modeling/necks/ttf_fpn.py">TTF-FPN</a></li>
        <li><a href="ppdet/modeling/necks/yolo_fpn.py">YOLO-FPN</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="ppdet/modeling/losses/smooth_l1_loss.py">Smooth-L1</a></li>
          <li><a href="ppdet/modeling/losses/detr_loss.py">Detr Loss</a></li> 
          <li><a href="ppdet/modeling/losses/fairmot_loss.py">Fairmot Loss</a></li>
          <li><a href="ppdet/modeling/losses/fcos_loss.py">Fcos Loss</a></li>
          <li><a href="ppdet/modeling/losses/gfocal_loss.py">GFocal Loss</a></li> 
          <li><a href="ppdet/modeling/losses/jde_loss.py">JDE Loss</a></li>
          <li><a href="ppdet/modeling/losses/keypoint_loss.py">KeyPoint Loss</a></li>
          <li><a href="ppdet/modeling/losses/solov2_loss.py">SoloV2 Loss</a></li>
          <li><a href="ppdet/modeling/losses/focal_loss.py">Focal Loss</a></li>
          <li><a href="ppdet/modeling/losses/iou_loss.py">GIoU/DIoU/CIoU</a></li>  
          <li><a href="ppdet/modeling/losses/iou_aware_loss.py">IoUAware</a></li>
          <li><a href="ppdet/modeling/losses/sparsercnn_loss.py">SparseRCNN Loss</a></li>
          <li><a href="ppdet/modeling/losses/ssd_loss.py">SSD Loss</a></li>
          <li><a href="ppdet/modeling/losses/focal_loss.py">YOLO Loss</a></li>
          <li><a href="ppdet/modeling/losses/yolo_loss.py">CT Focal Loss</a></li>
          <li><a href="ppdet/modeling/losses/varifocal_loss.py">VariFocal Loss</a></li>
        </ul>
      </td>
      <td>
      </ul>
          <li><b>Post-processing</b></li>
        <ul>
        <ul>
           <li><a href="ppdet/modeling/post_process.py">SoftNMS</a></li>
            <li><a href="ppdet/modeling/post_process.py">MatrixNMS</a></li>
            </ul>
            </ul>
          <li><b>Training</b></li>
        <ul>
        <ul>
            <li><a href="tools/train.py#L62">FP16 training</a></li>
            <li><a href="docs/tutorials/DistributedTraining_cn.md">Multi-machine training </a></li>
                        </ul>
            </ul>
          <li><b>Common</b></li>
        <ul>
        <ul> 
            <li><a href="ppdet/modeling/backbones/resnet.py#L41">Sync-BN</a></li>
            <li><a href="configs/gn/README.md">Group Norm</a></li>
            <li><a href="configs/dcn/README.md">DCNv2</a></li>
            <li><a href="ppdet/optimizer/ema.py">EMA</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="ppdet/data/transform/operators.py">Resize</a></li>  
          <li><a href="ppdet/data/transform/operators.py">Lighting</a></li>  
          <li><a href="ppdet/data/transform/operators.py">Flipping</a></li>  
          <li><a href="ppdet/data/transform/operators.py">Expand</a></li>
          <li><a href="ppdet/data/transform/operators.py">Crop</a></li>
          <li><a href="ppdet/data/transform/operators.py">Color Distort</a></li>  
          <li><a href="ppdet/data/transform/operators.py">Random Erasing</a></li>  
          <li><a href="ppdet/data/transform/operators.py">Mixup </a></li>
          <li><a href="ppdet/data/transform/operators.py">AugmentHSV</a></li>
          <li><a href="ppdet/data/transform/operators.py">Mosaic</a></li>
          <li><a href="ppdet/data/transform/operators.py">Cutmix </a></li>
          <li><a href="ppdet/data/transform/operators.py">Grid Mask</a></li>
          <li><a href="ppdet/data/transform/operators.py">Auto Augment</a></li>  
          <li><a href="ppdet/data/transform/operators.py">Random Perspective</a></li>  
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

## 📱模型庫

<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>2D Detection</b>
      </td>
      <td>
        <b>Multi Object Tracking</b>
      </td>
      <td>
        <b>KeyPoint Detection</b>
      </td>
      <td>
      <b>Others</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/faster_rcnn/README.md">Faster RCNN</a></li>
            <li><a href="ppdet/modeling/necks/fpn.py">FPN</a></li>
            <li><a href="configs/cascade_rcnn/README.md">Cascade-RCNN</a></li>
            <li><a href="configs/rcnn_enhance">PSS-Det</a></li>
            <li><a href="configs/retinanet/README.md">RetinaNet</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleYOLO">YOLOv3</a></li>  
            <li><a href="https://github.com/PaddlePaddle/PaddleYOLO">YOLOv5</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleYOLO">YOLOX</a></li>  
            <li><a href="https://github.com/PaddlePaddle/PaddleYOLO">YOLOv6</a></li>  
            <li><a href="https://github.com/PaddlePaddle/PaddleYOLO">YOLOv7</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleYOLO">RTMDet</a></li>   
            <li><a href="configs/ppyolo/README_cn.md">PP-YOLO</a></li>
            <li><a href="configs/ppyolo#pp-yolo-tiny">PP-YOLO-Tiny</a></li>
            <li><a href="configs/picodet">PP-PicoDet</a></li>
            <li><a href="configs/ppyolo/README_cn.md">PP-YOLOv2</a></li>
            <li><a href="configs/ppyoloe/README_legacy.md">PP-YOLOE</a></li>
            <li><a href="configs/ppyoloe/README_cn.md">PP-YOLOE+</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/smalldet">PP-YOLOE-SOD</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/rotate/README.md">PP-YOLOE-R</a></li>
            <li><a href="configs/ssd/README.md">SSD</a></li>
            <li><a href="configs/centernet">CenterNet</a></li>
            <li><a href="configs/fcos">FCOS</a></li>  
            <li><a href="configs/ttfnet">TTFNet</a></li>
            <li><a href="configs/tood">TOOD</a></li>
            <li><a href="configs/gfl">GFL</a></li>
            <li><a href="configs/gfl/gflv2_r50_fpn_1x_coco.yml">GFLv2</a></li>
            <li><a href="configs/detr">DETR</a></li>
            <li><a href="configs/deformable_detr">Deformable DETR</a></li>
            <li><a href="configs/sparse_rcnn">Sparse RCNN</a></li>
      </ul>
      </td>
      <td>
        <ul>
           <li><a href="configs/mot/jde">JDE</a></li>
            <li><a href="configs/mot/fairmot">FairMOT</a></li>
            <li><a href="configs/mot/deepsort">DeepSORT</a></li>
            <li><a href="configs/mot/bytetrack">ByteTrack</a></li>
            <li><a href="configs/mot/ocsort">OC-SORT</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/keypoint/hrnet">HRNet</a></li>
            <li><a href="configs/keypoint/higherhrnet">HigherHRNet</a></li>
            <li><a href="configs/keypoint/lite_hrnet">Lite-HRNet</a></li>
            <li><a href="configs/keypoint/tiny_pose">PP-TinyPose</a></li>
        </ul>
</td>
<td>
</ul>
          <li><b>Instance Segmentation</b></li>
        <ul>
        <ul>
          <li><a href="configs/mask_rcnn">Mask RCNN</a></li>
            <li><a href="configs/cascade_rcnn">Cascade Mask RCNN</a></li>
            <li><a href="configs/solov2">SOLOv2</a></li>
        </ul>
      </ul>
          <li><b>Face Detection</b></li>
        <ul>
        <ul>
            <li><a href="configs/face_detection">BlazeFace</a></li>
        </ul>
        </ul>
          <li><b>Semi-Supervised Detection</b></li>
        <ul>
        <ul>
            <li><a href="https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/semi_det">DenseTeacher</a></li>
        </ul>
        </ul>
          <li><b>3D Detection</b></li>
        <ul>
        <ul>
            <li><a href="https://github.com/PaddlePaddle/Paddle3D">Smoke</a></li>
            <li><a href="https://github.com/PaddlePaddle/Paddle3D">CaDDN</a></li>
            <li><a href="https://github.com/PaddlePaddle/Paddle3D">PointPillars</a></li>
            <li><a href="https://github.com/PaddlePaddle/Paddle3D">CenterPoint</a></li>
            <li><a href="https://github.com/PaddlePaddle/Paddle3D">SequeezeSegV3</a></li>
            <li><a href="https://github.com/PaddlePaddle/Paddle3D">IA-SSD</a></li>
            <li><a href="https://github.com/PaddlePaddle/Paddle3D">PETR</a></li>
        </ul>
        </ul>
          <li><b>Vehicle Analysis Toolbox</b></li>
        <ul>
        <ul>
            <li><a href="deploy/pipeline/README.md">PP-Vehicle</a></li>
        </ul>
        </ul>
          <li><b>Human Analysis Toolbox</b></li>
        <ul>
        <ul>
            <li><a href="deploy/pipeline/README.md">PP-Human</a></li>
            <li><a href="deploy/pipeline/README.md">PP-HumanV2</a></li>
        </ul>
        </ul>
          <li><b>Sport Analysis Toolbox</b></li>
        <ul>
        <ul>
            <li><a href="https://github.com/PaddlePaddle/PaddleSports">PP-Sports</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## ⚖️模型效能对比

#### 🖥️伺服器模型效能比較

各模型結構和骨幹網絡的代表模型在 COCO 資料集上精度 mAP 和單張 Tesla V100 上預測速度(FPS)對比圖。

  <div  align="center">
  <img src="https://user-images.githubusercontent.com/61035602/206434766-caaa781b-b922-481f-af09-15faac9ed33b.png" width="800"/>
</div>

<details>
<summary><b> 測試說明(點擊展開)</b></summary>

- ViT为ViT-Cascade-Faster-RCNN模型，COCO 資料集 mAP 高達 55.7%
- Cascade-Faster-RCNN 為 Cascade-Faster-RCNN-ResNet50vd-DCN，PaddleDetection 將其優化到COCO數據mAP為47.8%時推理速度為20FPS
- PP-YOLOE是對PP-YOLO v2模型的進一步優化，L版本在COCO資料集mAP為51.6%，Tesla V100預測速度78.1FPS
- PP-YOLOE+是對PPOLOE模型的進一步優化，L版本在COCO資料集mAP為53.3%，Tesla V100預測速度78.1FPS
- YOLOX和YOLOv5均為基於PaddleDetection復現算法，YOLOv5代碼在PaddleYOLO中，參照PaddleYOLO_MODEL
- 圖中模型均可在[📱模型庫](#模型庫)中獲取
</details>

#### ⌚️行動裝置模型效能比較

各行動裝置模型在 COCO 資料集上精度 mAP 和高通驍龍865處理器上預測速度(FPS)對比圖。

  <div  align="center">
  <img src="https://user-images.githubusercontent.com/61035602/206434741-10460690-8fc3-4084-a11a-16fe4ce2fc85.png" width="550"/>
</div>


<details>
<summary><b> 測試說明(點擊展開)</b></summary>

- 測試資料均使用高通驍龍865(4xA77+4xA55)處理器，batch size為1, 開啟4線程測試，測試使用NCNN預測庫，測試腳本見[MobileDetBenchmark](https://github.com/JiweiMaster/MobileDetBenchmark)
- PP-PicoDet及PP-YOLO-Tiny為PaddleDetection自研模型，可在[📱模型庫](#模型庫)中獲得，其餘模型PaddleDetection暫未提供
</details>

## 🎗️業界特色模型|業界工具

產業特色模型｜產業工具是 PaddleDetection 針對產業常見應用場景打造的，兼顧精度和速度的模型以及工具箱，注重從資料處理-模型訓練-模型調教到模型部署，且提供了實際生產環境中的範例程式碼，幫助擁有類似需求的開發者高效的完成產品開發。

該系列模型｜工具均以 PP 前綴命名，具體介紹、預訓練模型以及業界範例程式碼如下。

### 💎PP-YOLOE 高精度物件辨識模型

<details>
<summary><b> 簡介(點擊展開)</b></summary>

PP-YOLOE是基於 PP-YOLOv2 的卓越的 one-stage Anchor-free 模型，超越了多種流行的 YOLO 模型。 PP-YOLOE 避免了使用諸如Deformable Convolution 或者 Matrix NMS之類的特殊算子，以使其能輕鬆地部署在多種多樣的硬件上。其使用大規模資料集 obj365 預訓練模型進行預訓練，可以在不同場景資料集上快速調優收斂。

`傳送門`：[PP-YOLOE說明](configs/ppyoloe/README_cn.md)。

`傳送門`：[arXiv論文](https://arxiv.org/abs/2203.16250)。

</details>

<details>
<summary><b> 預訓練模型(點擊展開)</b></summary>

| 模型名稱    | COCO精度（mAP） | V100 TensorRT FP16速度(FPS) | 推薦部署硬體規格 |                        配置文件                         |                                        模型下載                                         |
| :---------- | :-------------: | :-------------------------: | :----------: | :-----------------------------------------------------: | :-------------------------------------------------------------------------------------: |
| PP-YOLOE+_l |      53.3       |            149.2            |    伺服器    | [連結](configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml) | [下載地址](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco.pdparams) |

`傳送門`：[全部預訓練模型](configs/ppyoloe/README_cn.md)。
</details>

<details>
<summary><b> 企業應用程式碼範例(點擊展開)</b></summary>

| 行業 | 類別              | 特點                                                                                          | 文件說明                                                      | 模型下載                                            |
| ---- | ----------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------- |
| 農業 | 農作物辨識        | 用於葡萄栽培中基於圖片的監測和現場機器人技術，提供了來自5種不同葡萄品種的實地實例             | [PP-YOLOE+ 下游任务](./configs/ppyoloe/application/README.md) | [下載連結](./configs/ppyoloe/application/README.md) |
|通用 | 低光場景檢測      | 低光資料集使用ExDark，包括從極低光環境到暮光環境等10種不同光照條件下的圖片。                  | [PP-YOLOE+ 下游任务](./configs/ppyoloe/application/README.md) | [下載連結](./configs/ppyoloe/application/README.md) |
| 工業 | PCB電路板瑕疵檢測 | 工業資料集使用PKU-Market-PCB，該資料集用於印刷電路板（PCB）的瑕疵檢測，提供了6種常見的PCB缺陷 | [PP-YOLOE+ 下游任务](./configs/ppyoloe/application/README.md) | [下載連結](./configs/ppyoloe/application/README.md) |
</details>

### 💎PP-YOLOE-R 高效能旋轉框辨識模型

<details>
<summary><b> 簡介(點擊展開)</b></summary>

PP-YOLOE-R是一個高效的 one-stage Anchor-free旋轉框辨識模型，基於PP-YOLOE+引入了一系列改進策略來提升辨識精度。根據不同的硬體設備對精度和速度的要求，PP-YOLOE-R包含s/m/l/x四個尺寸的模型。在DOTA 1.0資料集上，PP-YOLOE-R-l和PP-YOLOE-R-x在單尺度訓練和測試的情況下分別達到了78.14mAP和78.28 mAP，這在單尺度評估下超越了幾乎所有的旋轉框檢測模型。通過多尺度訓練和測試，PP-YOLOE-R-l和PP-YOLOE-R-x的檢測精度進一步提升至80.02mAP和80.73 mAP，超越了所有的Anchor-free方法並且和最先進的Anchor-based的兩階段模型精度幾乎相當。在保持高精度的同時，PP-YOLOE-R避免使用特殊的算子，例如Deformable Convolution或Rotated RoI Align，使其能輕鬆地部署在多種多樣的硬體設備上。

`傳送門`：[PP-YOLOE-R說明](https://github.com/thinkthinking/PaddleDetection/tree/develop/configs/rotate/ppyoloe_r)。

`傳送門`：[arXiv論文](https://arxiv.org/abs/2211.02386)。

</details>

<details>
<summary><b> 預訓練模型(點擊展開)</b></summary>

|     模型     | Backbone |  mAP  | V100 TRT FP16 (FPS) | RTX 2080 Ti TRT FP16 (FPS) | Params (M) | FLOPs (G) | 學習率策略 | 角度表示 | 資料擴增 | GPU數量 | 每 GPU 圖片數量 |                                      模型下載                                       |                                                            配置文件                                                            |
| :----------: | :------: | :---: | :-----------------: | :------------------------: | :--------: | :-------: | :--------: | :------: | :------: | :-----: | :-----------: | :---------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
| PP-YOLOE-R-l |  CRN-l   | 80.02 |        69.7         |            48.3            |   53.29    |  281.65   |     3x     |    oc    |  MS+RR   |    4    |       2       | [model](https://paddledet.bj.bcebos.com/models/ppyoloe_r_crn_l_3x_dota_ms.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota_ms.yml) |

`傳送門`：[全部預訓練模型](https://github.com/thinkthinking/PaddleDetection/tree/develop/configs/rotate/ppyoloe_r)。
</details>

<details>
<summary><b> 企業應用程式碼範例(點擊展開)</b></summary>

| 行業 | 類別       | 特點                                                                  | 文件說明                                                                                | 模型下載                                                              |
| ---- | ---------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 通用 | 旋轉框辨識 | 手把手教你上手PP-YOLOE-R旋轉框辨識，10分鐘將脊柱資料集精度訓練至95mAP | [基於PP-YOLOE-R的旋轉框辨識](https://aistudio.baidu.com/aistudio/projectdetail/5058293) | [下載連結](https://aistudio.baidu.com/aistudio/projectdetail/5058293) |
</details>

### 💎PP-YOLOE-SOD 高精度小型物件辨識模型

<details>
<summary><b> 簡介(點擊展開)</b></summary>

PP-YOLOE-SOD(Small Object Detection)是 PaddleDetection 團隊針對小型物件辨識提出的辨識方案，在 VisDrone-DET 資料集上單模型精度達到38.5mAP，達到了SOTA效能。其分別基於切圖拼圖流程優化的小型物件辨識方案以及基於原圖模型演算法優化的小型物件辨識方案。同時提供了資料集自動分析腳本，只需輸入資料集標註文件，便可得到資料集統計結果，輔助判斷資料集是否是小目標資料集以及是否需要採用切圖策略，同時給出網絡超參數參考值。

`傳送門`：[PP-YOLOE-SOD 高精度小型物件辨識模型](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/smalldet)。

</details>

<details>
<summary><b> 預訓練模型(點擊展開)</b></summary>
- VisDrone資料集預訓練模型

| 模型                | COCOAPI mAP<sup>val<br>0.5:0.95 | COCOAPI mAP<sup>val<br>0.5 | COCOAPI mAP<sup>test_dev<br>0.5:0.95 | COCOAPI mAP<sup>test_dev<br>0.5 | MatlabAPI mAP<sup>test_dev<br>0.5:0.95 | MatlabAPI mAP<sup>test_dev<br>0.5 |                                              下载                                               |                           配置文件                           |
| :------------------ | :-----------------------------: | :------------------------: | :----------------------------------: | :-----------------------------: | :------------------------------------: | :-------------------------------: | :---------------------------------------------------------------------------------------------: | :----------------------------------------------------------: |
| **PP-YOLOE+_SOD-l** |            **31.9**             |          **52.1**          |               **25.6**               |            **43.5**             |               **30.25**                |             **51.18**             | [下載連結](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_sod_crn_l_80e_visdrone.pdparams) | [配置文件](visdrone/ppyoloe_plus_sod_crn_l_80e_visdrone.yml) |

`傳送門`：[全部預訓練模型](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/smalldet)。
</details>

<details>
<summary><b> 企業應用程式碼範例(點擊展開)</b></summary>

| 行業 | 類別       | 特點                                                 | 文件說明                                                                                          | 模型下載                                                              |
| ---- | ---------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 通用 | 小型物件辨識 | 基於PP-YOLOE-SOD的無人機航拍圖片辨識完整範例。 | [基於PP-YOLOE-SOD的無人機航航拍圖辨識](https://aistudio.baidu.com/aistudio/projectdetail/5036782) | [下載連結](https://aistudio.baidu.com/aistudio/projectdetail/5036782) |
</details>

### 💫PP-PicoDet c

<details>
<summary><b> 簡介(點擊展開)</b></summary>

全新的輕量級系列模型PP-PicoDet，在行動裝置上具有卓越的效能，成為全新SOTA輕量級模型。

`傳送門`：[PP-PicoDet說明](configs/picodet/README.md)。

`傳送門`：[arXiv論文](https://arxiv.org/abs/2111.00902)。

</details>

<details>
<summary><b> 預訓練模型(點擊展開)</b></summary>

| 模型名稱  | COCO精度（mAP） | 驍龍865 四線程速度(FPS) |  推薦部署硬體規格  |                       配置文件                       |                                       模型下載                                       |
| :-------- | :-------------: | :---------------------: | :------------: | :--------------------------------------------------: | :----------------------------------------------------------------------------------: |
| PicoDet-L |      36.1       |          39.7           | 行動裝置、嵌入式 | [連結](configs/picodet/picodet_l_320_coco_lcnet.yml) | [下載地址](https://paddledet.bj.bcebos.com/models/picodet_l_320_coco_lcnet.pdparams) |

`傳送門`：[全部預訓練模型](configs/picodet/README.md)。
</details>


<details>
<summary><b> 企業應用程式碼範例(點擊展開)</b></summary>

| 行業     | 類別         | 特點                                                                                                                           | 文件說明                                                                                                          | 模型下載                                                                                      |
| -------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 智慧城市 |道路垃圾辨識 | 通過在市政垃圾車上安裝攝影機對路面垃圾辨識並分析，實現對路面上的垃圾進行監控，記錄並通知清潔人員清理，大大提升了清潔人員效率。 | [基於PP-PicoDet的路面垃圾辨識](https://aistudio.baidu.com/aistudio/projectdetail/3846170?channelType=0&channel=0) | [下載連結](https://aistudio.baidu.com/aistudio/projectdetail/3846170?channelType=0&channel=0) |
</details>

### 📡PP-Tracking 及時多物件追蹤系统

<details>
<summary><b> 簡介(點擊展開)</b></summary>

PaddleDetection團隊提供了實時多物件追蹤系統PP-Tracking，是基於PaddlePaddle深度學習框架的業界首個開源的實時多物件追蹤系統，具有模型豐富、應用廣泛和部署高效三大優勢。 PP-Tracking支持單鏡頭追蹤(MOT)和跨鏡頭追蹤(MTMCT)兩種模式，針對實際業務的難點和痛點，提供了行人追蹤、車輛追蹤、多類別追蹤、小物件追蹤、流量統計以及跨鏡頭追蹤等各種多目標追蹤功能和應用，部署方式支持API調用和GUI可視化界面，部署語言支持Python和C++，部署平台環境支持Linux、NVIDIA Jetson等。

`傳送門`：[PP-Tracking說明](configs/mot/README.md)。

</details>

<details>
<summary><b> 預訓練模型(點擊展開)</b></summary>

| 模型名稱  |               模型簡介               |          精度          | 速度(FPS) |      推薦部署硬體規格      |                          配置文件                          |                                              模型下載                                              |
| :-------- | :----------------------------------: | :--------------------: | :-------: | :--------------------: | :--------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| ByteTrack |   SDE多物件追蹤算法 僅包含檢測模型   |   MOT-17 test:  78.4   |     -     | 伺服器、行動裝置、嵌入式 |     [連結](configs/mot/bytetrack/bytetrack_yolox.yml)      |  [下載地址](https://bj.bcebos.com/v1/paddledet/models/mot/yolox_x_24e_800x1440_mix_det.pdparams)   |
| FairMOT   | JDE多物件追蹤算法 多任務聯合學習方法 |   MOT-16 test: 75.0    |     -     | 伺服器、行動裝置、嵌入式 | [連結](configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml) |     [下載地址](https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams)     |
| OC-SORT   |   SDE多物件追蹤算法 僅包含檢測模型   | MOT-17 half val:  75.5 |     -     | 伺服器、行動裝置、嵌入式 |        [連結](configs/mot/ocsort/ocsort_yolox.yml)         | [下載地址](https://bj.bcebos.com/v1/paddledet/models/mot/yolox_x_24e_800x1440_mix_mot_ch.pdparams) |
</details>

<details>
<summary><b> 企業應用程式碼範例(點擊展開)</b></summary>

| 行業 | 類別       | 特點                       | 文件說明                                                                                       | 模型下載                                                              |
| ---- | ---------- | -------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 通用 | 多物件追蹤 | 快速上手單鏡頭、多鏡頭追蹤 | [PP-Tracking之手把手玩轉多物件追蹤](https://aistudio.baidu.com/aistudio/projectdetail/3022582) | [下載連結](https://aistudio.baidu.com/aistudio/projectdetail/3022582) |
</details>

### ⛷️PP-TinyPose 人體骨骼特徵點識別

<details>
<summary><b> 簡介(點擊展開)</b></summary>

PaddleDetection 中的特徵點檢測部分緊跟最先進的算法，包括 Top-Down 和 Bottom-Up 兩種方法，可以滿足用戶的不同需求。同時，PaddleDetection 提供針對行動裝置設備優化的自研實時特徵點檢測模型 PP-TinyPose。

`傳送門`：[PP-TinyPose說明](configs/keypoint/tiny_pose)。

</details>

<details>
<summary><b> 預訓練模型(點擊展開)</b></summary>

|  模型名稱   |               模型簡介               | COCO精度（AP） |         速度(FPS)         |  推薦部署硬體規格  |                        配置文件                         |                                         模型下載                                         |
| :---------: | :----------------------------------: | :------------: | :-----------------------: | :------------: | :-----------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| PP-TinyPose |輕量級特徵點算法<br/>輸入尺寸256x192 |      68.8      | 驍龍865 四線程: 158.7 FPS | 行動裝置、嵌入式 | [連結](configs/keypoint/tiny_pose/tinypose_256x192.yml) | [下載地址](https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.pdparams) |

`傳送門`：[全部預訓練模型](configs/keypoint/README.md)。
</details>

<details>
<summary><b> 企業應用程式碼範例(點擊展開)</b></summary>

| 行業 | 類別 | 特點                                                                                                                                     | 文件說明                                                                                             | 模型下載                                                              |
| ---- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 運動 | 健身 | 提供從模型選擇、資料準備、模型訓練調教，到後處理邏輯和模型部署的全流程可複用方案，有效解決了複雜健身動作的及時辨識問題，打造AI虛擬健身教練！ | [基於PP-TinyPose增强版的智能健身动作辨識](https://aistudio.baidu.com/aistudio/projectdetail/4385813) | [下載連結](https://aistudio.baidu.com/aistudio/projectdetail/4385813) |
</details>

### 🏃🏻PP-Human 實時行人分析工具

<details>
<summary><b> 簡介(點擊展開)</b></summary>

PaddleDetection深入探索核心行業的常見場景，提供了行人開箱即用分析工具，支持圖片/單鏡頭影片/多鏡頭影片/在線影片流多種輸入方式，廣泛應用於智慧交通、智慧城市、工業巡檢等領域。支持伺服器端部署及TensorRT加速，T4伺服器上可達到及時。
PP-Human支持四大產業級功能：五大異常行為辨識、26種人體屬性分析、實時人流計數、跨鏡頭（ReID）追蹤。

`傳送門`：[PP-Human行人分析工具使用指南](deploy/pipeline/README.md)。

</details>

<details>
<summary><b> 預訓練模型(點擊展開)</b></summary>

|        任务        | T4 TensorRT FP16: 速度（FPS） | 推薦部署硬體規格 |                                                                                                                                         模型下載                                                                                                                                         |                             模型體积                              |
| :----------------: | :---------------------------: | :----------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------: |
| 行人辨識（高精度） |             39.8              |    伺服器    |                                                                                              [物件辨識](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                               |                               182M                                |
| 行人追蹤（高精度） |             31.4              |    伺服器    |                                                                                             [多物件追蹤](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                              |                               182M                                |
| 屬性辨識（高精度） |          單人 117.6           |    伺服器    |                                      [物件辨識](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br> [屬性辨識](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_small_person_attribute_954_infer.zip)                                       |                  物件辨識：182M<br>屬性辨識：86M                  |
|      摔倒辨識      |           單人 100            |    伺服器    | [多物件追蹤](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip) <br> [特徵點辨識](https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip) <br> [基於特徵點行为辨識](https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip) | 多物件追蹤：182M<br>特徵點辨識：101M<br>基於特徵點行为辨識：21.8M |
|      闖入辨識      |             31.4              |    伺服器    |                                                                                             [多物件追蹤](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                              |                               182M                                |
|      打架辨識      |             50.8              |    伺服器    |                                                                                              [视频分类](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)                                                                                               |                                90M                                |
|      抽烟辨識      |             340.1             |    伺服器    |                                    [物件辨識](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[基於人體id的物件辨識](https://bj.bcebos.com/v1/paddledet/models/pipeline/ppyoloe_crn_s_80e_smoking_visdrone.zip)                                    |            物件辨識：182M<br>基於人體id的物件辨識：27M            |
|     打电话辨識     |             166.7             |    伺服器    |                                      [物件辨識](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip)<br>[基於人體id的圖片分類](https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_calling_halfbody.zip)                                       |            物件辨識：182M<br>基於人體id的圖片分類：45M            |

`傳送門`：[完整預訓練模型](deploy/pipeline/README.md)。
</details>

<details>
<summary><b> 企業應用程式碼範例(點擊展開)</b></summary>

| 行業     | 類別     | 特點                                                                                                                                           | 文件說明                                                                                               | 模型下載                                                                                 |
| -------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| 智能安防 | 摔倒辨識 | 飞桨行人分析PP-Human中提供的摔倒辨識算法，採用了特徵點+時空圖捲積神經網路的技术，对摔倒姿势无限制、背景环境无要求。                                | [基於PP-Human v2的摔倒辨識](https://aistudio.baidu.com/aistudio/projectdetail/4606001)                 | [下載連結](https://aistudio.baidu.com/aistudio/projectdetail/4606001)                    |
| 智能安防 | 打架辨識 | 本項目基於PaddleVideo视频开发套件训练打架辨識模型，然后将训练好的模型集成到PaddleDetection的PP-Human中，助力行人行為分析。                     | [基於PP-Human的打架辨識](https://aistudio.baidu.com/aistudio/projectdetail/4086987?contributionType=1) | [下載連結](https://aistudio.baidu.com/aistudio/projectdetail/4086987?contributionType=1) |
| 智能安防 | 摔倒辨識 | 基於PP-Human完成来客分析整體流程。使用PP-Human完成來客分析中非常常見的場景： 1. 來客屬性辨識(單鏡和跨境可視化）；2. 來客行為辨識（摔倒辨識）。 | [基於PP-Human的来客分析案例教學](https://aistudio.baidu.com/aistudio/projectdetail/4537344)            | [下載連結](https://aistudio.baidu.com/aistudio/projectdetail/4537344)                    |
</details>

### 🏎️PP-Vehicle 即時車輛分析工具

<details>
<summary><b> 簡介(點擊展開)</b></summary>

PaddleDetection深入探索核心行業的常見場景，提供了車輛開箱即用分析工具，支持圖片/單鏡頭影片/多鏡頭影片/在線影片串流多種輸入方式，廣泛應用於智慧交通、智慧城市、工業巡檢等領域。支持伺服器部署及TensorRT加速，T4伺服器上可達到實時。
PP-Vehicle囊括四大交通場景核心功能：車牌辨識、屬性辨識、車流量統計、違章辨識。

`傳送門`：[PP-Vehicle車輛分析工具指南](deploy/pipeline/README.md)。

</details>

<details>
<summary><b> 預訓練模型(點擊展開)</b></summary>

|        任务        | T4 TensorRT FP16: 速度(FPS) | 推薦部署硬體規格 |                                                                                           模型方案                                                                                           |                模型體积                 |
| :----------------: | :-------------------------: | :----------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------: |
| 車輛辨識（高精度） |            38.9             |    伺服器    |                                                [物件辨識](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip)                                                |                  182M                   |
| 車輛追蹤（高精度） |             25              |    伺服器    |                                               [多物件追蹤](https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip)                                               |                  182M                   |
|      車牌辨識      |            213.7            |    伺服器    | [車牌辨識](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_det_infer.tar.gz) <br> [車牌辨識](https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_rec_infer.tar.gz) | 車牌辨識：3.9M  <br> 車牌字符辨識： 12M |
|      車輛属性      |            136.8            |    伺服器    |                                                  [屬性辨識](https://bj.bcebos.com/v1/paddledet/models/pipeline/vehicle_attribute_model.zip)                                                  |                  7.2M                   |

`傳送門`：[完整預訓練模型](deploy/pipeline/README.md)。
</details>

<details>
<summary><b> 企業應用程式碼範例(點擊展開)</b></summary>

| 行業     | 類別             | 特點                                                                                                               | 文件說明                                                                                      | 模型下載                                                              |
| -------- | ---------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 智慧交通 | 交通監控車輛分析 | 本項目基於PP-Vehicle演示智慧交通中最需要的車流量監控、車輛違規停車辨識以及車輛結構化（車牌、車型、顏色）分析三大場景。 | [基於PP-Vehicle的交通監控分析系统](https://aistudio.baidu.com/aistudio/projectdetail/4512254) | [下載連結](https://aistudio.baidu.com/aistudio/projectdetail/4512254) |
</details>

## 💡業界部屬範例

業界部屬案例是 PaddleDetection 針對常見物件辨識應用場景，提供的開發範例，幫助開發者從資料標註-模型訓練-模型調教到預測部署。
針對每個範例我們都通過[AI-Studio](https://ai.baidu.com/ai-doc/AISTUDIO/Tk39ty6ho)提供了專案程式碼以及說明，使用者可以同步運行體驗。

`傳送門`：[💡業界部屬範例完整列表](industrial_tutorial/README.md)

- [基於PP-YOLOE-R的旋轉框辨識](https://aistudio.baidu.com/aistudio/projectdetail/5058293)
- [基於PP-YOLOE-SOD的无人机航拍图像辨識](https://aistudio.baidu.com/aistudio/projectdetail/5036782)
- [基於PP-Vehicle的交通監控分析系统](https://aistudio.baidu.com/aistudio/projectdetail/4512254)
- [基於PP-Human v2的摔倒辨識](https://aistudio.baidu.com/aistudio/projectdetail/4606001)
- [基於PP-TinyPose增强版的智能健身动作辨識](https://aistudio.baidu.com/aistudio/projectdetail/4385813)
- [基於PP-Human的打架辨識](https://aistudio.baidu.com/aistudio/projectdetail/4086987?contributionType=1)
- [基於Faster-RCNN的瓷砖表面瑕疵辨識](https://aistudio.baidu.com/aistudio/projectdetail/2571419)
- [基於PaddleDetection的PCB瑕疵辨識](https://aistudio.baidu.com/aistudio/projectdetail/2367089)
- [基於FairMOT实现人流量统计](https://aistudio.baidu.com/aistudio/projectdetail/2421822)
- [基於YOLOv3实现跌倒辨識](https://aistudio.baidu.com/aistudio/projectdetail/2500639)
- [基於PP-PicoDetv2 的路面垃圾辨識](https://aistudio.baidu.com/aistudio/projectdetail/3846170?channelType=0&channel=0)
- [基於人體特徵點辨識的合规辨識](https://aistudio.baidu.com/aistudio/projectdetail/4061642?contributionType=1)
- [基於PP-Human的来客分析案例教學](https://aistudio.baidu.com/aistudio/projectdetail/4537344)
- 持續更新中...

## 🏆企業應用案例

企業應用案例是企業在真實的環境中應用 PaddleDetection 的方案設計思路，相比產業實踐範例其更強調整體專案的設計流程，可供開發者在專案設計中做參考。

`傳送門`：[企业应用案例完整列表](https://www.paddlepaddle.org.cn/customercase)

- [中国南方电网——变电站智慧巡检](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2330)
- [国铁电气——轨道在线智能巡检系统](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2280)
- [京东物流——园区車輛行为辨識](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2611)
- [中兴克拉—厂区传统仪表统计监测](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2618)
- [宁德时代—动力电池高精度质量辨識](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2609)
- [中国科学院空天信息创新研究院——高尔夫球场遥感监测](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2483)
- [御航智能——基於边缘的无人机智能巡检](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2481)
- [普宙无人机——高精度森林巡检](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2121)
- [领邦智能——红外无感测温監控](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2615)
- [北京地铁——口罩辨識](https://mp.weixin.qq.com/s/znrqaJmtA7CcjG0yQESWig)
- [音智达——工厂人员违规行为辨識](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2288)
- [华夏天信——输煤皮带机器人智能巡检](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2331)
- [优恩物联网——社区住户分类支持广告精准投放](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2485)
- [螳螂慧视——室内3D点云场景物體分割与辨識](https://www.paddlepaddle.org.cn/support/news?action=detail&id=2599)
- 持續更新中...

## 📝license

本專案的發布受[Apache 2.0 license](LICENSE)許可認證。


## 📌引用

```
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},