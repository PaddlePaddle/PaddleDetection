# 更多应用


## 1. 版面分析任务

版面分析指的是对图片形式的文档进行区域划分，定位其中的关键区域，如文字、标题、表格、图片等。

<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/ppstructure/docs/table/result_all.jpg" width="800">
</div>


### 1.1 数据集

训练版面分析模型时主要用到了以下几个数据集。

| dataset     | 简介                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [cTDaR2019_cTDaR](https://cndplab-founder.github.io/cTDaR2019/) | 用于表格检测(TRACKA)和表格识别(TRACKB)。图片类型包含历史数据集(以cTDaR_t0开头，如cTDaR_t00872.jpg)和现代数据集(以cTDaR_t1开头，cTDaR_t10482.jpg)。 |
| [IIIT-AR-13K](http://cvit.iiit.ac.in/usodi/iiitar13k.php)    | 手动注释公开的年度报告中的图形或页面而构建的数据集，包含5类：table, figure, natural image, logo, and signature |
| [CDLA](https://github.com/buptlihang/CDLA)                   | 中文文档版面分析数据集，面向中文文献类（论文）场景，包含10类：Table、Figure、Figure caption、Table、Table caption、Header、Footer、Reference、Equation |
| [TableBank](https://github.com/doc-analysis/TableBank)       | 用于表格检测和识别大型数据集，包含Word和Latex2种文档格式     |
| [DocBank](https://github.com/doc-analysis/DocBank)           | 使用弱监督方法构建的大规模数据集(500K文档页面)，用于文档布局分析，包含12类：Author、Caption、Date、Equation、Figure、Footer、List、Paragraph、Reference、Section、Table、Title |




### 1.2 模型库

| 模型     | 图像输入尺寸 |  mAP<sup>val<br>0.5 |  下载地址  | 
| :-------- | :--------: |  :----------------: | :---------------: |
| PicoDet-LCNet_x1_0 |  800*608   |   93.5    | [trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_layout.pdparams) &#124; [inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar)  |
| PicoDet-LCNet_x1_0 + FGD |  800*608   |   94     | [trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout.pdparams) &#124; [inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_layout_infer.tar)  |