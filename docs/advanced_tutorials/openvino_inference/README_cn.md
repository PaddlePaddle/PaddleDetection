# 将FairMOT模型为ONNX格式,并用OpenVINO做推理

## 简介

PaddleDetection是一个充满活力的开源项目，拥有大量的贡献者和维护者。 PaddleDetection是PaddlePaddle下面一个人工智能框物体检测工具集，能够帮助开发人员快速的将人工智能集成到自己的项目和应用程序中。
Intel OpenVINO 是一个广泛使用的免费工具包。 它能帮助优化深度学习模型，并使用推理引擎将其部署到英特尔硬件上。
很显然，当我们可以协同上下游（PaddlePaddle, OpenVINO）一起工作,这将可以极大的简化工作流程, 并且帮助我们实现AI模型从开发到部署的流水线工作模式, 这也让我们的生活更轻松。

本文将向您展示如何在 PaddleDetection 中使用 Model Zoo 中的FairMOT模型 [FairMOT](../../../configs/mot/fairmot/README.md) 并用OpenVINO来实现推理过程。

------------

## 前提要求

为了专注于介绍如何在OpenVINO中使用飞桨的模型这一主题,本文将不是一片入门级文章,它不会帮助您设置好您的开发环境, 本文只会提供最核心的组件安装, 并且会为每个需要用到的组件提供相应的链接.

在开始之前 请确保您已经安装了 PaddlePaddle.

```
conda install paddlepaddle==2.2.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

为了运行演示程序, 您还需要下载已经转换好了的[ONNX格式的FairMOT模型](https://bj.bcebos.com/v1/paddledet/models/mot/fairmot_576_320_v3.onnx).

## 将FairMOT模型到ONNX格式

1. 下载[FairMOT推理模型](https://bj.bcebos.com/v1/paddledet/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.tar).

2. 使用Paddle2ONNX来转换FairMOT模型.

请确保您已经安装了[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX).

```
paddle2onnx --model_dir . --model_filename model.pdmodel \
--params_filename model.pdiparams \
--input_shape_dict "{'image': [1, 3, 320, 576], 'scale_factor': [1, 2], 'im_shape': [1, 2]}" \
--save_file fairmot_576_320_v2.onnx \
--opset_version 12 \
--enable_onnx_checker True
```

更多关于如何使用Paddle2ONNX的详细信息, 请参考: [ONNX模型导出](../../../deploy/EXPORT_ONNX_MODEL_en.md).

## 使用ONNX模型以及OpenVINO进行推理

当我们把Paddle模型转换成ONNX模型之后, 我们可以直接使用OpenVINO读取其模型 并且进行推理.

*<sub>请确保您已经安装了OpenVINO, 这里是[OpenVINO的安装指南](https://docs.openvino.ai/cn/latest/openvino_docs_install_guides_installing_openvino_linux.html).<sub>*

1. ### 创建一个execution network

所以这里要做的第一件事是获得一个执行网络，以后可以使用它来进行推理。
代码如下:

```
def get_net():
    ie = IECore()
    model_path = root_path / "PaddleDetection/FairMot/fairmot_576_320_v3.onnx"
    net = ie.read_network(model= str(model_path))
    exec_net = ie.load_network(network=net, device_name="CPU")
    return net, exec_net
```

2. ### 预处理

每个 AI 模型都有自己不同的预处理步骤，让我们看看 FairMOT 模型是如何做的：

```
def prepare_input():
    transforms = [
        T.Resize(target_size=(target_width, target_height)), 
        T.Normalize(mean=(0,0,0), std=(1,1,1))
    ]
    img_file = root_path / "images/street.jpeg"
    img = cv2.imread(str(img_file))
    normalized_img, _ = T.Compose(transforms)(img)
    # add an new axis in front
    img_input = normalized_img[np.newaxis, :]
    # scale_factor is calculated as: im_shape / original_im_shape
    h_scale = target_height / img.shape[0]
    w_scale = target_width / img.shape[1]
    input = {"image": img_input, "im_shape": [target_height, target_width], "scale_factor": [h_scale, w_scale]}
    return input, img
```

3. ### 预测

在我们完成了所有的负载网络和预处理之后，终于开始了预测阶段。

```
def predict(exec_net, input):
    result = exec_net.infer(input)
    return result
```

您可能会惊讶地看到, 最激动人心的步骤居然如此简单。 不过下一个阶段会加复杂。 

4. ### 后处理

相较于大多数其他类型的AI推理, MOT（Multi-Object Tracking）显然是特殊的. FairMOT 需要一个称为跟踪器的特殊对象来处理预测结果。 这个预测结果则包括预测检测和预测的行人特征向量。

幸运的是，PaddleDetection 为我们简化了这个过程，我们可以从`ppdet`导出JDETracker，然后用这个tracker挑选出来符合条件的检测框,而且我们不需要编写太多代码来处理它。


```
def postprocess(pred_dets, pred_embs, threshold = 0.5):
    tracker = JDETracker()
    online_targets_dict = tracker.update(pred_dets, pred_embs)
    online_tlwhs = defaultdict(list)
    online_scores = defaultdict(list)
    online_ids = defaultdict(list)
    for cls_id in range(1):
        online_targets = online_targets_dict[cls_id]
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            tscore = t.score
            # make sure the tscore is no less then the threshold.
            if tscore < threshold: continue
            # make sure the target area is not less than the min_box_area.
            if tlwh[2] * tlwh[3] <= tracker.min_box_area:
                continue
            # make sure the vertical ratio of a found target is within the range (1.6 as default ratio).
            if tracker.vertical_ratio > 0 and tlwh[2] / tlwh[3] > tracker.vertical_ratio:
                continue
            online_tlwhs[cls_id].append(tlwh)
            online_ids[cls_id].append(tid)
            online_scores[cls_id].append(tscore)
    online_im = plot_tracking_dict(
        img,
        1,
        online_tlwhs,
        online_ids,
        online_scores,
        frame_id=0)
    return online_im
```

5. ### 画出检测框(可选)

这一步是可选的。出于演示目的，我只使用 `plot_tracking_dict()` 方法在图像上绘制所有边界框。 但是，如果您没有相同的要求，则不需要这样做。 

```
online_im = plot_tracking_dict(
    img,
    1,
    online_tlwhs,
    online_ids,
    online_scores,
    frame_id=0)
```

这些就是在您的硬件上运行 FairMOT 所需要遵循的所有步骤。

之后会有一篇详细解释此过程的配套文章将会发布，并且该文章的链接将很快在此处更新。

完整代码请查看 [Paddle OpenVINO 预测](docs/advanced_tutorials/openvino_inference/fairmot_onnx_openvino.py).