# Using OpenVINO for Inference

## Introduction
PaddleDetection has been a vibrant open-source project and has a large amout of contributors and maintainers around it. It is an AI framework which enables developers to quickly integrate AI capacities into their own projects and applications.

Intel OpenVINO is a widely used free toolkit. It facilitates the optimization of a deep learning model from a framework and deployment using an inference engine onto Intel hardware.

Apparently, the upstream(Paddle) and the downstream(Intel OpenVINO) can work together to streamline and simplify the process of developing an AI model and deploying the model onto hardware, which, in turn, makes our lives easier.

This article will show you how to use a PaddleDetection model [FairMOT](../../../configs/mot/fairmot/README.md) from the Model Zoo in PaddleDetection and use it with OpenVINO to do the inference.

------------

## Prerequisites

This article is not an entry level introduction to help you set up everything, in order to focus on its main purpose, the instruction of setting up environment will be kept at the minmum level and respective instructions will be provided by their official website links.

Before we can do anything, please make sure you have PaddlePaddle environment set up.

```
conda install paddlepaddle==2.2.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

Please also download the converted [ONNX format of FairMOT](https://bj.bcebos.com/v1/paddledet/models/mot/fairmot_576_320_v3.onnx)

## Export the PaddleDetection Model to ONNX format

1. Download the [FairMOT Inference Model](https://bj.bcebos.com/v1/paddledet/models/mot/fairmot_hrnetv2_w18_dlafpn_30e_576x320.tar)

2. Using Paddle2ONNX to convert the model

Make sure you have the [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) installed

```
paddle2onnx --model_dir . --model_filename model.pdmodel \
--params_filename model.pdiparams \
--input_shape_dict "{'image': [1, 3, 320, 576], 'scale_factor': [1, 2], 'im_shape': [1, 2]}" \
--save_file fairmot_576_320_v2.onnx \
--opset_version 12 \
--enable_onnx_checker True
```

For more details about how to convert Paddle models to ONNX, please see [Export ONNX Model](../../../deploy/EXPORT_ONNX_MODEL_en.md).

## Use the ONNX model for inference

Once the Paddle model has been converted to ONNX format, we can then use it with OpenVINO inference engine to do the prediction.

*<sub>Please make sure you have the OpenVINO installed, here is the [instruction for installation](https://docs.openvino.ai/cn/latest/openvino_docs_install_guides_installing_openvino_linux.html).<sub>*

1. ### Get the execution network

So the 1st thing to do here is to get an execution network which can be used later to do the inference.

Here is the code.

```
def get_net():
    ie = IECore()
    model_path = root_path / "PaddleDetection/FairMot/fairmot_576_320_v3.onnx"
    net = ie.read_network(model= str(model_path))
    exec_net = ie.load_network(network=net, device_name="CPU")
    return net, exec_net
```

2. ### Preprocessing

Every AI model has its own steps of preprocessing, let's have a look how to do it for the FairMOT model:

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

3. ### Prediction

After we have done all the load network and preprocessing, it finally comes to the stage of prediction.


```
def predict(exec_net, input):
    result = exec_net.infer(input)
    return result
```

You might be surprised to see the very exciting stage this small. Hang on there, the next stage is actually big again.

4. ### Post-processing

MOT(Multi-Object Tracking) is special, not like other AI models which require a few steps of post-processing. Instead, FairMOT requires a special object called tracker, to handle the prediction results. The prediction results are prediction detections and prediction embeddings.

Luckily, PaddleDetection has made this procesure easy for us, it has exported the JDETracker from `ppdet`, so that we do not need to write much code to handle it.

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

5. ### Plot the detections (Optional)

This step is optional. For demo purpose, I just use `plot_tracking_dict()` method to draw all boundary boxes on the image. But you do not need to do this if you don't have the same requirement.

```
online_im = plot_tracking_dict(
    img,
    1,
    online_tlwhs,
    online_ids,
    online_scores,
    frame_id=0)
```

So these are the all steps which you need to follow in order to run FairMOT on your machine.

A companion article which explains in details of this procedure will be released soon and a link to that article will be updated here soon.

To see the full code, please take a look at [Paddle OpenVINO Prediction](./fairmot_onnx_openvino.py).
