from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import paddle.vision.transforms as T
from openvino.inference_engine import IECore
from ppdet.modeling.mot.tracker import JDETracker
from ppdet.modeling.mot.visualization import plot_tracking_dict

root_path = Path(__file__).parent
target_height = 320
target_width = 576

# -------------------------------
def get_net():
    ie = IECore()
    model_path = root_path / "fairmot_576_320_v3.onnx"
    net = ie.read_network(model= str(model_path))
    exec_net = ie.load_network(network=net, device_name="CPU")
    return net, exec_net

def get_output_names(net):
    output_names = [key for key in net.outputs]
    return output_names

def prepare_input():
    transforms = [
        T.Resize(size=(target_height, target_width)), 
        T.Normalize(mean=(0,0,0), std=(1,1,1), data_format='HWC', to_rgb= True),
        T.Transpose()
    ]

    img_file = root_path / "street.jpeg"
    img = cv2.imread(str(img_file))
    normalized_img = T.Compose(transforms)(img)
    normalized_img = normalized_img.astype(np.float32, copy=False) / 255.0

    # add an new axis in front
    img_input = normalized_img[np.newaxis, :]
    # scale_factor is calculated as: im_shape / original_im_shape
    h_scale = target_height / img.shape[0]
    w_scale = target_width / img.shape[1]
    input = {"image": img_input, "im_shape": [target_height, target_width], "scale_factor": [h_scale, w_scale]}
    return input, img

def predict(exec_net, input):
    result = exec_net.infer(input)
    return result

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

# -------------------------------
net, exec_net = get_net()
output_names = get_output_names(net)
del net

input, img = prepare_input()
result = predict(exec_net, input)

pred_dets = result[output_names[0]]
pred_embs = result[output_names[1]]

processed_img = postprocess(pred_dets, pred_embs)
tracked_img_file_path = root_path / "tracked.jpg"
cv2.imwrite(str(tracked_img_file_path), processed_img)
