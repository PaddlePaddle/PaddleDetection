import numpy as np
import box_utils
import cv2
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.reader import create_reader

category_names = ['1']
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
random_sizes = [32 * i for i in range(10, 20)]
# train_reader = reader.train(416, 1, shuffle=True, mixup_iter=10, random_sizes=random_sizes, use_multiprocessing=False)
cfg = load_config('./configs/yolov3_mobilenet_v1_ncp.yml')
train_reader = create_reader(cfg.TrainReader, 100, cfg)
for i, data in enumerate(train_reader()):
    img, gtboxes, gtlabels, gtscores = data[0]
    print("img shape: ", img.shape)
    c, h, w = img.shape
    real_img = (img.transpose((1, 2, 0)) * stds + means) * 255.0
    real_img = cv2.cvtColor(real_img.astype('uint8'), cv2.COLOR_RGB2BGR)
    cv2.imwrite("output/test{}.jpg".format(i), real_img.astype("uint8"))
    gtboxes = box_utils.box_xywh_to_xyxy(gtboxes)
    gtboxes = box_utils.rescale_box_in_input_image(gtboxes, (h, w), 1.0)
    box_utils.draw_boxes_on_image(
        "output/test{}.jpg".format(i),
        gtboxes,
        gtscores,
        gtlabels,
        category_names,
        score_thresh=0.01)
    if i >= 10:
        break
