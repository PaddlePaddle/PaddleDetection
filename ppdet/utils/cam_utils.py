import numpy as np
import cv2
import os
import sys
import glob
from ppdet.utils.logger import setup_logger
logger = setup_logger('ppdet_cam')

import paddle
from ppdet.engine import Trainer


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def compute_ious(boxes1, boxes2):
    """[Compute pairwise IOU matrix for given two sets of boxes]

        Args:
            boxes1 ([numpy ndarray with shape N,4]): [representing bounding boxes with format (xmin,ymin,xmax,ymax)]
            boxes2 ([numpy ndarray with shape M,4]): [representing bounding boxes with format (xmin,ymin,xmax,ymax)]
        Returns:
            pairwise IOU maxtrix with shape (N,M)，where the value at ith row jth column hold the iou between ith
            box and jth box from box1 and box2 respectively.
    """
    lu = np.maximum(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # lu with shape N,M,2 ; boxes1[:,None,:2] with shape (N,1,2) boxes2 with shape(M,2)
    rd = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # rd same to lu
    intersection_wh = np.maximum(0.0, rd - lu)
    intersection_area = intersection_wh[:, :,
                                        0] * intersection_wh[:, :,
                                                             1]  # with shape (N,M)
    boxes1_wh = np.maximum(0.0, boxes1[:, 2:] - boxes1[:, :2])
    boxes1_area = boxes1_wh[:, 0] * boxes1_wh[:, 1]  # with shape (N,)
    boxes2_wh = np.maximum(0.0, boxes2[:, 2:] - boxes2[:, :2])
    boxes2_area = boxes2_wh[:, 0] * boxes2_wh[:, 1]  # with shape (M,)
    union_area = np.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area,
        1e-8)  # with shape (N,M)
    ious = np.clip(intersection_area / union_area, 0.0, 1.0)
    return ious


def grad_cam(feat, grad):
    """

    Args:
        feat:  CxHxW
        grad:  CxHxW

    Returns:
           cam: HxW
    """
    exp = (feat * grad.mean((1, 2), keepdims=True)).mean(axis=0)
    exp = np.maximum(-exp, 0)
    return exp


def resize_cam(explanation, resize_shape) -> np.ndarray:
    """

    Args:
        explanation: (width, height)
        resize_shape: (width, height)

    Returns:

    """
    assert len(explanation.shape) == 2, f"{explanation.shape}. " \
                                        f"Currently support 2D explanation results for visualization. " \
                                        "Reduce higher dimensions to 2D for visualization."

    explanation = (explanation - explanation.min()) / (
        explanation.max() - explanation.min())

    explanation = cv2.resize(explanation, resize_shape)
    explanation = np.uint8(255 * explanation)
    explanation = cv2.applyColorMap(explanation, cv2.COLORMAP_JET)
    explanation = cv2.cvtColor(explanation, cv2.COLOR_BGR2RGB)

    return explanation


class BBoxCAM:
    def __init__(self, FLAGS, cfg):
        self.FLAGS = FLAGS
        self.cfg = cfg
        # build model
        self.trainer = self.build_trainer(cfg)
        # num_class
        self.num_class = cfg.num_classes
        # set hook for extraction of featuremaps and grads
        self.set_hook()

        # cam image output_dir
        try:
            os.makedirs(FLAGS.cam_out)
        except:
            print('Path already exists.')
            pass

    def build_trainer(self, cfg):
        # build trainer
        trainer = Trainer(cfg, mode='test')
        # load weights
        trainer.load_weights(cfg.weights)

        # record the bbox index before nms
        # Todo: hard code for nms return index
        if cfg.architecture == 'FasterRCNN':
            trainer.model.bbox_post_process.nms.return_index = True
        elif cfg.architecture == 'YOLOv3':
            if trainer.model.post_process is not None:
                # anchor based YOLOs: YOLOv3,PP-YOLO
                trainer.model.post_process.nms.return_index = True
            else:
                # anchor free YOLOs: PP-YOLOE, PP-YOLOE+
                trainer.model.yolo_head.nms.return_index = True
        else:
            print(
                'Only supported cam for faster_rcnn based and yolov3 based architecture for now,  the others are not supported temporarily!'
            )
            sys.exit()

        return trainer

    def set_hook(self):
        # set hook for extraction of featuremaps and grads
        self.target_feats = {}
        self.target_layer_name = 'trainer.model.backbone'

        def hook(layer, input, output):
            self.target_feats[layer._layer_name_for_hook] = output

        self.trainer.model.backbone._layer_name_for_hook = self.target_layer_name
        self.trainer.model.backbone.register_forward_post_hook(hook)

    def get_bboxes(self):
        # get inference images
        images = get_test_images(self.FLAGS.infer_dir, self.FLAGS.infer_img)

        # inference
        result = self.trainer.predict(
            images,
            draw_threshold=self.FLAGS.draw_threshold,
            output_dir=self.FLAGS.output_dir,
            save_results=self.FLAGS.save_results,
            visualize=False)[0]
        return result

    def get_bboxes_cams(self):
        # Get the bboxes prediction(after nms result) of the input
        inference_result = self.get_bboxes()

        # read input image
        # Todo: Support folder multi-images process
        from PIL import Image
        img = np.array(Image.open(self.cfg.infer_img))

        # data for calaulate bbox grad_cam
        cam_data = inference_result['cam_data']
        """
        if Faster_RCNN based architecture:
            cam_data: {'scores': tensor with shape [num_of_bboxes_before_nms, num_classes], for example: [1000, 80]
                       'before_nms_indexes': tensor with shape [num_of_bboxes_after_nms, 1], for example: [300, 1]
                      }
        elif YOLOv3 based architecture:
            cam_data: {'scores': tensor with shape [1, num_classes, num_of_yolo_bboxes_before_nms], #for example: [1, 80, 8400]
                       'before_nms_indexes': tensor with shape [num_of_yolo_bboxes_after_nms, 1], # for example: [300, 1]
                      }
        """

        # array index of the predicted bbox before nms
        if self.cfg.architecture == 'FasterRCNN':
            # the bbox array shape of FasterRCNN before nms is [num_of_bboxes_before_nms, num_classes, 4],
            # we need to divide num_classes to get the before_nms_index；
            before_nms_indexes = cam_data['before_nms_indexes'].cpu().numpy(
            ) // self.num_class  # num_class
        elif self.cfg.architecture == 'YOLOv3':
            before_nms_indexes = cam_data['before_nms_indexes'].cpu().numpy()
        else:
            print(
                'Only supported cam for faster_rcnn based and yolov3 based architecture for now,  the others are not supported temporarily!'
            )
            sys.exit()

        # Calculate and visualize the heatmap of per predict bbox
        for index, target_bbox in enumerate(inference_result['bbox']):
            # target_bbox: [cls, score, x1, y1, x2, y2]
            # filter bboxes with low predicted scores
            if target_bbox[1] < self.FLAGS.draw_threshold:
                continue

            target_bbox_before_nms = int(before_nms_indexes[index])

            # bbox score vector
            if self.cfg.architecture == 'FasterRCNN':
                # the shape of faster_rcnn scores tensor is
                # [num_of_bboxes_before_nms, num_classes], for example: [1000, 80]
                score_out = cam_data['scores'][target_bbox_before_nms]
            elif self.cfg.architecture == 'YOLOv3':
                # the shape of yolov3 scores tensor is
                # [1, num_classes, num_of_yolo_bboxes_before_nms]
                score_out = cam_data['scores'][0, :, target_bbox_before_nms]
            else:
                print(
                    'Only supported cam for faster_rcnn based and yolov3 based architecture for now,  the others are not supported temporarily!'
                )
                sys.exit()

            # construct one_hot label and do backward to get the gradients
            predicted_label = paddle.argmax(score_out)
            label_onehot = paddle.nn.functional.one_hot(
                predicted_label, num_classes=len(score_out))
            label_onehot = label_onehot.squeeze()
            target = paddle.sum(score_out * label_onehot)
            target.backward(retain_graph=True)

            if isinstance(self.target_feats[self.target_layer_name], list):
                # when the backbone output contains features of multiple scales,
                # take the featuremap of the last scale
                # Todo: fuse the cam result from multisclae featuremaps
                backbone_grad = self.target_feats[self.target_layer_name][
                    -1].grad.squeeze().cpu().numpy()
                backbone_feat = self.target_feats[self.target_layer_name][
                    -1].squeeze().cpu().numpy()
            else:
                backbone_grad = self.target_feats[
                    self.target_layer_name].grad.squeeze().cpu().numpy()
                backbone_feat = self.target_feats[
                    self.target_layer_name].squeeze().cpu().numpy()

            # grad_cam:
            exp = grad_cam(backbone_feat, backbone_grad)

            # reshape the cam image to the input image size
            resized_exp = resize_cam(exp, (img.shape[1], img.shape[0]))

            # set the area outside the predic bbox to 0
            mask = np.zeros((img.shape[0], img.shape[1], 3))
            mask[int(target_bbox[3]):int(target_bbox[5]), int(target_bbox[2]):
                 int(target_bbox[4]), :] = 1
            resized_exp = resized_exp * mask

            # add the bbox cam back to the input image
            overlay_vis = np.uint8(resized_exp * 0.4 + img * 0.6)
            cv2.rectangle(
                overlay_vis, (int(target_bbox[2]), int(target_bbox[3])),
                (int(target_bbox[4]), int(target_bbox[5])), (0, 0, 255), 2)

            # save visualization result
            cam_image = Image.fromarray(overlay_vis)
            cam_image.save(self.FLAGS.cam_out + '/' + str(index) + '.jpg')

            # clear gradients after each bbox grad_cam
            target.clear_gradient()
            for n, v in self.trainer.model.named_sublayers():
                v.clear_gradients()
