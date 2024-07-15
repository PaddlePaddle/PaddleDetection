import numpy as np
import cv2
import os
import sys
import glob
from ppdet.utils.logger import setup_logger
import copy
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
        self.set_hook(cfg)
        self.nms_idx_need_divid_numclass_arch = [
            'FasterRCNN', 'MaskRCNN', 'CascadeRCNN'
        ]
        """
        In these networks, the bbox array shape before nms contain num_class,
        the nms_keep_idx of the bbox need to divide the num_class; 
        """

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

        # set for get extra_data before nms
        trainer.model.use_extra_data = True
        # set for record the bbox index before nms
        if cfg.architecture in ['FasterRCNN', 'MaskRCNN']:
            trainer.model.bbox_post_process.nms.return_index = True
        elif cfg.architecture in ['YOLOv3', 'PPYOLOE', 'PPYOLOEWithAuxHead']:
            if trainer.model.post_process is not None:
                # anchor based YOLOs: YOLOv3,PP-YOLO
                trainer.model.post_process.nms.return_index = True
            else:
                # anchor free YOLOs: PP-YOLOE, PP-YOLOE+
                trainer.model.yolo_head.nms.return_index = True
        elif cfg.architecture == 'BlazeFace' or cfg.architecture == 'SSD':
            trainer.model.post_process.nms.return_index = True
        elif cfg.architecture == 'RetinaNet':
            trainer.model.head.nms.return_index = True
        else:
            print(cfg.architecture + ' is not supported for cam temporarily!')
            sys.exit()
        # Todo: Unify the head/post_process name in each model

        return trainer

    def set_hook(self, cfg):
        # set hook for extraction of featuremaps and grads
        self.target_feats = {}
        self.target_layer_name = cfg.target_feature_layer_name

        # such as trainer.model.backbone, trainer.model.bbox_head.roi_extractor

        def hook(layer, input, output):
            self.target_feats[layer._layer_name_for_hook] = output

        try:
            exec('self.trainer.' + self.target_layer_name +
                 '._layer_name_for_hook = self.target_layer_name')
            # self.trainer.target_layer_name._layer_name_for_hook = self.target_layer_name
            exec('self.trainer.' + self.target_layer_name +
                 '.register_forward_post_hook(hook)')
            # self.trainer.target_layer_name.register_forward_post_hook(hook)
        except:
            print("Error! "
                  "The target_layer_name--" + self.target_layer_name +
                  " is not in model! "
                  "Please check the spelling and "
                  "the network's architecture!")
            sys.exit()

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
        extra_data = inference_result['extra_data']
        """
        Example of Faster_RCNN based architecture:
            extra_data: {'scores': tensor with shape [num_of_bboxes_before_nms, num_classes], for example: [1000, 80]
                       'nms_keep_idx': tensor with shape [num_of_bboxes_after_nms, 1], for example: [300, 1]
                      }
        Example of YOLOv3 based architecture:
            extra_data: {'scores': tensor with shape [1, num_classes, num_of_yolo_bboxes_before_nms], #for example: [1, 80, 8400]
                       'nms_keep_idx': tensor with shape [num_of_yolo_bboxes_after_nms, 1], # for example: [300, 1]
                      }
        """

        # array index of the predicted bbox before nms
        if self.cfg.architecture in self.nms_idx_need_divid_numclass_arch:
            # some network's bbox array shape before nms may be like [num_of_bboxes_before_nms, num_classes, 4],
            # we need to divide num_classes to get the before_nms_index；
            # currently, only include the rcnn architectures （fasterrcnn, maskrcnn, cascadercnn);
            before_nms_indexes = extra_data['nms_keep_idx'].cpu().numpy(
            ) // self.num_class  # num_class
        else:
            before_nms_indexes = extra_data['nms_keep_idx'].cpu().numpy()

        # Calculate and visualize the heatmap of per predict bbox
        for index, target_bbox in enumerate(inference_result['bbox']):
            # target_bbox: [cls, score, x1, y1, x2, y2]
            # filter bboxes with low predicted scores
            if target_bbox[1] < self.FLAGS.draw_threshold:
                continue

            target_bbox_before_nms = int(before_nms_indexes[index])

            if len(extra_data['scores'].shape) == 2:
                score_out = extra_data['scores'][target_bbox_before_nms]
            else:
                score_out = extra_data['scores'][0, :, target_bbox_before_nms]
            """
            There are two kinds array shape of bbox score output :
                1) [num_of_bboxes_before_nms, num_classes], for example: [1000, 80]
                2) [num_of_image, num_classes, num_of_yolo_bboxes_before_nms], for example: [1, 80, 1000]
            """

            # construct one_hot label and do backward to get the gradients
            predicted_label = paddle.argmax(score_out)
            label_onehot = paddle.nn.functional.one_hot(
                predicted_label, num_classes=len(score_out))
            label_onehot = label_onehot.squeeze()
            target = paddle.sum(score_out * label_onehot)
            target.backward(retain_graph=True)


            if 'backbone' in self.target_layer_name or \
                    'neck' in self.target_layer_name: # backbone/neck level feature
                if isinstance(self.target_feats[self.target_layer_name], list):
                    # when the featuremap contains of multiple scales,
                    # take the featuremap of the last scale
                    # Todo: fuse the cam result from multisclae featuremaps
                    if self.target_feats[self.target_layer_name][-1].shape[
                            -1] == 1:
                        """
                        if the last level featuremap is 1x1 size,
                        we take the second last one
                        """
                        cam_grad = self.target_feats[self.target_layer_name][
                            -2].grad.squeeze().cpu().numpy()
                        cam_feat = self.target_feats[self.target_layer_name][
                            -2].squeeze().cpu().numpy()
                    else:
                        cam_grad = self.target_feats[self.target_layer_name][
                            -1].grad.squeeze().cpu().numpy()
                        cam_feat = self.target_feats[self.target_layer_name][
                            -1].squeeze().cpu().numpy()
                else:
                    cam_grad = self.target_feats[
                        self.target_layer_name].grad.squeeze().cpu().numpy()
                    cam_feat = self.target_feats[
                        self.target_layer_name].squeeze().cpu().numpy()
            else:  # roi level feature
                cam_grad = self.target_feats[
                    self.target_layer_name].grad.squeeze().cpu().numpy()[
                        target_bbox_before_nms]
                cam_feat = self.target_feats[self.target_layer_name].squeeze(
                ).cpu().numpy()[target_bbox_before_nms]

            # grad_cam:
            exp = grad_cam(cam_feat, cam_grad)

            if 'backbone' in self.target_layer_name or \
                    'neck' in self.target_layer_name:
                """
                when use backbone/neck featuremap, 
                we first do the cam on whole image, 
                and then set the area outside the predic bbox to 0
                """
                # reshape the cam image to the input image size
                resized_exp = resize_cam(exp, (img.shape[1], img.shape[0]))
                mask = np.zeros((img.shape[0], img.shape[1], 3))
                mask[int(target_bbox[3]):int(target_bbox[5]), int(target_bbox[
                    2]):int(target_bbox[4]), :] = 1
                resized_exp = resized_exp * mask
                # add the bbox cam back to the input image
                overlay_vis = np.uint8(resized_exp * 0.4 + img * 0.6)
            elif 'roi' in self.target_layer_name:
                # get the bbox part of the image
                bbox_img = copy.deepcopy(img[int(target_bbox[3]):int(
                    target_bbox[5]), int(target_bbox[2]):int(target_bbox[
                        4]), :])
                # reshape the cam image to the bbox size
                resized_exp = resize_cam(exp,
                                         (bbox_img.shape[1], bbox_img.shape[0]))
                # add the bbox cam back to the bbox image
                bbox_overlay_vis = np.uint8(resized_exp * 0.4 + bbox_img * 0.6)
                # put the bbox_cam image to the original image
                overlay_vis = copy.deepcopy(img)
                overlay_vis[int(target_bbox[3]):int(target_bbox[5]), int(
                    target_bbox[2]):int(target_bbox[4]), :] = bbox_overlay_vis
            else:
                print(
                    'Only supported cam for  backbone/neck feature and roi feature,  the others are not supported temporarily!'
                )
                sys.exit()

            # put the bbox rectangle on image
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
