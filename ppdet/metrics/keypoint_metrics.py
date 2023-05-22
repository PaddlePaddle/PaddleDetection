# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

import os
import json
from collections import defaultdict, OrderedDict
import numpy as np
import paddle
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ..modeling.keypoint_utils import oks_nms, keypoint_pck_accuracy, keypoint_auc, keypoint_epe
from scipy.io import loadmat, savemat
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = [
    'KeyPointTopDownCOCOEval', 'KeyPointTopDownCOCOWholeBadyHandEval',
    'KeyPointTopDownMPIIEval'
]


class KeyPointTopDownCOCOEval(object):
    """refer to
        https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
        Copyright (c) Microsoft, under the MIT License.
    """

    def __init__(self,
                 anno_file,
                 num_samples,
                 num_joints,
                 output_eval,
                 iou_type='keypoints',
                 in_vis_thre=0.2,
                 oks_thre=0.9,
                 save_prediction_only=False):
        super(KeyPointTopDownCOCOEval, self).__init__()
        self.coco = COCO(anno_file)
        self.num_samples = num_samples
        self.num_joints = num_joints
        self.iou_type = iou_type
        self.in_vis_thre = in_vis_thre
        self.oks_thre = oks_thre
        self.output_eval = output_eval
        self.res_file = os.path.join(output_eval, "keypoints_results.json")
        self.save_prediction_only = save_prediction_only
        self.reset()

    def reset(self):
        self.results = {
            'all_preds': np.zeros(
                (self.num_samples, self.num_joints, 3), dtype=np.float32),
            'all_boxes': np.zeros((self.num_samples, 6)),
            'image_path': []
        }
        self.eval_results = {}
        self.idx = 0

    def update(self, inputs, outputs):
        kpts, _ = outputs['keypoint'][0]

        num_images = inputs['image'].shape[0]
        self.results['all_preds'][self.idx:self.idx + num_images, :, 0:
                                  3] = kpts[:, :, 0:3]
        self.results['all_boxes'][self.idx:self.idx + num_images, 0:2] = inputs[
            'center'].numpy()[:, 0:2] if isinstance(
                inputs['center'], paddle.Tensor) else inputs['center'][:, 0:2]
        self.results['all_boxes'][self.idx:self.idx + num_images, 2:4] = inputs[
            'scale'].numpy()[:, 0:2] if isinstance(
                inputs['scale'], paddle.Tensor) else inputs['scale'][:, 0:2]
        self.results['all_boxes'][self.idx:self.idx + num_images, 4] = np.prod(
            inputs['scale'].numpy() * 200,
            1) if isinstance(inputs['scale'], paddle.Tensor) else np.prod(
                inputs['scale'] * 200, 1)
        self.results['all_boxes'][
            self.idx:self.idx + num_images,
            5] = np.squeeze(inputs['score'].numpy()) if isinstance(
                inputs['score'], paddle.Tensor) else np.squeeze(inputs['score'])
        if isinstance(inputs['im_id'], paddle.Tensor):
            self.results['image_path'].extend(inputs['im_id'].numpy())
        else:
            self.results['image_path'].extend(inputs['im_id'])
        self.idx += num_images

    def _write_coco_keypoint_results(self, keypoints):
        data_pack = [{
            'cat_id': 1,
            'cls': 'person',
            'ann_type': 'keypoints',
            'keypoints': keypoints
        }]
        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        if not os.path.exists(self.output_eval):
            os.makedirs(self.output_eval)
        with open(self.res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
            logger.info(f'The keypoint result is saved to {self.res_file}.')
        try:
            json.load(open(self.res_file))
        except Exception:
            content = []
            with open(self.res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(self.res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))])
            _key_points = _key_points.reshape(_key_points.shape[0], -1)

            result = [{
                'image_id': img_kpts[k]['image'],
                'category_id': cat_id,
                'keypoints': _key_points[k].tolist(),
                'score': img_kpts[k]['score'],
                'center': list(img_kpts[k]['center']),
                'scale': list(img_kpts[k]['scale'])
            } for k in range(len(img_kpts))]
            cat_results.extend(result)

        return cat_results

    def get_final_results(self, preds, all_boxes, img_path):
        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx])
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = preds.shape[1]
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))],
                           oks_thre)

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(oks_nmsed_kpts)

    def accumulate(self):
        self.get_final_results(self.results['all_preds'],
                               self.results['all_boxes'],
                               self.results['image_path'])
        if self.save_prediction_only:
            logger.info(f'The keypoint result is saved to {self.res_file} '
                        'and do not evaluate the mAP.')
            return
        coco_dt = self.coco.loadRes(self.res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        keypoint_stats = []
        for ind in range(len(coco_eval.stats)):
            keypoint_stats.append((coco_eval.stats[ind]))
        self.eval_results['keypoint'] = keypoint_stats

    def log(self):
        if self.save_prediction_only:
            return
        stats_names = [
            'AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]
        num_values = len(stats_names)
        print(' '.join(['| {}'.format(name) for name in stats_names]) + ' |')
        print('|---' * (num_values + 1) + '|')

        print(' '.join([
            '| {:.3f}'.format(value) for value in self.eval_results['keypoint']
        ]) + ' |')

    def get_results(self):
        return self.eval_results


class KeyPointTopDownCOCOWholeBadyHandEval(object):
    def __init__(self,
                 anno_file,
                 num_samples,
                 num_joints,
                 output_eval,
                 save_prediction_only=False):
        super(KeyPointTopDownCOCOWholeBadyHandEval, self).__init__()
        self.coco = COCO(anno_file)
        self.num_samples = num_samples
        self.num_joints = num_joints
        self.output_eval = output_eval
        self.res_file = os.path.join(output_eval, "keypoints_results.json")
        self.save_prediction_only = save_prediction_only
        self.parse_dataset()
        self.reset()

    def parse_dataset(self):
        gt_db = []
        num_joints = self.num_joints
        coco = self.coco
        img_ids = coco.getImgIds()
        for img_id in img_ids:
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = coco.loadAnns(ann_ids)

            for obj in objs:
                for type in ['left', 'right']:
                    if (obj[f'{type}hand_valid'] and
                            max(obj[f'{type}hand_kpts']) > 0):

                        joints = np.zeros((num_joints, 3), dtype=np.float32)
                        joints_vis = np.zeros((num_joints, 3), dtype=np.float32)

                        keypoints = np.array(obj[f'{type}hand_kpts'])
                        keypoints = keypoints.reshape(-1, 3)
                        joints[:, :2] = keypoints[:, :2]
                        joints_vis[:, :2] = np.minimum(1, keypoints[:, 2:3])

                        gt_db.append({
                            'bbox': obj[f'{type}hand_box'],
                            'gt_joints': joints,
                            'joints_vis': joints_vis,
                        })
        self.db = gt_db

    def reset(self):
        self.results = {
            'preds': np.zeros(
                (self.num_samples, self.num_joints, 3), dtype=np.float32),
        }
        self.eval_results = {}
        self.idx = 0

    def update(self, inputs, outputs):
        kpts, _ = outputs['keypoint'][0]
        num_images = inputs['image'].shape[0]
        self.results['preds'][self.idx:self.idx + num_images, :, 0:
                              3] = kpts[:, :, 0:3]
        self.idx += num_images

    def accumulate(self):
        self.get_final_results(self.results['preds'])
        if self.save_prediction_only:
            logger.info(f'The keypoint result is saved to {self.res_file} '
                        'and do not evaluate the mAP.')
            return

        self.eval_results = self.evaluate(self.res_file, ('PCK', 'AUC', 'EPE'))

    def get_final_results(self, preds):
        kpts = []
        for idx, kpt in enumerate(preds):
            kpts.append({'keypoints': kpt.tolist()})

        self._write_keypoint_results(kpts)

    def _write_keypoint_results(self, keypoints):
        if not os.path.exists(self.output_eval):
            os.makedirs(self.output_eval)
        with open(self.res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)
            logger.info(f'The keypoint result is saved to {self.res_file}.')
        try:
            json.load(open(self.res_file))
        except Exception:
            content = []
            with open(self.res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(self.res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def log(self):
        if self.save_prediction_only:
            return
        for item, value in self.eval_results.items():
            print("{} : {}".format(item, value))

    def get_results(self):
        return self.eval_results

    def evaluate(self, res_file, metrics, pck_thr=0.2, auc_nor=30):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'AUC', 'EPE'.
            pck_thr (float): PCK threshold, default as 0.2.
            auc_nor (float): AUC normalization factor, default as 30 pixel.

        Returns:
            List: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)

        outputs = []
        gts = []
        masks = []
        threshold_bbox = []

        for pred, item in zip(preds, self.db):
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['gt_joints'])[:, :-1])
            masks.append((np.array(item['joints_vis'])[:, 0]) > 0)
            if 'PCK' in metrics:
                bbox = np.array(item['bbox'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))

        outputs = np.array(outputs)
        gts = np.array(gts)
        masks = np.array(masks)
        threshold_bbox = np.array(threshold_bbox)

        if 'PCK' in metrics:
            _, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
                                              threshold_bbox)
            info_str.append(('PCK', pck))

        if 'AUC' in metrics:
            info_str.append(('AUC', keypoint_auc(outputs, gts, masks, auc_nor)))

        if 'EPE' in metrics:
            info_str.append(('EPE', keypoint_epe(outputs, gts, masks)))

        name_value = OrderedDict(info_str)

        return name_value


class KeyPointTopDownMPIIEval(object):
    def __init__(self,
                 anno_file,
                 num_samples,
                 num_joints,
                 output_eval,
                 oks_thre=0.9,
                 save_prediction_only=False):
        super(KeyPointTopDownMPIIEval, self).__init__()
        self.ann_file = anno_file
        self.res_file = os.path.join(output_eval, "keypoints_results.json")
        self.save_prediction_only = save_prediction_only
        self.reset()

    def reset(self):
        self.results = []
        self.eval_results = {}
        self.idx = 0

    def update(self, inputs, outputs):
        kpts, _ = outputs['keypoint'][0]

        num_images = inputs['image'].shape[0]
        results = {}
        results['preds'] = kpts[:, :, 0:3]
        results['boxes'] = np.zeros((num_images, 6))
        results['boxes'][:, 0:2] = inputs['center'].numpy()[:, 0:2]
        results['boxes'][:, 2:4] = inputs['scale'].numpy()[:, 0:2]
        results['boxes'][:, 4] = np.prod(inputs['scale'].numpy() * 200, 1)
        results['boxes'][:, 5] = np.squeeze(inputs['score'].numpy())
        results['image_path'] = inputs['image_file']

        self.results.append(results)

    def accumulate(self):
        self._mpii_keypoint_results_save()
        if self.save_prediction_only:
            logger.info(f'The keypoint result is saved to {self.res_file} '
                        'and do not evaluate the mAP.')
            return

        self.eval_results = self.evaluate(self.results)

    def _mpii_keypoint_results_save(self):
        results = []
        for res in self.results:
            if len(res) == 0:
                continue
            result = [{
                'preds': res['preds'][k].tolist(),
                'boxes': res['boxes'][k].tolist(),
                'image_path': res['image_path'][k],
            } for k in range(len(res))]
            results.extend(result)
        with open(self.res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
            logger.info(f'The keypoint result is saved to {self.res_file}.')

    def log(self):
        if self.save_prediction_only:
            return
        for item, value in self.eval_results.items():
            print("{} : {}".format(item, value))

    def get_results(self):
        return self.eval_results

    def evaluate(self, outputs, savepath=None):
        """Evaluate PCKh for MPII dataset. refer to
        https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
        Copyright (c) Microsoft, under the MIT License.

        Args:
            outputs(list(preds, boxes)):

                * preds (np.ndarray[N,K,3]): The first two dimensions are
                  coordinates, score is the third dimension of the array.
                * boxes (np.ndarray[N,6]): [center[0], center[1], scale[0]
                  , scale[1],area, score]

        Returns:
            dict: PCKh for each joint
        """

        kpts = []
        for output in outputs:
            preds = output['preds']
            batch_size = preds.shape[0]
            for i in range(batch_size):
                kpts.append({'keypoints': preds[i]})

        preds = np.stack([kpt['keypoints'] for kpt in kpts])

        # convert 0-based index to 1-based index,
        # and get the first two dimensions.
        preds = preds[..., :2] + 1.0

        if savepath is not None:
            pred_file = os.path.join(savepath, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(
            os.path.dirname(self.ann_file), 'mpii_gt_val.mat')
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = headsizes * np.ones((len(uv_err), 1), dtype=np.float32)
        scaled_uv_err = uv_err / scale
        scaled_uv_err = scaled_uv_err * jnt_visible
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = (scaled_uv_err <= threshold) * jnt_visible
        PCKh = 100. * np.sum(less_than_threshold, axis=1) / jnt_count

        # save
        rng = np.arange(0, 0.5 + 0.01, 0.01)
        pckAll = np.zeros((len(rng), 16), dtype=np.float32)

        for r, threshold in enumerate(rng):
            less_than_threshold = (scaled_uv_err <= threshold) * jnt_visible
            pckAll[r, :] = 100. * np.sum(less_than_threshold,
                                         axis=1) / jnt_count

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [  #noqa
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('PCKh', np.sum(PCKh * jnt_ratio)),
            ('PCKh@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts
