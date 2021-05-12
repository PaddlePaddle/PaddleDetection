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

import copy
import os
import json
from collections import OrderedDict
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ..modeling.keypoint_utils import oks_nms

__all__ = ['KeyPointTopDownCOCOEval']


class KeyPointTopDownCOCOEval(object):
    def __init__(self,
                 anno_file,
                 num_samples,
                 num_joints,
                 output_eval,
                 iou_type='keypoints',
                 in_vis_thre=0.2,
                 oks_thre=0.9):
        super(KeyPointTopDownCOCOEval, self).__init__()
        self.coco = COCO(anno_file)
        self.num_samples = num_samples
        self.num_joints = num_joints
        self.iou_type = iou_type
        self.in_vis_thre = in_vis_thre
        self.oks_thre = oks_thre
        self.output_eval = output_eval
        self.res_file = os.path.join(output_eval, "keypoints_results.json")
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
            'center'].numpy()[:, 0:2]
        self.results['all_boxes'][self.idx:self.idx + num_images, 2:4] = inputs[
            'scale'].numpy()[:, 0:2]
        self.results['all_boxes'][self.idx:self.idx + num_images, 4] = np.prod(
            inputs['scale'].numpy() * 200, 1)
        self.results['all_boxes'][self.idx:self.idx + num_images,
                                  5] = np.squeeze(inputs['score'].numpy())
        self.results['image_path'].extend(inputs['im_id'].numpy())

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
