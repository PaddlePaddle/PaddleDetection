# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np

from ppdet.core.workspace import register, serializable

from .dataset import DataSet
import glob
import json
import logging
logger = logging.getLogger(__name__)


@register
@serializable
class VOCDataSet(DataSet):
    """
    Load dataset with PascalVOC format.

    Notes:
    `anno_path` must contains xml file and image file path for annotations.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): voc annotation file path.
        sample_num (int): number of samples to load, -1 means all.
        use_default_label (bool): whether use the default mapping of
            label to integer index. Default True.
        with_background (bool): whether load background as a class,
            default True.
        label_list (str): if use_default_label is False, will load
            mapping between category and class index.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 use_default_label=True,
                 with_background=True):
        super(VOCDataSet, self).__init__(
            image_dir=image_dir,
            anno_path=anno_path,
            sample_num=sample_num,
            dataset_dir=dataset_dir,
            with_background=with_background)
        # roidbs is list of dict whose structure is:
        # {
        #     'im_file': im_fname, # image file name
        #     'im_id': im_id, # image id
        #     'h': im_h, # height of image
        #     'w': im_w, # width
        #     'is_crowd': is_crowd,
        #     'gt_class': gt_class,
        #     'gt_score': gt_score,
        #     'gt_bbox': gt_bbox,
        #     'difficult': difficult
        # }
        self.roidbs = None
        # 'cname2id' is a dict to map category name to class id
        self.cname2cid = None
        self.use_default_label = use_default_label

    def load_roidb_and_cname2cid(self):
        def poly2xyxy(points):
            xs = []
            ys = []
            for n, p in points.items():
                if n.find('x') >= 0:
                    xs.append(float(p))
                if n.find('y') >= 0:
                    ys.append(float(p))
            return [min(xs), min(ys), max(xs), max(ys)]

        records = []
        ct = 0
        cname2cid = ncp_label(with_background=self.with_background)
        patient_list = glob.glob("{}/patient*".format(self.dataset_dir))
        for p in patient_list:
            images = glob.glob("{}/npy/*.npy".format(p))
            for image in images:
                layer_id = image.split('/')[-1].split('.')[0]
                anno_file = "{}/outputs/{}.json".format(p, layer_id)
                if not os.path.isfile(anno_file):
                    logger.warn("No json file {}".format(anno_file))
                    continue
                anno = json.load(open(anno_file))
                # if not anno['labeled']:
                #     logging.warn("{} not labeld".format(image))

                rec = {}
                rec['im_file'] = image
                # use for test_reader.py
                # rec['im_file'] = image.replace('npy', 'png')
                rec['im_id'] = ct
                if 'size' in anno:
                    rec['h'] = anno['size']['height']
                    rec['w'] = anno['size']['width']

                objs = anno['object']
                # uncomment this if not filter out 0 object sample
                if not objs or len(objs) == 0:
                    continue
                if objs is None:
                    objs = []

                gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
                gt_class = np.zeros((len(objs), 1), dtype=np.int32)
                gt_score = np.ones((len(objs), 1), dtype=np.float32)
                is_crowd = np.zeros((len(objs), 1), dtype=np.int32)
                difficult = np.zeros((len(objs), 1), dtype=np.int32)
                for i, obj in enumerate(objs):
                    if 'bbox' in obj:
                        gt_bbox[i, :] = [
                            float(obj['bbox']['xmin']),
                            float(obj['bbox']['ymin']),
                            float(obj['bbox']['xmax']),
                            float(obj['bbox']['ymax'])
                        ]
                    else:
                        gt_bbox[i, :] = poly2xyxy(obj['polygon'])
                    gt_class[i][0] = int(self.with_background)
                    is_crowd[i][0] = 0
                    difficult[i][0] = 0
                rec['gt_bbox'] = gt_bbox
                rec['gt_class'] = gt_class
                rec['gt_score'] = gt_score
                rec['is_crowd'] = is_crowd
                rec['difficult'] = difficult

                records.append(rec)
                ct += 1
                if self.sample_num > 0 and ct >= self.sample_num:
                    break

        assert len(records) > 0, 'not found any voc record in %s' % (
            self.anno_path)
        logger.info('{} samples in file {}'.format(ct, self.dataset_dir))
        self.roidbs, self.cname2cid = records, cname2cid


def ncp_label(with_background=True):
    labels_map = {'1': 1}
    if not with_background:
        labels_map = {k: v - 1 for k, v in labels_map.items()}
    return labels_map
