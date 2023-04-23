from ppdet.core.workspace import register,serializable
import cv2
import os
import math
import numpy as np
import os.path as osp
from ppdet.data.source.dataset import DetDataset
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from scipy.interpolate import InterpolatedUnivariateSpline
from ppdet.data.culane_utils import lane_to_linestrings, linestrings_to_lanes, sample_lane, filter_lane, transform_annotation
import pickle as pkl
from ppdet.utils.logger import setup_logger
# Copyright (c) Open-MMLab. All rights reserved.
import functools
import paddle
import ppdet.metrics.culane_metrics as culane_metrics

logger = setup_logger(__name__)


def trainTransforms(img_h, img_w):
    return [
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0)
        ]


def testTransforms(img_h, img_w):
    return [
            dict(name='Resize',
                parameters=dict(size=dict(height=img_h, width=img_w)),
                p=1.0)
         ]

@register
@serializable
class CULaneDataSet(DetDataset):
    def __init__(self,
                dataset_dir,
                img_w,
                img_h,
                cut_height,
                num_points,
                max_lanes,
                list_path,
                split='train',
                data_fields=['image'],
                transforms=None,
                 ):
        super(CULaneDataSet,self).__init__(
            dataset_dir=dataset_dir,
            cut_height=cut_height,
            split=split,
            data_fields=data_fields,
        )
        self.dataset_dir = dataset_dir
        self.list_path = osp.join(dataset_dir, list_path)
        self.cut_height = cut_height
        self.data_fields = data_fields
        self.split = split
        self.training = 'train' in split
        self.img_w = img_w
        self.img_h = img_h
        self.num_points = num_points
        self.n_offsets = num_points
        self.n_strips = num_points - 1
        self.strip_size = self.img_h / self.n_strips
        
        self.max_lanes = max_lanes
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        
        self.data_infos = []

        if self.training:
            transforms = trainTransforms(self.img_h, self.img_w)
        else:
            transforms = testTransforms(self.img_h, self.img_w)

        if transforms is not None:
            img_transforms = []
            for aug in transforms:
                p = aug['p']
                if aug['name'] != 'OneOf':
                    img_transforms.append(
                        iaa.Sometimes(p=p,
                                      then_list=getattr(
                                          iaa,
                                          aug['name'])(**aug['parameters'])))
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf([
                                getattr(iaa,
                                        aug_['name'])(**aug_['parameters'])
                                for aug_ in aug['transforms']
                            ])))
        else:
            img_transforms = []
        self.iaa_transform = iaa.Sequential(img_transforms)

        self.parse_dataset()
    
    def __len__(self):
        return len(self.data_infos)

    def parse_dataset(self):
        logger.info('Loading CULane annotations...')
        # Waiting for the dataset to load is tedious, let's cache it
        os.makedirs('cache', exist_ok=True)
        cache_path = 'cache/culane_paddle_{}.pkl'.format(self.split)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                self.data_infos = pkl.load(cache_file)
                self.max_lanes = max(
                    len(anno['lanes']) for anno in self.data_infos)
                return

        
        with open(self.list_path) as list_file:
            for line in list_file:
                infos = self.load_annotation(line.split())
                self.data_infos.append(infos)
        
        # cache data infos to file
        with open(cache_path, 'wb') as cache_file:
            pkl.dump(self.data_infos, cache_file)

    def load_annotation(self, line):
        infos = {}
        img_line = line[0]
        img_line = img_line[1 if img_line[0] == '/' else 0::]
        img_path = os.path.join(self.dataset_dir, img_line)
        infos['img_name'] = img_line
        infos['img_path'] = img_path
        if len(line) > 1:
            mask_line = line[1]
            mask_line = mask_line[1 if mask_line[0] == '/' else 0::]
            mask_path = os.path.join(self.dataset_dir, mask_line)
            infos['mask_path'] = mask_path

        if len(line) > 2:
            exist_list = [int(l) for l in line[2:]]
            infos['lane_exist'] = np.array(exist_list)

        anno_path = img_path[:-3] + 'lines.txt'  # remove sufix jpg and add lines.txt
        with open(anno_path, 'r') as anno_file:
            data = [
                list(map(float, line.split()))
                for line in anno_file.readlines()
            ]
        lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)
                  if lane[i] >= 0 and lane[i + 1] >= 0] for lane in data]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [lane for lane in lanes
                 if len(lane) > 2]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1])
                 for lane in lanes]  # sort by y
        infos['lanes'] = lanes

        return infos
    
    
    def __getitem__(self,idx):
        data_info = self.data_infos[idx]
        img = cv2.imread(data_info['img_path'])
        img = img[self.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'img': img})

        if self.training:
            label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cut_height:, :]
            sample.update({'mask': label})
            if self.cut_height != 0:
                new_lanes = []
                for i in sample['lanes']:
                    lanes = []
                    for p in i:
                        lanes.append((p[0], p[1] - self.cut_height))
                    new_lanes.append(lanes)
                sample.update({'lanes': new_lanes})
                
        img_org = sample['img']
        line_strings_org = lane_to_linestrings(sample['lanes'])
        line_strings_org = LineStringsOnImage(line_strings_org,
                                              shape=img_org.shape)
        if self.training:
            mask_org = SegmentationMapsOnImage(sample['mask'],
                                                shape=img_org.shape)
            img, line_strings, seg = self.iaa_transform(
                image=img_org.copy().astype(np.uint8),
                line_strings=line_strings_org,
                segmentation_maps=mask_org)
        else:
            img, line_strings = self.iaa_transform(
                image=img_org.copy().astype(np.uint8),
                line_strings=line_strings_org)
        line_strings.clip_out_of_image_()
        new_anno = {'lanes': linestrings_to_lanes(line_strings)}
        
        for i in range(30):
            try:
                annos = transform_annotation(self.img_w,self.img_h,self.max_lanes,self.n_offsets,
                                             self.offsets_ys,self.n_strips,self.strip_size,new_anno)
                label = annos['label']
                lane_endpoints = annos['lane_endpoints']
                break
            except:
                if (i + 1) == 30:
                    logger.critical(
                        'Transform annotation failed 30 times :(')
                    exit()
                    
        
        sample['img'] = img.astype(np.float32) / 255.
        
        sample['lane_line'] = label
        
        sample['lanes_endpoints'] = lane_endpoints
        sample['gt_points'] = new_anno['lanes']
        sample['seg'] = seg.get_arr() if self.training else np.zeros(
            img_org.shape)
        
        sample['img'] = sample['img'].transpose(2,0,1)

        final_sample = {}
        final_sample['img'] = sample['img']
        final_sample['lane_line'] = sample['lane_line']
        final_sample['seg'] = sample['seg']

        final_sample['full_img_path'] = data_info['img_path']
        final_sample['img_name'] = data_info['img_name']
        final_sample['im_id'] = np.array([idx])
        return final_sample