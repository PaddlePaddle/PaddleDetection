from ppdet.core.workspace import register, serializable
import cv2
import os
import tarfile
import numpy as np
import os.path as osp
from ppdet.data.source.dataset import DetDataset
from imgaug.augmentables.lines import LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from ppdet.data.culane_utils import lane_to_linestrings
import pickle as pkl
from ppdet.utils.logger import setup_logger
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from .dataset import DetDataset, _make_dataset, _is_valid_file
from ppdet.utils.download import download_dataset

logger = setup_logger(__name__)


@register
@serializable
class CULaneDataSet(DetDataset):
    def __init__(
            self,
            dataset_dir,
            cut_height,
            list_path,
            split='train',
            data_fields=['image'],
            video_file=None,
            frame_rate=-1, ):
        super(CULaneDataSet, self).__init__(
            dataset_dir=dataset_dir,
            cut_height=cut_height,
            split=split,
            data_fields=data_fields)
        self.dataset_dir = dataset_dir
        self.list_path = osp.join(dataset_dir, list_path)
        self.cut_height = cut_height
        self.data_fields = data_fields
        self.split = split
        self.training = 'train' in split
        self.data_infos = []
        self.video_file = video_file
        self.frame_rate = frame_rate
        self._imid2path = {}
        self.predict_dir = None

    def __len__(self):
        return len(self.data_infos)

    def check_or_download_dataset(self):
        if not osp.exists(self.dataset_dir):
            download_dataset("dataset", dataset="culane")
            # extract .tar files in self.dataset_dir
            for fname in os.listdir(self.dataset_dir):
                logger.info("Decompressing {}...".format(fname))
                # ignore .* files
                if fname.startswith('.'):
                    continue
                if fname.find('.tar.gz') >= 0:
                    with tarfile.open(osp.join(self.dataset_dir, fname)) as tf:
                        tf.extractall(path=self.dataset_dir)
        logger.info("Dataset files are ready.")

    def parse_dataset(self):
        logger.info('Loading CULane annotations...')
        if self.predict_dir is not None:
            logger.info('switch to predict mode')
            return
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

        anno_path = img_path[:
                             -3] + 'lines.txt'  # remove sufix jpg and add lines.txt
        with open(anno_path, 'r') as anno_file:
            data = [
                list(map(float, line.split())) for line in anno_file.readlines()
            ]
        lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)
                  if lane[i] >= 0 and lane[i + 1] >= 0] for lane in data]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [lane for lane in lanes
                 if len(lane) > 2]  # remove lanes with less than 2 points

        lanes = [sorted(
            lane, key=lambda x: x[1]) for lane in lanes]  # sort by y
        infos['lanes'] = lanes

        return infos

    def set_images(self, images):
        self.predict_dir = images
        self.data_infos = self._load_images()

    def _find_images(self):
        predict_dir = self.predict_dir
        if not isinstance(predict_dir, Sequence):
            predict_dir = [predict_dir]
        images = []
        for im_dir in predict_dir:
            if os.path.isdir(im_dir):
                im_dir = os.path.join(self.predict_dir, im_dir)
                images.extend(_make_dataset(im_dir))
            elif os.path.isfile(im_dir) and _is_valid_file(im_dir):
                images.append(im_dir)
        return images

    def _load_images(self):
        images = self._find_images()
        ct = 0
        records = []
        for image in images:
            assert image != '' and os.path.isfile(image), \
                    "Image {} not found".format(image)
            if self.sample_num > 0 and ct >= self.sample_num:
                break
            rec = {
                'im_id': np.array([ct]),
                "img_path": os.path.abspath(image),
                "img_name": os.path.basename(image),
                "lanes": []
            }
            self._imid2path[ct] = image
            ct += 1
            records.append(rec)
        assert len(records) > 0, "No image file found"
        return records

    def get_imid2path(self):
        return self._imid2path

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img = cv2.imread(data_info['img_path'])
        img = img[self.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'image': img})
        img_org = sample['image']

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

            sample['mask'] = SegmentationMapsOnImage(
                sample['mask'], shape=img_org.shape)

        sample['full_img_path'] = data_info['img_path']
        sample['img_name'] = data_info['img_name']
        sample['im_id'] = np.array([idx])

        sample['image'] = sample['image'].copy().astype(np.uint8)
        sample['lanes'] = lane_to_linestrings(sample['lanes'])
        sample['lanes'] = LineStringsOnImage(
            sample['lanes'], shape=img_org.shape)
        sample['seg'] = np.zeros(img_org.shape)

        return sample
