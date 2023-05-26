import os
import cv2
import numpy as np
import os.path as osp
from functools import partial
from .metrics import Metric
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon
from ppdet.utils.logger import setup_logger

logger = setup_logger(__name__)

__all__ = [
    'draw_lane', 'discrete_cross_iou', 'continuous_cross_iou', 'interp',
    'culane_metric', 'load_culane_img_data', 'load_culane_data',
    'eval_predictions', "CULaneMetric"
]

LIST_FILE = {
    'train': 'list/train_gt.txt',
    'val': 'list/val.txt',
    'test': 'list/test.txt',
}

CATEGORYS = {
    'normal': 'list/test_split/test0_normal.txt',
    'crowd': 'list/test_split/test1_crowd.txt',
    'hlight': 'list/test_split/test2_hlight.txt',
    'shadow': 'list/test_split/test3_shadow.txt',
    'noline': 'list/test_split/test4_noline.txt',
    'arrow': 'list/test_split/test5_arrow.txt',
    'curve': 'list/test_split/test6_curve.txt',
    'cross': 'list/test_split/test7_cross.txt',
    'night': 'list/test_split/test8_night.txt',
}


def draw_lane(lane, img=None, img_shape=None, width=30):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(
            img, tuple(p1), tuple(p2), color=(255, 255, 255), thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious


def continuous_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    h, w, _ = img_shape
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
    xs = [
        LineString(lane).buffer(
            distance=width / 2., cap_style=1, join_style=2).intersection(image)
        for lane in xs
    ]
    ys = [
        LineString(lane).buffer(
            distance=width / 2., cap_style=1, join_style=2).intersection(image)
        for lane in ys
    ]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = x.intersection(y).area / x.union(y).area

    return ious


def interp(points, n=50):
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T


def culane_metric(pred,
                  anno,
                  width=30,
                  iou_thresholds=[0.5],
                  official=True,
                  img_shape=(590, 1640, 3)):
    _metric = {}
    for thr in iou_thresholds:
        tp = 0
        fp = 0 if len(anno) != 0 else len(pred)
        fn = 0 if len(pred) != 0 else len(anno)
        _metric[thr] = [tp, fp, fn]

    interp_pred = np.array(
        [interp(
            pred_lane, n=5) for pred_lane in pred], dtype=object)  # (4, 50, 2)
    interp_anno = np.array(
        [interp(
            anno_lane, n=5) for anno_lane in anno], dtype=object)  # (4, 50, 2)

    if official:
        ious = discrete_cross_iou(
            interp_pred, interp_anno, width=width, img_shape=img_shape)
    else:
        ious = continuous_cross_iou(
            interp_pred, interp_anno, width=width, img_shape=img_shape)

    row_ind, col_ind = linear_sum_assignment(1 - ious)

    _metric = {}
    for thr in iou_thresholds:
        tp = int((ious[row_ind, col_ind] > thr).sum())
        fp = len(pred) - tp
        fn = len(anno) - tp
        _metric[thr] = [tp, fp, fn]
    return _metric


def load_culane_img_data(path):
    with open(path, 'r') as data_file:
        img_data = data_file.readlines()
    img_data = [line.split() for line in img_data]
    img_data = [list(map(float, lane)) for lane in img_data]
    img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
                for lane in img_data]
    img_data = [lane for lane in img_data if len(lane) >= 2]

    return img_data


def load_culane_data(data_dir, file_list_path):
    with open(file_list_path, 'r') as file_list:
        filepaths = [
            os.path.join(data_dir,
                         line[1 if line[0] == '/' else 0:].rstrip().replace(
                             '.jpg', '.lines.txt'))
            for line in file_list.readlines()
        ]

    data = []
    for path in filepaths:
        img_data = load_culane_img_data(path)
        data.append(img_data)

    return data


def eval_predictions(pred_dir,
                     anno_dir,
                     list_path,
                     iou_thresholds=[0.5],
                     width=30,
                     official=True,
                     sequential=False):
    logger.info('Calculating metric for List: {}'.format(list_path))
    predictions = load_culane_data(pred_dir, list_path)
    annotations = load_culane_data(anno_dir, list_path)
    img_shape = (590, 1640, 3)
    if sequential:
        results = map(partial(
            culane_metric,
            width=width,
            official=official,
            iou_thresholds=iou_thresholds,
            img_shape=img_shape),
                      predictions,
                      annotations)
    else:
        from multiprocessing import Pool, cpu_count
        from itertools import repeat
        with Pool(cpu_count()) as p:
            results = p.starmap(culane_metric,
                                zip(predictions, annotations,
                                    repeat(width),
                                    repeat(iou_thresholds),
                                    repeat(official), repeat(img_shape)))

    mean_f1, mean_prec, mean_recall, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0, 0
    ret = {}
    for thr in iou_thresholds:
        tp = sum(m[thr][0] for m in results)
        fp = sum(m[thr][1] for m in results)
        fn = sum(m[thr][2] for m in results)
        precision = float(tp) / (tp + fp) if tp != 0 else 0
        recall = float(tp) / (tp + fn) if tp != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if tp != 0 else 0
        logger.info('iou thr: {:.2f}, tp: {}, fp: {}, fn: {},'
                    'precision: {}, recall: {}, f1: {}'.format(
                        thr, tp, fp, fn, precision, recall, f1))
        mean_f1 += f1 / len(iou_thresholds)
        mean_prec += precision / len(iou_thresholds)
        mean_recall += recall / len(iou_thresholds)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        ret[thr] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    if len(iou_thresholds) > 2:
        logger.info(
            'mean result, total_tp: {}, total_fp: {}, total_fn: {},'
            'precision: {}, recall: {}, f1: {}'.format(
                total_tp, total_fp, total_fn, mean_prec, mean_recall, mean_f1))
        ret['mean'] = {
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'Precision': mean_prec,
            'Recall': mean_recall,
            'F1': mean_f1
        }
    return ret


class CULaneMetric(Metric):
    def __init__(self,
                 cfg,
                 output_eval=None,
                 split="test",
                 dataset_dir="dataset/CULane/"):
        super(CULaneMetric, self).__init__()
        self.output_eval = "evaluation" if output_eval is None else output_eval
        self.dataset_dir = dataset_dir
        self.split = split
        self.list_path = osp.join(dataset_dir, LIST_FILE[split])
        self.predictions = []
        self.img_names = []
        self.lanes = []
        self.eval_results = {}
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.predictions = []
        self.img_names = []
        self.lanes = []
        self.eval_results = {}

    def get_prediction_string(self, pred):
        ys = np.arange(270, 590, 8) / self.cfg.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.cfg.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.cfg.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join([
                '{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)
            ])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def accumulate(self):
        loss_lines = [[], [], [], []]
        for idx, pred in enumerate(self.predictions):
            output_dir = os.path.join(self.output_eval,
                                      os.path.dirname(self.img_names[idx]))
            output_filename = os.path.basename(self.img_names[
                idx])[:-3] + 'lines.txt'
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred)

            # store loss lines
            lanes = self.lanes[idx]
            if len(lanes) - len(pred) in [1, 2, 3, 4]:
                loss_lines[len(lanes) - len(pred) - 1].append(self.img_names[
                    idx])

            with open(os.path.join(output_dir, output_filename),
                      'w') as out_file:
                out_file.write(output)

        for i, names in enumerate(loss_lines):
            with open(
                    os.path.join(output_dir, 'loss_{}_lines.txt'.format(i + 1)),
                    'w') as f:
                for name in names:
                    f.write(name + '\n')

        for cate, cate_file in CATEGORYS.items():
            result = eval_predictions(
                self.output_eval,
                self.dataset_dir,
                os.path.join(self.dataset_dir, cate_file),
                iou_thresholds=[0.5],
                official=True)

        result = eval_predictions(
            self.output_eval,
            self.dataset_dir,
            self.list_path,
            iou_thresholds=np.linspace(0.5, 0.95, 10),
            official=True)
        self.eval_results['F1@50'] = result[0.5]['F1']
        self.eval_results['result'] = result

    def update(self, inputs, outputs):
        assert len(inputs['img_name']) == len(outputs['lanes'])
        self.predictions.extend(outputs['lanes'])
        self.img_names.extend(inputs['img_name'])
        self.lanes.extend(inputs['lane_line'])

    def log(self):
        logger.info(self.eval_results)

    # abstract method for getting metric results
    def get_results(self):
        return self.eval_results
