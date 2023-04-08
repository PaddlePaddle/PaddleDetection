from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np


class Lane:
    def __init__(self, points=None, invalid_value=-2., metadata=None):
        super(Lane, self).__init__()
        self.curr_iter = 0
        self.points = points
        self.invalid_value = invalid_value
        self.function = InterpolatedUnivariateSpline(points[:, 1],
                                                     points[:, 0],
                                                     k=min(3,
                                                           len(points) - 1))
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01

        self.metadata = metadata or {}

    def __repr__(self):
        return '[Lane]\n' + str(self.points) + '\n[/Lane]'

    def __call__(self, lane_ys):
        lane_xs = self.function(lane_ys)

        lane_xs[(lane_ys < self.min_y) |
                (lane_ys > self.max_y)] = self.invalid_value
        return lane_xs

    def to_array(self, cfg):
        sample_y = cfg.sample_y
        img_w, img_h = cfg.ori_img_w, cfg.ori_img_h
        ys = np.array(sample_y) / float(img_h)
        xs = self(ys)
        valid_mask = (xs >= 0) & (xs < 1)
        lane_xs = xs[valid_mask] * img_w
        lane_ys = ys[valid_mask] * img_h
        lane = np.concatenate((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                              axis=1)
        return lane

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_iter < len(self.points):
            self.curr_iter += 1
            return self.points[self.curr_iter - 1]
        self.curr_iter = 0
        raise StopIteration
