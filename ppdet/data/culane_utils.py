import math
import numpy as np
from imgaug.augmentables.lines import LineString
from scipy.interpolate import InterpolatedUnivariateSpline


def lane_to_linestrings(lanes):
    lines = []
    for lane in lanes:
        lines.append(LineString(lane))

    return lines


def linestrings_to_lanes(lines):
    lanes = []
    for line in lines:
        lanes.append(line.coords)

    return lanes


def sample_lane(points, sample_ys, img_w):
    # this function expects the points to be sorted
    points = np.array(points)
    if not np.all(points[1:, 1] < points[:-1, 1]):
        raise Exception('Annotaion points have to be sorted')
    x, y = points[:, 0], points[:, 1]

    # interpolate points inside domain
    assert len(points) > 1
    interp = InterpolatedUnivariateSpline(
        y[::-1], x[::-1], k=min(3, len(points) - 1))
    domain_min_y = y.min()
    domain_max_y = y.max()
    sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y) & (
        sample_ys <= domain_max_y)]
    assert len(sample_ys_inside_domain) > 0
    interp_xs = interp(sample_ys_inside_domain)

    # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
    two_closest_points = points[:2]
    extrap = np.polyfit(
        two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
    extrap_ys = sample_ys[sample_ys > domain_max_y]
    extrap_xs = np.polyval(extrap, extrap_ys)
    all_xs = np.hstack((extrap_xs, interp_xs))

    # separate between inside and outside points
    inside_mask = (all_xs >= 0) & (all_xs < img_w)
    xs_inside_image = all_xs[inside_mask]
    xs_outside_image = all_xs[~inside_mask]

    return xs_outside_image, xs_inside_image


def filter_lane(lane):
    assert lane[-1][1] <= lane[0][1]
    filtered_lane = []
    used = set()
    for p in lane:
        if p[1] not in used:
            filtered_lane.append(p)
            used.add(p[1])

    return filtered_lane


def transform_annotation(img_w, img_h, max_lanes, n_offsets, offsets_ys,
                         n_strips, strip_size, anno):
    old_lanes = anno['lanes']

    # removing lanes with less than 2 points
    old_lanes = filter(lambda x: len(x) > 1, old_lanes)
    # sort lane points by Y (bottom to top of the image)
    old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
    # remove points with same Y (keep first occurrence)
    old_lanes = [filter_lane(lane) for lane in old_lanes]
    # normalize the annotation coordinates
    old_lanes = [[[x * img_w / float(img_w), y * img_h / float(img_h)]
                  for x, y in lane] for lane in old_lanes]
    # create tranformed annotations
    lanes = np.ones(
        (max_lanes, 2 + 1 + 1 + 2 + n_offsets), dtype=np.float32
    ) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, S+1 coordinates
    lanes_endpoints = np.ones((max_lanes, 2))
    # lanes are invalid by default
    lanes[:, 0] = 1
    lanes[:, 1] = 0
    for lane_idx, lane in enumerate(old_lanes):
        if lane_idx >= max_lanes:
            break

        try:
            xs_outside_image, xs_inside_image = sample_lane(lane, offsets_ys,
                                                            img_w)
        except AssertionError:
            continue
        if len(xs_inside_image) <= 1:
            continue
        all_xs = np.hstack((xs_outside_image, xs_inside_image))
        lanes[lane_idx, 0] = 0
        lanes[lane_idx, 1] = 1
        lanes[lane_idx, 2] = len(xs_outside_image) / n_strips
        lanes[lane_idx, 3] = xs_inside_image[0]

        thetas = []
        for i in range(1, len(xs_inside_image)):
            theta = math.atan(
                i * strip_size /
                (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / math.pi
            theta = theta if theta > 0 else 1 - abs(theta)
            thetas.append(theta)

        theta_far = sum(thetas) / len(thetas)

        # lanes[lane_idx,
        #       4] = (theta_closest + theta_far) / 2  # averaged angle
        lanes[lane_idx, 4] = theta_far
        lanes[lane_idx, 5] = len(xs_inside_image)
        lanes[lane_idx, 6:6 + len(all_xs)] = all_xs
        lanes_endpoints[lane_idx, 0] = (len(all_xs) - 1) / n_strips
        lanes_endpoints[lane_idx, 1] = xs_inside_image[-1]

    new_anno = {
        'label': lanes,
        'old_anno': anno,
        'lane_endpoints': lanes_endpoints
    }
    return new_anno
