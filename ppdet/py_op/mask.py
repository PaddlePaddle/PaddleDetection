import six
import math
import numpy as np
from numba import jit


@jit
def decode(cnts, m):
    v = 0
    mask = []
    for j in range(m):
        for k in range(cnts[j]):
            mask.append(v)
        v = 1 - v
    return mask


#@jit 
def poly2mask(xy, k, h, w):
    scale = 5.
    x = [int(scale * p + 0.5) for p in xy[::2]]
    x = x + [x[0]]
    y = [int(scale * p + 0.5) for p in xy[1::2]]
    y = y + [y[0]]
    m = sum([
        int(max(abs(x[j] - x[j + 1]), abs(y[j] - y[j + 1]))) + int(1)
        for j in range(k)
    ])
    u, v = [], []
    for j in range(k):
        xs = x[j]
        xe = x[j + 1]
        ys = y[j]
        ye = y[j + 1]
        dx = abs(xe - xs)
        dy = abs(ys - ye)
        flip = (dx >= dy and xs > xe) or (dx < dy and ys > ye)
        if flip:
            xs, xe = xe, xs
            ys, ye = ye, ys

        if dx >= dy:
            if (dx == 0):
                assert ye - ys == 0

            s = 0 if dx == 0 else float(ye - ys) / dx
        else:
            if (dy == 0):
                assert xe - xs == 0
            s = 0 if dy == 0 else float(xe - xs) / dy

        if dx >= dy:
            ts = [dx - d if flip else d for d in range(dx + 1)]
            u.extend([xs + t for t in ts])
            v.extend([int(ys + s * t + .5) for t in ts])
        else:
            ts = [dy - d if flip else d for d in range(dy + 1)]
            v.extend([t + ys for t in ts])
            u.extend([int(xs + s * t + .5) for t in ts])

    k = len(u)
    x = np.zeros((k), np.int)
    y = np.zeros((k), np.int)
    m = 0
    for j in six.moves.xrange(1, k):
        if u[j] != u[j - 1]:
            xd = float(u[j] if (u[j] < u[j - 1]) else (u[j] - 1))
            xd = (xd + .5) / scale - .5
            if (math.floor(xd) != xd or xd < 0 or xd > (w - 1)):
                continue
            yd = float(v[j] if v[j] < v[j - 1] else v[j - 1])
            yd = (yd + .5) / scale - .5
            yd = math.ceil(0 if yd < 0 else (h if yd > h else yd))
            x[m] = int(xd)
            y[m] = int(yd)
            m += 1
    k = m
    a = [int(x[i] * h + y[i]) for i in range(k)]
    a.append(h * w)
    a.sort()
    b = [0] + a[:len(a) - 1]
    a = [c - d for (c, d) in zip(a, b)]

    k += 1
    b = [0 for i in range(k)]
    b[0] = a[0]
    m, j = 1, 1
    while (j < k):
        if a[j] > 0:
            b[m] = a[j]
            m += 1
            j += 1
        else:
            j += 1
            if (j < k):
                b[m - 1] += a[j]
                j += 1
    mask = decode(b, m)
    mask = np.array(mask, dtype=np.int).reshape((w, h))
    mask = mask.transpose((1, 0))
    return mask


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for j in range(len(polys)):
        x_min, y_min = 10000000, 10000000
        x_max, y_max = 0, 0
        for i in range(len(polys[j])):
            poly = polys[j][i]
            x0 = min(min(p[::2]) for p in poly)
            x_min = min(x0, x_min)
            y0 = min(min(p[1::2]) for p in poly)
            y_min = min(y0, y_min)
            x1 = max(max(p[::2]) for p in poly)
            x_max = max(x_max, x1)
            y1 = max(max(p[1::2]) for p in poly)
            y_max = max(y1, y_max)
        boxes_from_polys[j, :] = [x_min, y_min, x_max, y_max]
    return boxes_from_polys


@jit
def bbox_overlaps_mask(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) *\
                   (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) -\
                 max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) -\
                     max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    ua = float(
                         (boxes[n, 2] - boxes[n, 0] + 1) *\
                         (boxes[n, 3] - boxes[n, 1] + 1) +\
                         box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps


@jit
def polys_to_mask_wrt_box(polygons, box, M):
    """Convert from the COCO polygon segmentation format to a binary mask
    encoded as a 2D array of data type numpy.float32. The polygon segmentation
    is understood to be enclosed in the given box and rasterized to an M x M
    mask. The resulting mask is therefore of shape (M, M).
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    polygons_norm = []
    i = 0
    for poly in polygons:
        p = np.array(poly, dtype=np.float32)
        p = p.reshape(-1)
        p[0::2] = (p[0::2] - box[0]) * M / w
        p[1::2] = (p[1::2] - box[1]) * M / h
        polygons_norm.append(p)

    mask = []
    for polygons in polygons_norm:
        assert polygons.shape[0] % 2 == 0, polygons.shape
        k = polygons.shape[0] // 2

        one_msk = poly2mask(polygons, k, M, M)
        mask.append(one_msk)

    mask = np.array(mask)
    # Flatten in case polygons was a list
    mask = np.sum(mask, axis=0)
    mask = np.array(mask > 0, dtype=np.float32)
    return mask


@jit
def expand_mask_targets(masks, mask_class_labels, resolution, num_classes):
    """Expand masks from shape (#masks, resolution ** 2)
    to (#masks, #classes * resolution ** 2) to encode class
    specific mask targets.
    """
    assert masks.shape[0] == mask_class_labels.shape[0]
    # Target values of -1 are "don't care" / ignore labels
    mask_targets = -np.ones(
        (masks.shape[0], num_classes * resolution**2), dtype=np.int32)
    for i in range(masks.shape[0]):
        cls = int(mask_class_labels[i])
        start = resolution**2 * cls
        end = start + resolution**2
        # Ignore background instance
        # (only happens when there is no fg samples in an image)
        if cls > 0:
            mask_targets[i, start:end] = masks[i, :]

    return mask_targets
