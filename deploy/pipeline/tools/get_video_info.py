import os
import sys
import cv2
import numpy as np

video_file = 'demo.mp4'
region_polygon = [
    [200, 200],
    [400, 200],
    [300, 400],
    [100, 400],
]  # modify by yourself, at least 3 pairs of points
region_polygon = np.array(region_polygon)

if not os.path.exists(video_file):
    print("video path '{}' not exists".format(video_file))
    sys.exit(-1)
capture = cv2.VideoCapture(video_file)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("video width: %d, height: %d" % (width, height))
np_masks = np.zeros((height, width, 1), np.uint8)
cv2.fillPoly(np_masks, [region_polygon], 255)

fps = int(capture.get(cv2.CAP_PROP_FPS))
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print("video fps: %d, frame_count: %d" % (fps, frame_count))
cnt = 0
while (1):
    ret, frame = capture.read()
    cnt += 1
    if cnt == 3: break

alpha = 0.3
img = np.array(frame).astype('float32')
mask = np_masks[:, :, 0]
color_mask = [0, 0, 255]
idx = np.nonzero(mask)
color_mask = np.array(color_mask)
img[idx[0], idx[1], :] *= 1.0 - alpha
img[idx[0], idx[1], :] += alpha * color_mask
cv2.imwrite('region_vis.jpg', img)

points_info = 'Your region_polygon points are:'
for p in region_polygon:
    points_info += ' {} {}'.format(p[0], p[1])
