import os
import sys
import cv2
import numpy as np
import argparse


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video_file",
        type=str,
        default=None,
        help="Path of video file, `video_file` or `camera_id` has a highest priority."
    )
    parser.add_argument(
        '--region_polygon',
        nargs='+',
        type=int,
        default=[],
        help="Clockwise point coords (x0,y0,x1,y1...) of polygon of area when "
        "do_break_in_counting. Note that only support single-class MOT and "
        "the video should be taken by a static camera.")
    return parser


def get_video_info(video_file, region_polygon):
    entrance = []
    assert len(region_polygon
               ) % 2 == 0, "region_polygon should be pairs of coords points."
    for i in range(0, len(region_polygon), 2):
        entrance.append([region_polygon[i], region_polygon[i + 1]])

    if not os.path.exists(video_file):
        print("video path '{}' not exists".format(video_file))
        sys.exit(-1)
    capture = cv2.VideoCapture(video_file)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("video width: %d, height: %d" % (width, height))
    np_masks = np.zeros((height, width, 1), np.uint8)

    entrance = np.array(entrance)
    cv2.fillPoly(np_masks, [entrance], 255)

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


if __name__ == "__main__":
    parser = argsparser()
    FLAGS = parser.parse_args()
    get_video_info(FLAGS.video_file, FLAGS.region_polygon)

    # python get_video_info.py --video_file=demo.mp4 --region_polygon 200 200 400 200 300 400 100 400
