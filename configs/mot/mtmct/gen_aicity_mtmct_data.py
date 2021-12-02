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

import os
import sys
import cv2
import glob


def video2frames(sourceVdo, dstDir):
    videoData = cv2.VideoCapture(sourceVdo)
    count = 0
    while (videoData.isOpened()):
        count += 1
        ret, frame = videoData.read()
        if ret:
            cv2.imwrite(f"{dstDir}/{count:07d}.jpg", frame)
            if count % 20 == 0:
                print(f"{dstDir}/{count:07d}.jpg")
        else:
            break
    videoData.release()


def transSeq(seqs_path, new_root):
    sonCameras = glob.glob(seqs_path + "/*")
    sonCameras.sort()
    for vdoList in sonCameras:
        Seq = vdoList.split('/')[-2]
        Camera = vdoList.split('/')[-1]
        os.system(f"mkdir -p {new_root}/{Seq}/images/{Camera}/img1")

        roi_path = vdoList + '/roi.jpg'
        new_roi_path = f"{new_root}/{Seq}/images/{Camera}"
        os.system(f"cp {roi_path} {new_roi_path}")

        video2frames(f"{vdoList}/vdo.avi",
                     f"{new_root}/{Scd eq}/images/{Camera}/img1")


if __name__ == "__main__":
    seq_path = sys.argv[1]
    new_root = 'aic21mtmct_vehicle'

    seq_name = seq_path.split('/')[-1]
    data_path = seq_path.split('/')[-3]
    os.system(f"mkdir -p {new_root}/{seq_name}/gt")
    os.system(f"cp {data_path}/eval/ground*.txt {new_root}/{seq_name}/gt")

    # extract video frames
    transSeq(seq_path, new_root)
