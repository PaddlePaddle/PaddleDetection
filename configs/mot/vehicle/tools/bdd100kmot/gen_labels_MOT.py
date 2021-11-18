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
import os.path as osp
import numpy as np
import argparse


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BDD100K to MOT format')
    parser.add_argument(
        "--mot_data", default='./bdd100k')
    parser.add_argument("--phase", default='train')
    args = parser.parse_args()

    MOT_data = args.mot_data
    phase = args.phase
    seq_root = osp.join(MOT_data, 'bdd100kmot_vehicle', 'images', phase)
    label_root = osp.join(MOT_data, 'bdd100kmot_vehicle', 'labels_with_ids',
                          phase)
    mkdirs(label_root)
    seqs = [s for s in os.listdir(seq_root)]
    tid_curr = 0
    tid_last = -1

    os.system(f'rm -r {MOT_data}/bdd100kmot_vehicle/labels_with_ids')
    for seq in seqs:
        print('seq => ', seq)
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find(
            '\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find(
            '\nimExt')])

        gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

        seq_label_root = osp.join(label_root, seq, 'img1')
        mkdirs(seq_label_root)

        for fid, tid, x, y, w, h, mark, label, _ in gt:
            fid = int(fid)
            tid = int(tid)
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root,
                                   seq + '-' + '{:07d}.txt'.format(fid))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / seq_width, y / seq_height, w / seq_width,
                h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)
