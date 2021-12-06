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
"""
This code is based on https://github.com/LCFractal/AIC21-MTMC/tree/main/reid/reid-matching/tools

Note: The following codes are strongly related to camera parameters of the AIC21 test-set S06,
    so they can only be used in S06, and can not be used for other MTMCT datasets.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from .utils import get_dire, get_match, get_cid_tid, combin_feature, combin_cluster
from .utils import normalize, intracam_ignore, visual_rerank

__all__ = [
    'st_filter',
    'get_labels_with_camera',
]

CAM_DIST = [[0, 40, 55, 100, 120, 145], [40, 0, 15, 60, 80, 105],
            [55, 15, 0, 40, 65, 90], [100, 60, 40, 0, 20, 45],
            [120, 80, 65, 20, 0, 25], [145, 105, 90, 45, 25, 0]]


def st_filter(st_mask, cid_tids, cid_tid_dict):
    count = len(cid_tids)
    for i in range(count):
        i_tracklet = cid_tid_dict[cid_tids[i]]
        i_cid = i_tracklet['cam']
        i_dire = get_dire(i_tracklet['zone_list'], i_cid)
        i_iot = i_tracklet['io_time']
        for j in range(count):
            j_tracklet = cid_tid_dict[cid_tids[j]]
            j_cid = j_tracklet['cam']
            j_dire = get_dire(j_tracklet['zone_list'], j_cid)
            j_iot = j_tracklet['io_time']

            match_dire = True
            cam_dist = CAM_DIST[i_cid - 41][j_cid - 41]
            # if time overlopped
            if i_iot[0] - cam_dist < j_iot[0] and j_iot[0] < i_iot[
                    1] + cam_dist:
                match_dire = False
            if i_iot[0] - cam_dist < j_iot[1] and j_iot[1] < i_iot[
                    1] + cam_dist:
                match_dire = False

            # not match after go out
            if i_dire[1] in [1, 2]:  # i out
                if i_iot[0] < j_iot[1] + cam_dist:
                    match_dire = False

            if i_dire[1] in [1, 2]:
                if i_dire[0] in [3] and i_cid > j_cid:
                    match_dire = False
                if i_dire[0] in [4] and i_cid < j_cid:
                    match_dire = False

            if i_cid in [41] and i_dire[1] in [4]:
                if i_iot[0] < j_iot[1] + cam_dist:
                    match_dire = False
                if i_iot[1] > 199:
                    match_dire = False
            if i_cid in [46] and i_dire[1] in [3]:
                if i_iot[0] < j_iot[1] + cam_dist:
                    match_dire = False

            # match after come into
            if i_dire[0] in [1, 2]:
                if i_iot[1] > j_iot[0] - cam_dist:
                    match_dire = False

            if i_dire[0] in [1, 2]:
                if i_dire[1] in [3] and i_cid > j_cid:
                    match_dire = False
                if i_dire[1] in [4] and i_cid < j_cid:
                    match_dire = False

            is_ignore = False
            if ((i_dire[0] == i_dire[1] and i_dire[0] in [3, 4]) or
                (j_dire[0] == j_dire[1] and j_dire[0] in [3, 4])):
                is_ignore = True

            if not is_ignore:
                # direction conflict
                if (i_dire[0] in [3] and j_dire[0] in [4]) or (
                        i_dire[1] in [3] and j_dire[1] in [4]):
                    match_dire = False
                # filter before going next scene
                if i_dire[1] in [3] and i_cid < j_cid:
                    if i_iot[1] > j_iot[1] - cam_dist:
                        match_dire = False
                if i_dire[1] in [4] and i_cid > j_cid:
                    if i_iot[1] > j_iot[1] - cam_dist:
                        match_dire = False

                if i_dire[0] in [3] and i_cid < j_cid:
                    if i_iot[0] < j_iot[0] + cam_dist:
                        match_dire = False
                if i_dire[0] in [4] and i_cid > j_cid:
                    if i_iot[0] < j_iot[0] + cam_dist:
                        match_dire = False
                ## 3-30
                ## 4-1
                if i_dire[0] in [3] and i_cid > j_cid:
                    if i_iot[1] > j_iot[0] - cam_dist:
                        match_dire = False
                if i_dire[0] in [4] and i_cid < j_cid:
                    if i_iot[1] > j_iot[0] - cam_dist:
                        match_dire = False
                # filter before going next scene
                ## 4-7
                if i_dire[1] in [3] and i_cid > j_cid:
                    if i_iot[0] < j_iot[1] + cam_dist:
                        match_dire = False
                if i_dire[1] in [4] and i_cid < j_cid:
                    if i_iot[0] < j_iot[1] + cam_dist:
                        match_dire = False
            else:
                if i_iot[1] > 199:
                    if i_dire[0] in [3] and i_cid < j_cid:
                        if i_iot[0] < j_iot[0] + cam_dist:
                            match_dire = False
                    if i_dire[0] in [4] and i_cid > j_cid:
                        if i_iot[0] < j_iot[0] + cam_dist:
                            match_dire = False
                    if i_dire[0] in [3] and i_cid > j_cid:
                        match_dire = False
                    if i_dire[0] in [4] and i_cid < j_cid:
                        match_dire = False
                if i_iot[0] < 1:
                    if i_dire[1] in [3] and i_cid > j_cid:
                        match_dire = False
                    if i_dire[1] in [4] and i_cid < j_cid:
                        match_dire = False

            if not match_dire:
                st_mask[i, j] = 0.0
                st_mask[j, i] = 0.0
    return st_mask


def subcam_list(cid_tid_dict, cid_tids):
    sub_3_4 = dict()
    sub_4_3 = dict()
    for cid_tid in cid_tids:
        cid, tid = cid_tid
        tracklet = cid_tid_dict[cid_tid]
        zs, ze = get_dire(tracklet['zone_list'], cid)
        if zs in [3] and cid not in [46]:  # 4 to 3
            if not cid + 1 in sub_4_3:
                sub_4_3[cid + 1] = []
            sub_4_3[cid + 1].append(cid_tid)
        if ze in [4] and cid not in [41]:  # 4 to 3
            if not cid in sub_4_3:
                sub_4_3[cid] = []
            sub_4_3[cid].append(cid_tid)
        if zs in [4] and cid not in [41]:  # 3 to 4
            if not cid - 1 in sub_3_4:
                sub_3_4[cid - 1] = []
            sub_3_4[cid - 1].append(cid_tid)
        if ze in [3] and cid not in [46]:  # 3 to 4
            if not cid in sub_3_4:
                sub_3_4[cid] = []
            sub_3_4[cid].append(cid_tid)
    sub_cid_tids = dict()
    for i in sub_3_4:
        sub_cid_tids[(i, i + 1)] = sub_3_4[i]
    for i in sub_4_3:
        sub_cid_tids[(i, i - 1)] = sub_4_3[i]
    return sub_cid_tids


def subcam_list2(cid_tid_dict, cid_tids):
    sub_dict = dict()
    for cid_tid in cid_tids:
        cid, tid = cid_tid
        if cid not in [41]:
            if not cid in sub_dict:
                sub_dict[cid] = []
            sub_dict[cid].append(cid_tid)
        if cid not in [46]:
            if not cid + 1 in sub_dict:
                sub_dict[cid + 1] = []
            sub_dict[cid + 1].append(cid_tid)
    return sub_dict


def get_sim_matrix(cid_tid_dict,
                   cid_tids,
                   use_ff=True,
                   use_rerank=True,
                   use_st_filter=False):
    # Note: camera releated get_sim_matrix function,
    # which is different from the one in utils.py.
    count = len(cid_tids)

    q_arr = np.array(
        [cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    g_arr = np.array(
        [cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    q_arr = normalize(q_arr, axis=1)
    g_arr = normalize(g_arr, axis=1)

    st_mask = np.ones((count, count), dtype=np.float32)
    st_mask = intracam_ignore(st_mask, cid_tids)

    # different from utils.py
    if use_st_filter:
        st_mask = st_filter(st_mask, cid_tids, cid_tid_dict)

    visual_sim_matrix = visual_rerank(
        q_arr, g_arr, cid_tids, use_ff=use_ff, use_rerank=use_rerank)
    visual_sim_matrix = visual_sim_matrix.astype('float32')

    np.set_printoptions(precision=3)
    sim_matrix = visual_sim_matrix * st_mask

    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix


def get_labels_with_camera(cid_tid_dict,
                           cid_tids,
                           use_ff=True,
                           use_rerank=True,
                           use_st_filter=False):
    # 1st cluster
    sub_cid_tids = subcam_list(cid_tid_dict, cid_tids)
    sub_labels = dict()
    dis_thrs = [0.7, 0.5, 0.5, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5]

    for i, sub_c_to_c in enumerate(sub_cid_tids):
        sim_matrix = get_sim_matrix(
            cid_tid_dict,
            sub_cid_tids[sub_c_to_c],
            use_ff=use_ff,
            use_rerank=use_rerank,
            use_st_filter=use_st_filter)
        cluster_labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - dis_thrs[i],
            affinity='precomputed',
            linkage='complete').fit_predict(1 - sim_matrix)
        labels = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(labels, sub_cid_tids[sub_c_to_c])
        sub_labels[sub_c_to_c] = cluster_cid_tids
    labels, sub_cluster = combin_cluster(sub_labels, cid_tids)

    # 2nd cluster
    cid_tid_dict_new = combin_feature(cid_tid_dict, sub_cluster)
    sub_cid_tids = subcam_list2(cid_tid_dict_new, cid_tids)
    sub_labels = dict()
    for i, sub_c_to_c in enumerate(sub_cid_tids):
        sim_matrix = get_sim_matrix(
            cid_tid_dict_new,
            sub_cid_tids[sub_c_to_c],
            use_ff=use_ff,
            use_rerank=use_rerank,
            use_st_filter=use_st_filter)
        cluster_labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - 0.1,
            affinity='precomputed',
            linkage='complete').fit_predict(1 - sim_matrix)
        labels = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(labels, sub_cid_tids[sub_c_to_c])
        sub_labels[sub_c_to_c] = cluster_cid_tids
    labels, sub_cluster = combin_cluster(sub_labels, cid_tids)

    return labels
