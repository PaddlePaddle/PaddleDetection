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
"""

import os
import re
import cv2
import gc
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
import motmetrics as mm
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

__all__ = [
    'parse_pt', 'parse_bias', 'get_dire', 'parse_pt_gt',
    'compare_dataframes_mtmc', 'get_sim_matrix', 'get_labels', 'getData',
    'gen_new_mot'
]


def parse_pt(mot_feature, zones=None):
    mot_list = dict()
    for line in mot_feature:
        fid = int(re.sub('[a-z,A-Z]', "", mot_feature[line]['frame']))
        tid = mot_feature[line]['id']
        bbox = list(map(lambda x: int(float(x)), mot_feature[line]['bbox']))
        if tid not in mot_list:
            mot_list[tid] = dict()
        out_dict = mot_feature[line]
        if zones is not None:
            out_dict['zone'] = zones.get_zone(bbox)
        else:
            out_dict['zone'] = None
        mot_list[tid][fid] = out_dict
    return mot_list


def gen_new_mot(mot_list):
    out_dict = dict()
    for tracklet in mot_list:
        tracklet = mot_list[tracklet]
        for f in tracklet:
            out_dict[tracklet[f]['imgname']] = tracklet[f]
    return out_dict


def mergesetfeat1_notrk(P, neg_vector, in_feats, in_labels):
    out_feats = []
    for i in range(in_feats.shape[0]):
        camera_id = in_labels[i, 1]
        feat = in_feats[i] - neg_vector[camera_id]
        feat = P[camera_id].dot(feat)
        feat = feat / np.linalg.norm(feat, ord=2)
        out_feats.append(feat)
    out_feats = np.vstack(out_feats)
    return out_feats


def compute_P2(prb_feats, gal_feats, gal_labels, la=3.0):
    X = gal_feats
    neg_vector = {}
    u_labels = np.unique(gal_labels[:, 1])
    P = {}
    for label in u_labels:
        curX = gal_feats[gal_labels[:, 1] == label, :]
        neg_vector[label] = np.mean(curX, axis=0)
        P[label] = np.linalg.inv(
            curX.T.dot(curX) + curX.shape[0] * la * np.eye(X.shape[1]))
    return P, neg_vector


def parse_bias(cameras_bias):
    cid_bias = dict()
    for cameras in cameras_bias.keys():
        cameras_id = re.sub('[a-z,A-Z]', "", cameras)
        cameras_id = int(cameras_id)
        bias = cameras_bias[cameras]
        cid_bias[cameras_id] = float(bias)
    return cid_bias


def get_dire(zone_list, cid):
    zs, ze = zone_list[0], zone_list[-1]
    return (zs, ze)


def intracam_ignore(st_mask, cid_tids):
    count = len(cid_tids)
    for i in range(count):
        for j in range(count):
            if cid_tids[i][0] == cid_tids[j][0]:
                st_mask[i, j] = 0.
    return st_mask


def mergesetfeat(in_feats, in_labels, in_tracks):
    trackset = list(set(list(in_tracks)))
    out_feats = []
    out_labels = []
    for track in trackset:
        feat = np.mean(in_feats[in_tracks == track], axis=0)
        feat = feat / np.linalg.norm(feat, ord=2)
        label = in_labels[in_tracks == track][0]
        out_feats.append(feat)
        out_labels.append(label)
    out_feats = np.vstack(out_feats)
    out_labels = np.vstack(out_labels)
    return out_feats, out_labels


def mergesetfeat3(X, labels, gX, glabels, beta=0.08, knn=20, lr=0.5):
    for i in range(0, X.shape[0]):
        if i % 1000 == 0:
            print('feat3:%d/%d' % (i, X.shape[0]))
        knnX = gX[glabels[:, 1] != labels[i, 1], :]
        sim = knnX.dot(X[i, :])
        knnX = knnX[sim > 0, :]
        sim = sim[sim > 0]
        if len(sim) > 0:
            idx = np.argsort(-sim)
            if len(sim) > 2 * knn:
                sim = sim[idx[:2 * knn]]
                knnX = knnX[idx[:2 * knn], :]
            else:
                sim = sim[idx]
                knnX = knnX[idx, :]
                knn = min(knn, len(sim))
            knn_pos_weight = np.exp((sim[:knn] - 1) / beta)
            knn_neg_weight = np.ones(len(sim) - knn)
            knn_pos_prob = knn_pos_weight / np.sum(knn_pos_weight)
            knn_neg_prob = knn_neg_weight / np.sum(knn_neg_weight)
            X[i, :] += lr * (knn_pos_prob.dot(knnX[:knn, :]) -
                             knn_neg_prob.dot(knnX[knn:, :]))
            X[i, :] /= np.linalg.norm(X[i, :])
    return X


def run_fic(prb_feats, gal_feats, prb_labels, gal_labels, la=3.0):
    P, neg_vector = compute_P2(prb_feats, gal_feats, gal_labels, la)
    prb_feats_new = mergesetfeat1_notrk(P, neg_vector, prb_feats, prb_labels)
    gal_feats_new = mergesetfeat1_notrk(P, neg_vector, gal_feats, gal_labels)
    return prb_feats_new, gal_feats_new


def run_fac(prb_feats,
            gal_feats,
            prb_labels,
            gal_labels,
            beta=0.08,
            knn=20,
            lr=0.5,
            prb_epoch=2,
            gal_epoch=3):
    gal_feats_new = gal_feats.copy()
    for i in range(prb_epoch):
        gal_feats_new = mergesetfeat3(gal_feats_new, gal_labels, gal_feats,
                                      gal_labels, beta, knn, lr)
    prb_feats_new = prb_feats.copy()
    for i in range(gal_epoch):
        prb_feats_new = mergesetfeat3(prb_feats_new, prb_labels, gal_feats_new,
                                      gal_labels, beta, knn, lr)
    return prb_feats_new, gal_feats_new


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = 2 - 2 * np.matmul(qf, gf.T)
    return dist_mat


def find_topk(a, k, axis=-1, largest=True, sorted=True):
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size-k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
    else:
        index_array = np.argpartition(a, k-1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis)
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices


def batch_numpy_topk(qf, gf, k1, N=6000):
    m = qf.shape[0]
    n = gf.shape[0]
    initial_rank = []
    for j in range(n // N + 1):
        temp_gf = gf[j * N:j * N + N]
        temp_qd = []
        for i in range(m // N + 1):
            temp_qf = qf[i * N:i * N + N]
            temp_d = euclidean_distance(temp_qf, temp_gf)
            temp_qd.append(temp_d)
        temp_qd = np.concatenate(temp_qd, axis=0)
        temp_qd = temp_qd / (np.max(temp_qd, axis=0)[0])
        temp_qd = temp_qd.T
        initial_rank.append(
            find_topk(temp_qd, k=k1, axis=1, largest=False, sorted=True)[1])
    del temp_qd
    del temp_gf
    del temp_qf
    del temp_d
    initial_rank = np.concatenate(initial_rank, axis=0)
    return initial_rank


def batch_euclidean_distance(qf, gf, N=6000):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = []
    for j in range(n // N + 1):
        temp_gf = gf[j * N:j * N + N]
        temp_qd = []
        for i in range(m // N + 1):
            temp_qf = qf[i * N:i * N + N]
            temp_d = euclidean_distance(temp_qf, temp_gf)
            temp_qd.append(temp_d)
        temp_qd = np.concatenate(temp_qd, axis=0)
        temp_qd = temp_qd / (np.max(temp_qd, axis=0)[0])
        dist_mat.append(temp_qd.T)
    del temp_qd
    del temp_gf
    del temp_qf
    del temp_d
    dist_mat = np.concatenate(dist_mat, axis=0)
    return dist_mat


def batch_v(feat, R, all_num):
    V = np.zeros((all_num, all_num), dtype=np.float32)
    m = feat.shape[0]
    for i in tqdm(range(m)):
        temp_gf = feat[i].reshape(1, -1)
        temp_qd = euclidean_distance(temp_gf, feat)
        temp_qd = temp_qd / (np.max(temp_qd))
        temp_qd = temp_qd.reshape(-1)
        temp_qd = temp_qd[R[i].tolist()]
        weight = np.exp(-temp_qd)
        weight = weight / np.sum(weight)
        V[i, R[i]] = weight.astype(np.float32)
    return V


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def ReRank2(probFea, galFea, k1=20, k2=6, lambda_value=0.3):
    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]
    feat = np.concatenate((probFea, galFea), axis=0)

    initial_rank = batch_numpy_topk(feat, feat, k1 + 1, N=6000)
    del probFea
    del galFea
    gc.collect()  # empty memory
    R = []
    for i in tqdm(range(all_num)):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(
                initial_rank, candidate, int(np.around(k1 / 2)))
            if len(
                    np.intersect1d(candidate_k_reciprocal_index,
                                   k_reciprocal_index)) > 2. / 3 * len(
                                       candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        R.append(k_reciprocal_expansion_index)

    gc.collect()  # empty memory
    V = batch_v(feat, R, all_num)
    del R
    gc.collect()  # empty memory
    initial_rank = initial_rank[:, :k2]

    # Faster version
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    gc.collect()  # empty memory
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])
    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)
    for i in tqdm(range(query_num)):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)
    del V
    gc.collect()  # empty memory
    original_dist = batch_euclidean_distance(feat, feat[:query_num, :])
    final_dist = jaccard_dist * (1 - lambda_value
                                 ) + original_dist * lambda_value
    del original_dist
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def visual_rerank(prb_feats,
                  gal_feats,
                  cid_tids,
                  use_ff=False,
                  use_rerank=False):
    """Rerank by visual cures."""
    gal_labels = np.array([[0, item[0]] for item in cid_tids])
    prb_labels = gal_labels.copy()
    if use_ff:
        print('current use ff finetuned parameters....')
        # Step1-1: fic. finetuned parameters: [la]
        prb_feats, gal_feats = run_fic(prb_feats, gal_feats, prb_labels,
                                       gal_labels, 3.0)
        # Step1=2: fac. finetuned parameters: [beta,knn,lr,prb_epoch,gal_epoch]
        prb_feats, gal_feats = run_fac(prb_feats, gal_feats, prb_labels,
                                       gal_labels, 0.08, 20, 0.5, 1, 1)
    if use_rerank:
        print('current use rerank finetuned parameters....')
        # Step2: k-reciprocal. finetuned parameters: [k1,k2,lambda_value]
        sims = ReRank2(prb_feats, gal_feats, 20, 3, 0.3)
    else:
        sims = 1.0 - np.dot(prb_feats, gal_feats.T)

    # NOTE: sims here is actually dist, the smaller the more similar
    return 1.0 - sims


def normalize(nparray, axis=0):
    nparray = preprocessing.normalize(nparray, norm='l2', axis=axis)
    return nparray


def get_match(cluster_labels):
    cluster_dict = dict()
    cluster = list()
    for i, l in enumerate(cluster_labels):
        if l in list(cluster_dict.keys()):
            cluster_dict[l].append(i)
        else:
            cluster_dict[l] = [i]
    for idx in cluster_dict:
        cluster.append(cluster_dict[idx])
    return cluster


def get_cid_tid(cluster_labels, cid_tids):
    cluster = list()
    for labels in cluster_labels:
        cid_tid_list = list()
        for label in labels:
            cid_tid_list.append(cid_tids[label])
        cluster.append(cid_tid_list)
    return cluster


def combin_feature(cid_tid_dict, sub_cluster):
    for sub_ct in sub_cluster:
        if len(sub_ct) < 2: continue
        mean_feat = np.array([cid_tid_dict[i]['mean_feat'] for i in sub_ct])
        for i in sub_ct:
            cid_tid_dict[i]['mean_feat'] = mean_feat.mean(axis=0)
    return cid_tid_dict


def combin_cluster(sub_labels, cid_tids):
    cluster = list()
    for sub_c_to_c in sub_labels:
        if len(cluster) < 1:
            cluster = sub_labels[sub_c_to_c]
            continue
        for c_ts in sub_labels[sub_c_to_c]:
            is_add = False
            for i_c, c_set in enumerate(cluster):
                if len(set(c_ts) & set(c_set)) > 0:
                    new_list = list(set(c_ts) | set(c_set))
                    cluster[i_c] = new_list
                    is_add = True
                    break
            if not is_add:
                cluster.append(c_ts)
    labels = list()
    num_tr = 0
    for c_ts in cluster:
        label_list = list()
        for c_t in c_ts:
            label_list.append(cid_tids.index(c_t))
            num_tr += 1
        label_list.sort()
        labels.append(label_list)
    return labels, cluster


def parse_pt_gt(mot_feature):
    img_rects = dict()
    for line in mot_feature:
        fid = int(re.sub('[a-z,A-Z]', "", mot_feature[line]['frame']))
        tid = mot_feature[line]['id']
        rect = list(map(lambda x: int(float(x)), mot_feature[line]['bbox']))
        if fid not in img_rects:
            img_rects[fid] = list()
        rect.insert(0, tid)
        img_rects[fid].append(rect)
    return img_rects


# eval result
def compare_dataframes_mtmc(gts, ts):
    """Compute ID-based evaluation metrics for MTMCT
    Return:
        df (pandas.DataFrame): Results of the evaluations in a df with only the 'idf1', 'idp', and 'idr' columns.
    """
    gtds = []
    tsds = []
    gtcams = gts['CameraId'].drop_duplicates().tolist()
    tscams = ts['CameraId'].drop_duplicates().tolist()
    maxFrameId = 0

    for k in sorted(gtcams):
        gtd = gts.query('CameraId == %d' % k)
        gtd = gtd[['FrameId', 'Id', 'X', 'Y', 'Width', 'Height']]
        # max FrameId in gtd only
        mfid = gtd['FrameId'].max()
        gtd['FrameId'] += maxFrameId
        gtd = gtd.set_index(['FrameId', 'Id'])
        gtds.append(gtd)

        if k in tscams:
            tsd = ts.query('CameraId == %d' % k)
            tsd = tsd[['FrameId', 'Id', 'X', 'Y', 'Width', 'Height']]
            # max FrameId among both gtd and tsd
            mfid = max(mfid, tsd['FrameId'].max())
            tsd['FrameId'] += maxFrameId
            tsd = tsd.set_index(['FrameId', 'Id'])
            tsds.append(tsd)

        maxFrameId += mfid

    # compute multi-camera tracking evaluation stats
    multiCamAcc = mm.utils.compare_to_groundtruth(
        pd.concat(gtds), pd.concat(tsds), 'iou')
    metrics = list(mm.metrics.motchallenge_metrics)
    metrics.extend(['num_frames', 'idfp', 'idfn', 'idtp'])
    mh = mm.metrics.create()
    summary = mh.compute(multiCamAcc, metrics=metrics, name='MultiCam')
    return summary


def get_sim_matrix(cid_tid_dict,
                   cid_tids,
                   use_ff=True,
                   use_rerank=True,
                   use_st_filter=False):
    # Note: carame independent get_sim_matrix function,
    # which is different from the one in camera_utils.py.
    count = len(cid_tids)

    q_arr = np.array(
        [cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    g_arr = np.array(
        [cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    q_arr = normalize(q_arr, axis=1)
    g_arr = normalize(g_arr, axis=1)

    st_mask = np.ones((count, count), dtype=np.float32)
    st_mask = intracam_ignore(st_mask, cid_tids)

    visual_sim_matrix = visual_rerank(
        q_arr, g_arr, cid_tids, use_ff=use_ff, use_rerank=use_rerank)
    visual_sim_matrix = visual_sim_matrix.astype('float32')

    np.set_printoptions(precision=3)
    sim_matrix = visual_sim_matrix * st_mask

    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix


def get_labels(cid_tid_dict,
               cid_tids,
               use_ff=True,
               use_rerank=True,
               use_st_filter=False):
    # 1st cluster
    sim_matrix = get_sim_matrix(
        cid_tid_dict,
        cid_tids,
        use_ff=use_ff,
        use_rerank=use_rerank,
        use_st_filter=use_st_filter)
    cluster_labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.5,
        affinity='precomputed',
        linkage='complete').fit_predict(1 - sim_matrix)
    labels = get_match(cluster_labels)
    sub_cluster = get_cid_tid(labels, cid_tids)

    # 2nd cluster
    cid_tid_dict_new = combin_feature(cid_tid_dict, sub_cluster)
    sim_matrix = get_sim_matrix(
        cid_tid_dict_new,
        cid_tids,
        use_ff=use_ff,
        use_rerank=use_rerank,
        use_st_filter=use_st_filter)
    cluster_labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.9,
        affinity='precomputed',
        linkage='complete').fit_predict(1 - sim_matrix)
    labels = get_match(cluster_labels)
    sub_cluster = get_cid_tid(labels, cid_tids)

    return labels


def getData(fpath, names=None, sep='\s+|\t+|,'):
    """ Get the necessary track data from a file handle.
    Args:
        fpath (str) : Original path of file reading from.
        names (list[str]): List of column names for the data.
        sep (str): Allowed separators regular expression string.
    Return:
        df (pandas.DataFrame): Data frame containing the data loaded from the
            stream with optionally assigned column names. No index is set on the data.
    """
    try:
        df = pd.read_csv(
            fpath,
            sep=sep,
            index_col=None,
            skipinitialspace=True,
            header=None,
            names=names,
            engine='python')
        return df

    except Exception as e:
        raise ValueError("Could not read input from %s. Error: %s" %
                         (fpath, repr(e)))
