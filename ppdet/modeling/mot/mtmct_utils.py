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
import re
import gc
import sys
import paddle
import numpy as np
import pandas as pd
import motmetrics as mm
from tqdm import tqdm
from collections import defaultdict

from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering

__all__ = [
    'print_results',
    'trajectory_fusion',
    'parse_pt',
    'parse_bias',
    'gen_res',
    'sub_cluster',
    'print_mtmct_result'
]


def compute_P(prb_feats,gal_feats):
    '''
    if prb_feats.shape[0] == gal_feats.shape[0]:
        if (prb_feats == gal_feats).all():
            X = prb_feats
    else:
        X = np.vstack([prb_feats,gal_feats])
    '''
    X = gal_feats
    neg_vector = np.mean(X, axis=0)
    la = 0.04
    P = np.linalg.inv(X.T.dot(X)+X.shape[0]*la*np.eye(X.shape[1]))
    return P, neg_vector

def compute_P2(prb_feats,gal_feats,gal_labels,la=3.0):
    X = gal_feats
    neg_vector = {}
    u_labels = np.unique(gal_labels[:,1])
    P = {}
    for label in u_labels:
        curX = gal_feats[gal_labels[:,1]==label,:]
        neg_vector[label] = np.mean(curX,axis=0)
        P[label] = np.linalg.inv(curX.T.dot(curX)+curX.shape[0]*la*np.eye(X.shape[1]))
    return P, neg_vector

def mergesetfeat3(X,labels,gX,glabels,beta=0.08,knn=20,lr=0.5):
    for i in range(0,X.shape[0]):
        knnX = gX[glabels[:,1]!=labels[i,1],:]
        sim = knnX.dot(X[i,:])
        knnX = knnX[sim>0,:]
        sim = sim[sim>0]
        if len(sim)>0:
            idx = np.argsort(-sim)
            if len(sim)>2*knn:
                sim = sim[idx[:2*knn]]
                knnX = knnX[idx[:2*knn],:]
            else:
                sim = sim[idx]
                knnX = knnX[idx,:]
                knn = min(knn,len(sim))
            knn_pos_weight = np.exp((sim[:knn]-1)/beta)
            knn_neg_weight = np.ones(len(sim)-knn)
            knn_pos_prob = knn_pos_weight/np.sum(knn_pos_weight)
            knn_neg_prob = knn_neg_weight/np.sum(knn_neg_weight)
            X[i,:] += lr*(knn_pos_prob.dot(knnX[:knn,:]) - knn_neg_prob.dot(knnX[knn:,:]))
            X[i,:] /= np.linalg.norm(X[i,:])
    return X

def mergesetfeat1_notrk(P,neg_vector,in_feats,in_labels):
    out_feats = []
    for i in range(in_feats.shape[0]):
        camera_id = in_labels[i,1]
        feat = in_feats[i] - neg_vector[camera_id]
        feat = P[camera_id].dot(feat)
        feat = feat/np.linalg.norm(feat,ord=2)
        out_feats.append(feat)
    out_feats = np.vstack(out_feats)
    return out_feats

def mergesetfeat1(P,neg_vector,in_feats,in_labels,in_tracks):
    trackset = list(set(list(in_tracks)))
    out_feats = []
    out_labels = []
    for track in trackset:
        camera_id = in_labels[in_tracks==track,1][0]
        feat = np.mean(in_feats[in_tracks==track],axis=0) - neg_vector[camera_id]
        feat = P[camera_id].dot(feat)
        feat = feat/np.linalg.norm(feat,ord=2)
        label = in_labels[in_tracks==track][0]
        out_feats.append(feat)
        out_labels.append(label)
    out_feats = np.vstack(out_feats)
    out_labels = np.vstack(out_labels)
    return out_feats,out_labels

def mergesetfeat(in_feats,in_labels,in_tracks):
    trackset = list(set(list(in_tracks)))
    out_feats = []
    out_labels = []
    for track in trackset:
        feat = np.mean(in_feats[in_tracks==track],axis=0)
        feat = feat/np.linalg.norm(feat,ord=2)
        label = in_labels[in_tracks==track][0]
        out_feats.append(feat)
        out_labels.append(label)
    out_feats = np.vstack(out_feats)
    out_labels = np.vstack(out_labels)
    return out_feats,out_labels

def run_fic(prb_feats,gal_feats,prb_labels,gal_labels,la=3.0):
    P,neg_vector = compute_P2(prb_feats,gal_feats,gal_labels,la)
    prb_feats_new = mergesetfeat1_notrk(P,neg_vector,prb_feats,prb_labels)
    gal_feats_new = mergesetfeat1_notrk(P,neg_vector,gal_feats,gal_labels)
    return prb_feats_new,gal_feats_new

def run_fac(prb_feats,gal_feats,prb_labels,gal_labels,beta=0.08,knn=20,lr=0.5,prb_epoch=2,gal_epoch=3): 
    gal_feats_new = gal_feats.copy()
    for i in range(prb_epoch):
        gal_feats_new = mergesetfeat3(gal_feats_new,gal_labels,gal_feats,gal_labels,beta,knn,lr)
    prb_feats_new = prb_feats.copy()
    for i in range(gal_epoch):
        prb_feats_new = mergesetfeat3(prb_feats_new,prb_labels,gal_feats_new,gal_labels,beta,knn,lr)
    return prb_feats_new,gal_feats_new

# rerank2
def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = 2 - 2 * paddle.matmul(qf, gf.t()) # for L2-norm feature
    return dist_mat

def batch_paddle_topk(qf, gf, k1, N=6000):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = []
    initial_rank = []
    for j in range(n // N + 1):
        temp_gf = gf[j * N:j * N + N]
        temp_qd = []
        for i in range(m // N + 1):
            temp_qf = qf[i * N:i * N + N]
            temp_d = euclidean_distance(temp_qf, temp_gf)
            temp_qd.append(temp_d)
        temp_qd = paddle.concat(temp_qd, axis=0)
        temp_qd = temp_qd / (paddle.max(temp_qd, axis=0)[0])
        temp_qd = temp_qd.t()
        initial_rank.append(paddle.topk(temp_qd, k=k1, axis=1, largest=False, sorted=True)[1])
    del temp_qd
    del temp_gf
    del temp_qf
    del temp_d
    initial_rank = paddle.concat(initial_rank, axis=0).numpy()
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
        temp_qd = paddle.concat(temp_qd, axis=0)
        temp_qd = temp_qd / (paddle.max(temp_qd, axis=0)[0])
        dist_mat.append(temp_qd.t()) # transpose
    del temp_qd
    del temp_gf
    del temp_qf
    del temp_d
    dist_mat = paddle.concat(dist_mat, axis=0)
    return dist_mat

def batch_v(feat, R, all_num):
    V = np.zeros((all_num, all_num), dtype=np.float32)
    m = feat.shape[0]
    for i in tqdm(range(m)):
        temp_gf = feat[i].unsqueeze(0)
        temp_qd = euclidean_distance(temp_gf, feat)
        temp_qd = temp_qd / (paddle.max(temp_qd))
        temp_qd = temp_qd.squeeze()
        temp_qd = temp_qd.numpy()[R[i].tolist()]
        temp_qd = paddle.to_tensor(temp_qd)
        weight = paddle.exp(-temp_qd)
        weight = (weight / paddle.sum(weight)).numpy()
        V[i, R[i]] = weight.astype(np.float32)
    return V

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]

def ReRank2(probFea, galFea, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]
    feat = paddle.concat([probFea, galFea],axis=0)
    initial_rank = batch_paddle_topk(feat, feat, k1 + 1, N=6000)
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
            candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
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
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)
    del V
    gc.collect()  # empty memory
    original_dist = batch_euclidean_distance(feat, feat[:query_num, :]).numpy()
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del jaccard_dist

    final_dist = final_dist[:query_num, query_num:]
    return final_dist

def intracam_ignore(st_mask, cid_tids):
    # filter
    count = len(cid_tids)
    for i in range(count):
        for j in range(count):
            if cid_tids[i][0] == cid_tids[j][0]:
                st_mask[i, j] = 0.
    return st_mask

# eval function
names = ['CameraId','Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld']
mh = mm.metrics.create()

def getData(fpath, names=None, sep='\s+|\t+|,'):
    """ Get the necessary track data from a file handle.
    Params
    ------
    fpath : str
        Original path of file reading from.
    names : list<str>
        List of column names for the data.
    sep : str
        Allowed separators regular expression string.
    Returns
    -------
    df : pandas.DataFrame
        Data frame containing the data loaded from the stream with optionally assigned column names.
        No index is set on the data.
    """
    try:
        df = pd.read_csv(
            fpath, 
            sep=sep, 
            index_col=None, 
            skipinitialspace=True, 
            header=None,
            names=names,
            engine='python'
        )
        return df
    except Exception as e:
        raise ValueError("Could not read input from %s. Error: %s" % (fpath, repr(e)))

def compare_dataframes_mtmc(gts, ts):
    """Compute ID-based evaluation metrics for multi-camera multi-object tracking.
    
    Params
    ------
    gts : pandas.DataFrame
        Ground truth data.
    ts : pandas.DataFrame
        Prediction/test data.
    Returns
    -------
    df : pandas.DataFrame
        Results of the evaluations in a df with only the 'idf1', 'idp', and 'idr' columns.
    """
    gtds = []
    tsds = []
    gtcams = gts['CameraId'].drop_duplicates().tolist()
    tscams = ts['CameraId'].drop_duplicates().tolist()
    maxFrameId = 0;

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
    multiCamAcc = mm.utils.compare_to_groundtruth(pd.concat(gtds), pd.concat(tsds), 'iou')
    metrics=list(mm.metrics.motchallenge_metrics)
    metrics.extend(['num_frames','idfp','idfn','idtp'])
    summary = mh.compute(multiCamAcc, metrics=metrics, name='MultiCam')
    return summary

def print_results(summary, mread=False):
    print('summary => ', summary.columns.tolist())
    if mread:
        print('{"results":%s}' % summary.iloc[-1].to_json())
        return
    
    formatters = {'idf1': '{:2.2f}'.format,
                  'idp': '{:2.2f}'.format,
                  'idr': '{:2.2f}'.format,
                  'mota': '{:2.2f}'.format}

    summary = summary[['idf1','idp','idr','mota']]
    summary.loc[:,'idp'] *= 100
    summary.loc[:,'idr'] *= 100
    summary.loc[:,'idf1'] *= 100
    summary.loc[:,'mota'] *= 100

    print(mm.io.render_summary(summary, formatters=formatters, namemap=mm.io.motchallenge_metric_names))
    return

def get_mtmct_reuslt(gt_file, pred_file):
    gt = getData(gt_file, names=names)
    pred = getData(pred_file, names=names)
    summary = compare_dataframes_mtmc(gt, pred)
    print_results(summary)

# trajectory_fusion
def parse_pt(mot_feature):
    mot_list = dict()
    for line in mot_feature:
        fid = int(re.sub('[a-z,A-Z]',"",mot_feature[line]['frame']))
        tid = mot_feature[line]['id']
        bbox = list(map(lambda x:int(float(x)), mot_feature[line]['bbox']))
        if tid not in mot_list:
            mot_list[tid] = dict()
        out_dict = mot_feature[line]
        mot_list[tid][fid] = out_dict
    return mot_list

def parse_bias(cameras_bias):
    cid_bias = dict()
    for cameras in cameras_bias.keys():
        cameras_id = re.sub('[a-z,A-Z]', "", cameras)
        cameras_id = int(cameras_id)
        bias = cameras_bias[cameras]
        cid_bias[cameras_id] = float(bias)
    return cid_bias

def trajectory_fusion(mot_feature, cid, cid_bias):
    cur_bias = cid_bias[cid]
    mot_list = parse_pt(mot_feature)
    tid_data = dict()
    for tid in mot_list:
        tracklet = mot_list[tid]
        if len(tracklet) <= 1: continue
        frame_list = list(tracklet.keys())
        frame_list.sort()
        # filter area too large
        feature_list = [tracklet[f]['feat'] for f in frame_list if (tracklet[f]['bbox'][3]-tracklet[f]['bbox'][1])*(tracklet[f]['bbox'][2]-tracklet[f]['bbox'][0])>2000]
        if len(feature_list)<2:
            feature_list = [tracklet[f]['feat'] for f in frame_list]
        all_feat = np.array([feat for feat in feature_list])
        mean_feat = np.mean(all_feat, axis=0)
        tid_data[tid]={
            'cam': cid,
            'tid': tid,
            'mean_feat': mean_feat,
            'frame_list': frame_list,
            'tracklet': tracklet,
        }
    return tid_data

# sub_cluster
def normalize(nparray, axis=0):
    nparray = preprocessing.normalize(nparray, norm='l2', axis=axis)
    return nparray

def visual_rerank(prb_feats, gal_feats, cid_tids, use_ff=False, use_rerank=False):
    """Rerank by visual cures."""
    gal_labels = np.array([[0, item[0]] for item in cid_tids])
    prb_labels = gal_labels.copy()
    if use_ff:
        # Step1-1: fic. finetuned parameters: [la]
        prb_feats, gal_feats = run_fic(prb_feats, gal_feats,
                                          prb_labels, gal_labels, 3.0)
        # Step1=2: fac. finetuned parameters: [beta,knn,lr,prb_epoch,gal_epoch]
        prb_feats, gal_feats = run_fac(prb_feats, gal_feats,
                                          prb_labels, gal_labels,
                                          0.08, 20, 0.5, 1, 1)
    if use_rerank:
        # Step2: k-reciprocal. finetuned parameters: [k1,k2,lambda_value]
        sims = ReRank2(paddle.to_tensor(prb_feats),
                          paddle.to_tensor(gal_feats), 20, 3, 0.3)
    else:
        sims = 1.0 - np.dot(prb_feats, gal_feats.T)

    # sims here is actually dist, the smaller the more similar
    sims = 1.0 - np.dot(prb_feats, gal_feats.T)
    return 1.0 - sims

def get_cid_tid(cluster_labels,cid_tids):
    cluster = list()
    for labels in cluster_labels:
        cid_tid_list = list()
        for label in labels:
            cid_tid_list.append(cid_tids[label])
        cluster.append(cid_tid_list)
    return cluster

def get_sim_matrix(cid_tid_dict, cid_tids, use_ff=False, use_rerank=False):
    count = len(cid_tids)
    q_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    g_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    q_arr = normalize(q_arr, axis=1)
    g_arr = normalize(g_arr, axis=1)
    st_mask = np.ones((count, count), dtype=np.float32)
    st_mask = intracam_ignore(st_mask, cid_tids)
    visual_sim_matrix = visual_rerank(q_arr, g_arr, cid_tids, use_ff=use_ff, use_rerank=use_rerank)
    visual_sim_matrix = visual_sim_matrix.astype('float32')
    np.set_printoptions(precision=3)
    sim_matrix = visual_sim_matrix * st_mask
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix

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

def combin_cluster(sub_labels,cid_tids):
    cluster = list()
    for sub_c_to_c in sub_labels:
        if len(cluster)<1:
            cluster = sub_labels[sub_c_to_c]
            continue
        for c_ts in sub_labels[sub_c_to_c]:
            is_add = False
            for i_c, c_set in enumerate(cluster):
                if len(set(c_ts) & set(c_set))>0:
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
            num_tr+=1
        label_list.sort()
        labels.append(label_list)
    return labels,cluster

def combin_feature(cid_tid_dict,sub_cluster):
    for sub_ct in sub_cluster:
        if len(sub_ct)<2: continue
        mean_feat = np.array([cid_tid_dict[i]['mean_feat'] for i in sub_ct])
        for i in sub_ct:
            cid_tid_dict[i]['mean_feat'] = mean_feat.mean(axis=0)
    return cid_tid_dict

def get_labels(cid_tid_dict, cid_tids, score_thr,use_ff=False, use_rerank=False):
    sub_cid_tids = list(cid_tid_dict.keys())
    sub_labels = dict()
    sim_matrix = get_sim_matrix(cid_tid_dict,cid_tids,use_ff=use_ff,use_rerank=use_rerank)
    cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
    labels = get_match(cluster_labels)
    sub_cluster = get_cid_tid(labels,cid_tids)
    # 2nd cluster
    cid_tid_dict_new = combin_feature(cid_tid_dict, sub_cluster)
    sub_labels = dict()
    sim_matrix = get_sim_matrix(cid_tid_dict_new,cid_tids,use_ff=use_ff,use_rerank=use_rerank)
    cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=0.9, affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
    labels = get_match(cluster_labels)
    sub_cluster = get_cid_tid(labels,cid_tids)
    return labels

def sub_cluster(cid_tid_dict, scene_cluster, score_thr,use_ff=False, use_rerank=False):
    '''
    cid_tid_dict: all camera_id and track_id
    scene_cluster: like [41, 42, 43, 44, 45, 46]
    score_thr: config param
    '''
    print('use_ff => ',use_ff, 'use_rerank =>', use_rerank)
    cid_tids = sorted([key for key in cid_tid_dict.keys() if key[0] in scene_cluster])
    clu = get_labels(cid_tid_dict,cid_tids,score_thr=score_thr,use_ff=use_ff, use_rerank=use_ff)
    print('all_clu:', len(clu))
    new_clu = list()
    for c_list in clu:
        if len(c_list) <= 1: continue
        cam_list = [cid_tids[c][0] for c in c_list]
        if len(cam_list)!=len(set(cam_list)): continue
        new_clu.append([cid_tids[c] for c in c_list])
    print('new_clu: ', len(new_clu))
    all_clu = new_clu
    cid_tid_label = dict()
    for i, c_list in enumerate(all_clu):
        for c in c_list:
            cid_tid_label[c] = i + 1
    return cid_tid_label

def parse_pt_gt(mot_feature):
    # gen result
    img_rects = dict()
    for line in mot_feature:
        fid = int(re.sub('[a-z,A-Z]',"",mot_feature[line]['frame']))
        tid = mot_feature[line]['id']
        rect = list(map(lambda x: int(float(x)), mot_feature[line]['bbox']))
        if fid not in img_rects:
            img_rects[fid] = list()
        rect.insert(0, tid)
        img_rects[fid].append(rect)
    return img_rects

def gen_res(output_dir_filename, scene_cluster, map_tid, mot_features):
    f_w = open(output_dir_filename, 'w')
    for cid in scene_cluster:
        for mot_feature in mot_features:
            img_rects = parse_pt_gt(mot_feature)
            for fid in img_rects:
                tid_rects = img_rects[fid]
                fid = int(fid)+1 # frameId add from 1
                for tid_rect in tid_rects:
                    tid = tid_rect[0]
                    rect = tid_rect[1:]
                    cx = 0.5*rect[0] + 0.5*rect[2]
                    cy = 0.5*rect[1] + 0.5*rect[3]
                    w = rect[2] - rect[0]
                    w = min(w*1.2,w+40)
                    h = rect[3] - rect[1]
                    h = min(h*1.2,h+40)
                    rect[2] -= rect[0]
                    rect[3] -= rect[1]
                    rect[0] = max(0, rect[0])
                    rect[1] = max(0, rect[1])
                    x1, y1 = max(0, cx - 0.5*w), max(0, cy - 0.5*h)
                    # x2, y2 = min(width, cx + 0.5*w), min(height, cy + 0.5*h)
                    x2, y2 = cx + 0.5*w, cy + 0.5*h
                    w , h = x2-x1 , y2-y1
                    new_rect = list(map(int, [x1, y1, w, h]))
                    rect = list(map(int, rect))
                    if (cid, tid) in map_tid:
                        new_tid = map_tid[(cid, tid)]
                        f_w.write(str(cid) + ' ' + str(new_tid) + ' ' + str(fid) + ' ' + ' '.join(map(str, new_rect)) + ' -1 -1' '\n')
    f_w.close()

def print_mtmct_result(gt_file, pred_file):
    gt = getData(gt_file, names=names)
    pred = getData(pred_file, names=names)
    summary = compare_dataframes_mtmc(gt, pred)
    print_results(summary)
