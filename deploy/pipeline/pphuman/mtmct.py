# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from pptracking.python.mot.visualize import plot_tracking
from python.visualize import visualize_attr
import os
import re
import cv2
import gc
import numpy as np
try:
    from sklearn import preprocessing
    from sklearn.cluster import AgglomerativeClustering
except:
    print(
        'Warning: Unable to use MTMCT in PP-Human, please install sklearn, for example: `pip install sklearn`'
    )
    pass
import pandas as pd
from tqdm import tqdm
from functools import reduce
import warnings
warnings.filterwarnings("ignore")


def gen_restxt(output_dir_filename, map_tid, cid_tid_dict):
    pattern = re.compile(r'c(\d)_t(\d)')
    f_w = open(output_dir_filename, 'w')
    for key, res in cid_tid_dict.items():
        cid, tid = pattern.search(key).groups()
        cid = int(cid) + 1
        rects = res["rects"]
        frames = res["frames"]
        for idx, bbox in enumerate(rects):
            bbox[0][3:] -= bbox[0][1:3]
            fid = frames[idx] + 1
            rect = [max(int(x), 0) for x in bbox[0][1:]]
            if key in map_tid:
                new_tid = map_tid[key]
                f_w.write(
                    str(cid) + ' ' + str(new_tid) + ' ' + str(fid) + ' ' +
                    ' '.join(map(str, rect)) + '\n')
    print('gen_res: write file in {}'.format(output_dir_filename))
    f_w.close()


def get_mtmct_matching_results(pred_mtmct_file, secs_interval=0.5,
                               video_fps=20):
    res = np.loadtxt(pred_mtmct_file)  # 'cid, tid, fid, x1, y1, w, h, -1, -1'
    camera_ids = list(map(int, np.unique(res[:, 0])))

    res = res[:, :7]
    # each line in res: 'cid, tid, fid, x1, y1, w, h'

    camera_tids = []
    camera_results = dict()
    for c_id in camera_ids:
        camera_results[c_id] = res[res[:, 0] == c_id]
        tids = np.unique(camera_results[c_id][:, 1])
        tids = list(map(int, tids))
        camera_tids.append(tids)

    # select common tids throughout each video
    common_tids = reduce(np.intersect1d, camera_tids)

    # get mtmct matching results by cid_tid_fid_results[c_id][t_id][f_id]
    cid_tid_fid_results = dict()
    cid_tid_to_fids = dict()
    interval = int(secs_interval * video_fps)  # preferably less than 10
    for c_id in camera_ids:
        cid_tid_fid_results[c_id] = dict()
        cid_tid_to_fids[c_id] = dict()
        for t_id in common_tids:
            tid_mask = camera_results[c_id][:, 1] == t_id
            cid_tid_fid_results[c_id][t_id] = dict()

            camera_trackid_results = camera_results[c_id][tid_mask]
            fids = np.unique(camera_trackid_results[:, 2])
            fids = fids[fids % interval == 0]
            fids = list(map(int, fids))
            cid_tid_to_fids[c_id][t_id] = fids

            for f_id in fids:
                st_frame = f_id
                ed_frame = f_id + interval

                st_mask = camera_trackid_results[:, 2] >= st_frame
                ed_mask = camera_trackid_results[:, 2] < ed_frame
                frame_mask = np.logical_and(st_mask, ed_mask)
                cid_tid_fid_results[c_id][t_id][f_id] = camera_trackid_results[
                    frame_mask]

    return camera_results, cid_tid_fid_results


def save_mtmct_vis_results(camera_results, captures, output_dir,
                           multi_res=None):
    # camera_results: 'cid, tid, fid, x1, y1, w, h'
    camera_ids = list(camera_results.keys())

    import shutil
    save_dir = os.path.join(output_dir, 'mtmct_vis')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    for idx, video_file in enumerate(captures):
        capture = cv2.VideoCapture(video_file)
        cid = camera_ids[idx]
        basename = os.path.basename(video_file)
        video_out_name = "vis_" + basename
        out_path = os.path.join(save_dir, video_out_name)
        print("Start visualizing output video: {}".format(out_path))

        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        frame_id = 0
        while (1):
            if frame_id % 50 == 0:
                print('frame id: ', frame_id)
            ret, frame = capture.read()
            frame_id += 1
            if not ret:
                if frame_id == 1:
                    print("video read failed!")
                break
            frame_results = camera_results[cid][camera_results[cid][:, 2] ==
                                                frame_id]
            boxes = frame_results[:, -4:]
            ids = frame_results[:, 1]
            image = plot_tracking(frame, boxes, ids, frame_id=frame_id, fps=fps)

            # add attr vis
            if multi_res:
                tid_list = multi_res.keys()  # c0_t1, c0_t2...
                all_attr_result = [multi_res[i]["attrs"]
                                   for i in tid_list]  # all cid_tid result
                if any(
                        all_attr_result
                ):  # at least one cid_tid[attrs] is not None will goes to attrs_vis
                    attr_res = []
                    cid_str = 'c' + str(cid - 1) + "_"
                    for k in tid_list:
                        if not k.startswith(cid_str):
                            continue
                        if (frame_id - 1) >= len(multi_res[k]['attrs']):
                            t_attr = None
                        else:
                            t_attr = multi_res[k]['attrs'][frame_id - 1]
                            attr_res.append(t_attr)
                    assert len(attr_res) == len(boxes)
                    image = visualize_attr(
                        image, attr_res, boxes, is_mtmct=True)

            writer.write(image)
        writer.release()


def get_euclidean(x, y, **kwargs):
    m = x.shape[0]
    n = y.shape[0]
    distmat = (np.power(x, 2).sum(axis=1, keepdims=True).repeat(
        n, axis=1) + np.power(y, 2).sum(axis=1, keepdims=True).repeat(
            m, axis=1).T)
    distmat -= np.dot(2 * x, y.T)
    return distmat


def cosine_similarity(x, y, eps=1e-12):
    """
    Computes cosine similarity between two tensors.
    Value == 1 means the same vector
    Value == 0 means perpendicular vectors
    """
    x_n, y_n = np.linalg.norm(
        x, axis=1, keepdims=True), np.linalg.norm(
            y, axis=1, keepdims=True)
    x_norm = x / np.maximum(x_n, eps * np.ones_like(x_n))
    y_norm = y / np.maximum(y_n, eps * np.ones_like(y_n))
    sim_mt = np.dot(x_norm, y_norm.T)
    return sim_mt


def get_cosine(x, y, eps=1e-12):
    """
    Computes cosine distance between two tensors.
    The cosine distance is the inverse cosine similarity
    -> cosine_distance = abs(-cosine_distance) to make it
    similar in behavior to euclidean distance
    """
    sim_mt = cosine_similarity(x, y, eps)
    return sim_mt


def get_dist_mat(x, y, func_name="euclidean"):
    if func_name == "cosine":
        dist_mat = get_cosine(x, y)
    elif func_name == "euclidean":
        dist_mat = get_euclidean(x, y)
    print("Using {} as distance function during evaluation".format(func_name))
    return dist_mat


def intracam_ignore(st_mask, cid_tids):
    count = len(cid_tids)
    for i in range(count):
        for j in range(count):
            if cid_tids[i][1] == cid_tids[j][1]:
                st_mask[i, j] = 0.
    return st_mask


def get_sim_matrix_new(cid_tid_dict, cid_tids):
    # Note: camera independent get_sim_matrix function,
    # which is different from the one in camera_utils.py.
    count = len(cid_tids)

    q_arr = np.array(
        [cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    g_arr = np.array(
        [cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    #compute distmat
    distmat = get_dist_mat(q_arr, g_arr, func_name="cosine")

    #mask the element which belongs to same video
    st_mask = np.ones((count, count), dtype=np.float32)
    st_mask = intracam_ignore(st_mask, cid_tids)

    sim_matrix = distmat * st_mask
    np.fill_diagonal(sim_matrix, 0.)
    return 1. - sim_matrix


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


def get_labels(cid_tid_dict, cid_tids):
    #compute cost matrix between features
    cost_matrix = get_sim_matrix_new(cid_tid_dict, cid_tids)

    #cluster all the features
    cluster1 = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.5,
        affinity='precomputed',
        linkage='complete')
    cluster_labels1 = cluster1.fit_predict(cost_matrix)
    labels = get_match(cluster_labels1)

    sub_cluster = get_cid_tid(labels, cid_tids)
    return labels


def sub_cluster(cid_tid_dict):
    '''
    cid_tid_dict: all camera_id and track_id
    '''
    #get all keys
    cid_tids = sorted([key for key in cid_tid_dict.keys()])

    #cluster all trackid
    clu = get_labels(cid_tid_dict, cid_tids)

    #relabel every cluster groups
    new_clu = list()
    for c_list in clu:
        new_clu.append([cid_tids[c] for c in c_list])
    cid_tid_label = dict()
    for i, c_list in enumerate(new_clu):
        for c in c_list:
            cid_tid_label[c] = i + 1
    return cid_tid_label


def distill_idfeat(mot_res):
    qualities_list = mot_res["qualities"]
    feature_list = mot_res["features"]
    rects = mot_res["rects"]

    qualities_new = []
    feature_new = []
    #filter rect less than 100*20
    for idx, rect in enumerate(rects):
        conf, xmin, ymin, xmax, ymax = rect[0]
        if (xmax - xmin) * (ymax - ymin) and (xmax > xmin) > 2000:
            qualities_new.append(qualities_list[idx])
            feature_new.append(feature_list[idx])
    #take all features if available rect is less than 2
    if len(qualities_new) < 2:
        qualities_new = qualities_list
        feature_new = feature_list

    #if available frames number is more than 200, take one frame data per 20 frames
    skipf = 1
    if len(qualities_new) > 20:
        skipf = 2
    quality_skip = np.array(qualities_new[::skipf])
    feature_skip = np.array(feature_new[::skipf])

    #sort features with image qualities, take the most trustworth features
    topk_argq = np.argsort(quality_skip)[::-1]
    if (quality_skip > 0.6).sum() > 1:
        topk_feat = feature_skip[topk_argq[quality_skip > 0.6]]
    else:
        topk_feat = feature_skip[topk_argq]

    #get final features by mean or cluster, at most take five
    mean_feat = np.mean(topk_feat[:5], axis=0)
    return mean_feat


def res2dict(multi_res):
    cid_tid_dict = {}
    for cid, c_res in enumerate(multi_res):
        for tid, res in c_res.items():
            key = "c" + str(cid) + "_t" + str(tid)
            if key not in cid_tid_dict:
                if len(res["features"]) == 0:
                    continue
                cid_tid_dict[key] = res
                cid_tid_dict[key]['mean_feat'] = distill_idfeat(res)
    return cid_tid_dict


def mtmct_process(multi_res, captures, mtmct_vis=True, output_dir="output"):
    cid_tid_dict = res2dict(multi_res)
    if len(cid_tid_dict) == 0:
        print("no tracking result found, mtmct will be skiped.")
        return
    map_tid = sub_cluster(cid_tid_dict)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    pred_mtmct_file = os.path.join(output_dir, 'mtmct_result.txt')
    gen_restxt(pred_mtmct_file, map_tid, cid_tid_dict)

    if mtmct_vis:
        camera_results, cid_tid_fid_res = get_mtmct_matching_results(
            pred_mtmct_file)

        save_mtmct_vis_results(
            camera_results,
            captures,
            output_dir=output_dir,
            multi_res=cid_tid_dict)
