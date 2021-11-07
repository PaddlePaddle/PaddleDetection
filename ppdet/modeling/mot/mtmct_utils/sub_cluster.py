
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing

from ppdet.modeling.mot.mtmct_utils.utils.filter import intracam_ignore,subcam_list2
from ppdet.modeling.mot.mtmct_utils.utils.visual_rr import visual_rerank


def get_cid_tid(cluster_labels,cid_tids):
    cluster = list()
    for labels in cluster_labels:
        cid_tid_list = list()
        for label in labels:
            cid_tid_list.append(cid_tids[label])
        cluster.append(cid_tid_list)
    return cluster


def normalize(nparray, axis=0):
    nparray = preprocessing.normalize(nparray, norm='l2', axis=axis)
    return nparray


def get_sim_matrix(cid_tid_dict, cid_tids):
    count = len(cid_tids)
    q_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    g_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    q_arr = normalize(q_arr, axis=1)
    g_arr = normalize(g_arr, axis=1)
    # sim_matrix = np.matmul(q_arr, g_arr.T)
    # st mask
    st_mask = np.ones((count, count), dtype=np.float32)
    st_mask = intracam_ignore(st_mask, cid_tids)
    # st_mask = st_filter(st_mask, cid_tids, cid_tid_dict) # use zone match

    # visual rerank
    visual_sim_matrix = visual_rerank(q_arr, g_arr, cid_tids)
    visual_sim_matrix = visual_sim_matrix.astype('float32')
    # merge result
    np.set_printoptions(precision=3)
    sim_matrix = visual_sim_matrix * st_mask
    # sim_matrix[sim_matrix < 0] = 0
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
    print("new tricklets:{}".format(num_tr))
    return labels,cluster

def combin_feature(cid_tid_dict,sub_cluster):
    for sub_ct in sub_cluster:
        if len(sub_ct)<2: continue
        mean_feat = np.array([cid_tid_dict[i]['mean_feat'] for i in sub_ct])
        for i in sub_ct:
            cid_tid_dict[i]['mean_feat'] = mean_feat.mean(axis=0)
    return cid_tid_dict

def get_labels(cid_tid_dict, cid_tids, score_thr):
    # 1st cluster subcam_list 本质上留下的都是否cid_tid_dict的key
    # sub_cid_tids = subcam_list(cid_tid_dict,cid_tids)
    sub_cid_tids = list(cid_tid_dict.keys())
    sub_labels = dict()
    sim_matrix = get_sim_matrix(cid_tid_dict,cid_tids)
    cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
    labels = get_match(cluster_labels)
    sub_cluster = get_cid_tid(labels,cid_tids)
    print('labels => ',labels)
    print('sub_cluster =>', sub_cluster)
    # 2ed cluster
    cid_tid_dict_new = combin_feature(cid_tid_dict, sub_cluster)
    sub_cid_tids = subcam_list2(cid_tid_dict_new,cid_tids)
    sub_labels = dict()
    sim_matrix = get_sim_matrix(cid_tid_dict_new,cid_tids)
    cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=0.9, affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
    labels = get_match(cluster_labels)
    sub_cluster = get_cid_tid(labels,cid_tids)

    # 3rd cluster
    # cid_tid_dict_new = combin_feature(cid_tid_dict,sub_cluster)
    # sim_matrix = get_sim_matrix(_cfg,cid_tid_dict_new, cid_tids)
    # cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - 0.2, affinity='precomputed',
    #                                          linkage='complete').fit_predict(1 - sim_matrix)
    # labels = get_match(cluster_labels)
    return labels


def sub_cluster(cid_tid_dict, scene_cluster, score_thr):
    '''
    cid_tid_dict: all camera_id and track_id
    scene_cluster: like [41, 42, 43, 44, 45, 46]
    score_thr: config param
    '''
    cid_tids = sorted([key for key in cid_tid_dict.keys() if key[0] in scene_cluster])
    clu = get_labels(cid_tid_dict,cid_tids,score_thr=score_thr)
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






