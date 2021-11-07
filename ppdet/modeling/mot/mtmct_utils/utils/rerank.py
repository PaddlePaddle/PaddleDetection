import numpy as np
import os
import scipy.misc
from glob import glob
from PIL import Image
import sys
import copy
from scipy.spatial.distance import cdist
import torch as paddle
# import paddle
import time
import gc
from tqdm import tqdm

def euclidean_distance(qf, gf):

    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = 2 - 2 * paddle.matmul(qf, gf.t())
    return dist_mat

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
        temp_qd = paddle.cat(temp_qd, dim=0)
        temp_qd = temp_qd / (paddle.max(temp_qd, dim=0)[0])
        dist_mat.append(temp_qd.t().cpu())
    del temp_qd
    del temp_gf
    del temp_qf
    del temp_d
    paddle.cuda.empty_cache()  # empty GPU memory
    dist_mat = paddle.cat(dist_mat, dim=0)
    return dist_mat

# Compute TopK in GPU and return (k1+1) results
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
        temp_qd = paddle.cat(temp_qd, dim=0)
        temp_qd = temp_qd / (paddle.max(temp_qd, dim=0)[0])
        temp_qd = temp_qd.t()
        initial_rank.append(paddle.topk(temp_qd, k=k1, dim=1, largest=False, sorted=True)[1])

    del temp_qd
    del temp_gf
    del temp_qf
    del temp_d
    paddle.cuda.empty_cache()  # empty GPU memory
    initial_rank = paddle.cat(initial_rank, dim=0).cpu().numpy()
    return initial_rank

def batch_v(feat, R, all_num):
    V = np.zeros((all_num, all_num), dtype=np.float32)
    m = feat.shape[0]
    for i in tqdm(range(m)):
        temp_gf = feat[i].unsqueeze(0)
        # temp_qd = []
        temp_qd = euclidean_distance(temp_gf, feat)
        temp_qd = temp_qd / (paddle.max(temp_qd))
        temp_qd = temp_qd.squeeze()
        temp_qd = temp_qd[R[i]]
        weight = paddle.exp(-temp_qd)
        weight = (weight / paddle.sum(weight)).cpu().numpy()
        V[i, R[i]] = weight.astype(np.float32)
    return V


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]

def ReRank1(probFea,galFea,k1=20,k2=6,lambda_value=0.3):

    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]    
    feat = np.append(probFea,galFea,axis = 0)
    feat = feat.astype(np.float32)
    print('computing original distance')
    
    original_dist = cdist(feat,feat).astype(np.float32)  
    original_dist = np.power(original_dist,2).astype(np.float32)
    del feat    
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
            
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = weight/np.sum(weight)
    original_dist = original_dist[:query_num,]    
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])
    
    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    
    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2-temp_min)
    
    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist

def ReRank2(probFea, galFea, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    t1 = time.time()
    # query_num = probFea.size(0)
    # all_num = query_num + galFea.size(0)
    print(type(probFea), probFea.shape)
    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]
    feat = paddle.cat([probFea, galFea]).cuda()
    initial_rank = batch_paddle_topk(feat, feat, k1 + 1, N=6000)
    # del feat
    del probFea
    del galFea
    paddle.cuda.empty_cache()  # empty GPU memory
    gc.collect()  # empty memory
    print('Using totally {:.2f}s to compute initial_rank'.format(time.time() - t1))
    print('starting re_ranking')

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
    print('Using totally {:.2f}S to compute R'.format(time.time() - t1))
    V = batch_v(feat, R, all_num)
    del R
    gc.collect()  # empty memory
    print('Using totally {:.2f}S to compute V-1'.format(time.time() - t1))
    initial_rank = initial_rank[:, :k2]

    ### Faster version
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank

    ### Low-memory version
    '''gc.collect()  # empty memory
    N = 2000
    for j in range(all_num // N + 1):

        if k2 != 1:
            V_qe = np.zeros_like(V[:, j * N:j * N + N], dtype=np.float32)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i], j * N:j * N + N], axis=0)
            V[:, j * N:j * N + N] = V_qe
            del V_qe
    del initial_rank'''

    gc.collect()  # empty memory
    print('Using totally {:.2f}S to compute V-2'.format(time.time() - t1))
    invIndex = []

    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])
    print('Using totally {:.2f}S to compute invIndex'.format(time.time() - t1))

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
    # print(jaccard_dist)
    del original_dist

    del jaccard_dist

    final_dist = final_dist[:query_num, query_num:]
    print(final_dist)
    print('Using totally {:.2f}S to compute final_distance'.format(time.time() - t1))
    return final_dist

