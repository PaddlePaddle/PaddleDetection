"""Visual rerank by multiple tricks."""
import numpy as np

# def ComputeEuclid(array1,array2,fg_sqrt):
#     #array1:[m1,n],array2:[m2,n]
#     assert array1.shape[1]==array2.shape[1];
#     # shape [m1, 1]
#     square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
#     # shape [1, m2]
#     square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
#     #print 'array1,array2 shape:',array1.shape,array2.shape
#     squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
#     squared_dist[squared_dist < 0] = 0
#     #shape [m1,m2]
#     if fg_sqrt:
#         dist = np.sqrt(squared_dist)
#         #print('[test] using sqrt for distance')
#     else:
#         dist = squared_dist
#         #print('[test] not using sqrt for distance')
#     sim = 1/(1+dist)
#     return 1-sim

def visual_rerank(prb_feats, gal_feats, cid_tids, use_ff=False, use_rerank=False):
    """Rerank by visual cures."""

    gal_labels = np.array([[0, item[0]] for item in cid_tids])
    prb_labels = gal_labels.copy()
    # if use_ff:
    #     # Step1-1: fic. finetuned parameters: [la]
    #     prb_feats, gal_feats = ff.run_fic(prb_feats, gal_feats,
    #                                       prb_labels, gal_labels, 3.0)
    #     # Step1=2: fac. finetuned parameters: [beta,knn,lr,prb_epoch,gal_epoch]
    #     prb_feats, gal_feats = ff.run_fac(prb_feats, gal_feats,
    #                                       prb_labels, gal_labels,
    #                                       0.08, 20, 0.5, 1, 1)
    # if use_rerank:
    #     # Step2: k-reciprocal. finetuned parameters: [k1,k2,lambda_value]
    #     sims = rr.ReRank2(torch.from_numpy(prb_feats).cuda(),
    #                       torch.from_numpy(gal_feats).cuda(), 20, 3, 0.3)
    # else:
    #     # sims = ComputeEuclid(prb_feats, gal_feats, 1)
    #     sims = 1.0 - np.dot(prb_feats, gal_feats.T)

    # # NOTE: sims here is actually dist, the smaller the more similar

    sims = 1.0 - np.dot(prb_feats, gal_feats.T)

    return 1.0 - sims
