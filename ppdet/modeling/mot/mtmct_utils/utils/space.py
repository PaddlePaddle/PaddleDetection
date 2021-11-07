import os
import numpy as np
import math

def st_name2label(names):
    labels = []
    cams = []
    for name in names:
        name = name[0].decode()
        id,cam,_ = name.split('_')
        id = int(id)
        labels.append(id)
        cams.append(cam)
    labels = np.vstack(labels)
    return labels,cams

def build_topo(sims,prb_names,gal_names,dist_thrd=0.4):
    prb_labels,prb_cams = st_name2label(prb_names)
    gal_labels,gal_cams = st_name2label(gal_names)

    camlist = list(set(prb_cams + gal_cams))
    topo = np.zeros((len(camlist),len(camlist)))

    #label_sim = np.equal(prb_labels.reshape(-1,1),gal_labels.T)
    #p_idxs,g_idxs = np.where(label_sim==1)
    label_sim = sims
    p_idxs,g_idxs = np.where(sims<=dist_thrd)
    for i in range(len(p_idxs)):
        p_idx,g_idx = p_idxs[i],g_idxs[i]
        p_cam,g_cam = prb_cams[p_idx],gal_cams[g_idx]
        p_camidx,g_camidx = camlist.index(p_cam),camlist.index(g_cam)
        topo[p_camidx,g_camidx] += 1#-label_sim[p_idx,g_idx]
        topo[g_camidx,p_camidx] += 1#-label_sim[p_idx,g_idx]
 
    return topo,camlist

def get_GaussKernal(inputs,var=50):
    return np.exp(-inputs**2/(2.*var**2))/(np.sqrt(2.*math.pi)*var)

def smooth_topo(cam_infos):
    cam_probs = cam_infos/(np.sum(cam_infos)+1e-8)

    # step 1: norm L1
    #cam_probs = cam_infos/(np.sum(cam_infos,axis=2,keepdims=True)+1e-6)

    '''
    # step 2: smooth
    var = 1.0
    tmp_inv = np.arange(0,cam_probs.shape[2],1)
    tmp_inv = np.expand_dims(tmp_inv,1) - np.expand_dims(tmp_inv,0)
    tmp_inv = get_GaussKernal(tmp_inv,var)
    for i in xrange(0,cam_probs.shape[0]):
        for j in xrange(i,cam_probs.shape[0]):
            tmp_prob = cam_probs[i,j,:]
            tmp_prob = tmp_prob.reshape(-1,1)*tmp_inv
            tmp_prob = np.sum(tmp_prob,axis=0,keepdims=False)
            cam_probs[i,j,:] = tmp_prob
            cam_probs[j,i,:] = tmp_prob
    cam_probs = cam_probs/(np.sum(cam_probs,axis=2,keepdims=True)+1e-6)
    '''

    # step 3: exp
    cam_probs = np.exp(cam_probs)
    cam_probs = cam_probs/(np.sum(cam_probs)+1e-8)

    # step 4: reweight by cam x cam
    #CxC_sum = np.sum(cam_infos,axis=2,keepdims=True)
    #CxC_sum = CxC_sum/np.sum(CxC_sum)
    #CxC_sum = np.exp(CxC_sum)
    #CxC_weight = CxC_sum/np.sum(CxC_sum+1e-6)
    #cam_probs = CxC_weight * cam_probs

    # step 5: mask
    cam_infos_mask = np.where(cam_infos==0,0,1)
    cam_probs = cam_probs*cam_infos_mask
    return cam_probs

def compute_space(sims,prb_names,gal_names,topo,camlist):
    prb_labels,prb_cams = st_name2label(prb_names)
    gal_labels,gal_cams = st_name2label(gal_names)
    st_sims = np.zeros_like(sims)
    for i in range(sims.shape[0]):
        for j in range(sims.shape[1]):
            p_cam,g_cam = prb_cams[i],gal_cams[j]
            if p_cam in camlist and g_cam in camlist:
                p_idx,g_idx = camlist.index(p_cam),camlist.index(g_cam)
                st_sims[i,j] = topo[p_idx,g_idx]
            else:
                st_sims[i,j] = 1./(topo.shape[0]*topo.shape[1])
    return st_sims

def add_space(sims,prb_names,gal_names,dist_thrd=0.4):
    topo,camlist = build_topo(sims,prb_names,gal_names,dist_thrd)
    topo = smooth_topo(topo.copy())
    st_sims = compute_space(sims,prb_names,gal_names,topo,camlist)
    outsims = (1-sims) * st_sims
    return 1-outsims

