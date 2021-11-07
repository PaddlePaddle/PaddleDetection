import os
import numpy as np
import math

def st_name2label(names):
    labels = []
    cams = []
    frms = []
    for name in names:
        id,cam,frm,_ = os.path.basename(name).split('_')
        id = int(id)
        frm = int(frm)
        labels.append(id)
        cams.append(cam)
        frms.append(frm)
    labels = np.vstack(labels)
    frms = np.vstack(frms)
    return labels,cams,frms

def build_topo(sims,prb_names,gal_names,t_inv=100,dist_thrd=0.4):
    prb_labels,prb_cams,prb_frms = st_name2label(prb_names)
    gal_labels,gal_cams,gal_frms = st_name2label(gal_names)

    t_max = max(np.max(prb_frms)-np.min(gal_frms),np.max(gal_frms)-np.min(prb_frms))
    t_bins = int(np.ceil(t_max/float(t_inv)))
    camlist = list(set(prb_cams + gal_cams))
    topo = np.zeros((len(camlist),len(camlist),t_bins))

    label_sim = np.equal(prb_labels.reshape(-1,1),gal_labels.T)
    #p_idxs,g_idxs = np.where(label_sim==1)
    p_idxs,g_idxs = np.where(sims<=dist_thrd)
    for i in range(len(p_idxs)):
        p_idx,g_idx = p_idxs[i],g_idxs[i]
        p_cam,g_cam = prb_cams[p_idx],gal_cams[g_idx]
        p_frm,g_frm = prb_frms[p_idx],gal_frms[g_idx]
        p_idx,g_idx = camlist.index(p_cam),camlist.index(g_cam)
        t_bin = int(np.floor(abs(g_frm-p_frm)/float(t_inv)))
        topo[p_idx,g_idx,t_bin] += 1-label_sim[p_idx,g_idx]
        topo[g_idx,p_idx,t_bin] += 1-label_sim[p_idx,g_idx]
  
    return topo,camlist

def get_GaussKernal(inputs,var=50):
    return np.exp(-inputs**2/(2.*var**2))/(np.sqrt(2.*math.pi)*var)

def smooth_topo(cam_infos):
    #cam_probs = cam_infos/np.sum(cam_infos)

    # step 1: norm L1
    cam_probs = cam_infos/(np.sum(cam_infos,axis=2,keepdims=True)+1e-6)

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
    cam_probs = cam_probs/(np.sum(cam_probs,axis=2,keepdims=True)+1e-6)

    # step 4: reweight by cam x cam
    #CxC_sum = np.sum(cam_infos,axis=2,keepdims=True)
    #CxC_sum = CxC_sum/np.sum(CxC_sum)
    #CxC_sum = np.exp(CxC_sum)
    #CxC_weight = CxC_sum/np.sum(CxC_sum+1e-6)
    #cam_probs = CxC_weight * cam_probs

    # step 5: mask
    #cam_infos_mask = np.where(cam_infos==0,0,1)
    #cam_infos = cam_infos*cam_infos_mask
    return cam_probs

def compute_spacetime(sims,prb_names,gal_names,topo,camlist,t_inv=100):
    prb_labels,prb_cams,prb_frms = st_name2label(prb_names)
    gal_labels,gal_cams,gal_frms = st_name2label(gal_names)
    st_sims = np.zeros_like(sims)
    for i in range(sims.shape[0]):
        for j in range(sims.shape[1]):
            p_cam,g_cam = prb_cams[i],gal_cams[j]
            if p_cam in camlist and g_cam in camlist:
                p_frm,g_frm = prb_frms[i],gal_frms[j]
                p_idx,g_idx = camlist.index(p_cam),camlist.index(g_cam)
                t_bin = int(np.floor(abs(g_frm-p_frm)/float(t_inv)))
                st_sims[i,j] = topo[p_idx,g_idx,t_bin]
            else:
                st_sims[i,j] = 1./topo.shape[2]
    return st_sims

def add_spacetime(sims,prb_names,gal_names,t_inv=100,dist_thrd=0.4):
    topo,camlist = build_topo(sims,prb_names,gal_names,t_inv,dist_thrd)
    topo = smooth_topo(topo.copy())
    st_sims = compute_spacetime(sims,prb_names,gal_names,topo,camlist,t_inv)
    outsims = (1-sims) * st_sims
    return 1-outsims

