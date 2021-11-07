import numpy as np

def compute_P(prb_feats,gal_feats):
    '''
    if prb_feats.shape[0] == gal_feats.shape[0]:
        if (prb_feats == gal_feats).all():
            X = prb_feats
    else:
        X = np.vstack([prb_feats,gal_feats])
    '''
    X = gal_feats
    neg_vector = np.mean(X,axis=0)
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
        if i%1000==0:
            print('feat3:%d/%d' %(i,X.shape[0]))
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
        #if len(out_feats) % 100 == 0:
        #    print('%d/%d' %(len(out_feats),in_feats.shape[0]))
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
        if len(out_feats) % 1000 == 0:
            print('%d/%d' %(len(out_feats),len(trackset)))
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
        if len(out_feats) % 1000 == 0:
            print('%d/%d' %(len(out_feats),len(trackset)))
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

