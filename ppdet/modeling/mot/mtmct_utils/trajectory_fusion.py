
import os
from os.path import join as opj
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import re
import sys
sys.path.append('ppdet/engine/mtmct_utils')

from ppdet.modeling.mot.mtmct_utils.utils.visual_rr import visual_rerank


def parse_pt(mot_feature):
    mot_list = dict()
    for line in mot_feature:
        fid = int(re.sub('[a-z,A-Z]',"",mot_feature[line]['frame']))
        tid = mot_feature[line]['id']
        bbox = list(map(lambda x:int(float(x)), mot_feature[line]['bbox']))
        if tid not in mot_list:
            mot_list[tid] = dict()
        out_dict = mot_feature[line]
        # out_dict['zone'] = zones.get_zone(bbox)
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
    # zones.set_cam(cid)
    mot_list = parse_pt(mot_feature)
    # mot_list = zones.break_mot(mot_list, cid)
    # mot_list = zones.comb_mot(mot_list, cid)
    # mot_list = zones.filter_mot(mot_list, cid) # filter by zone
    # mot_list = zones.filter_bbox(mot_list, cid)  # filter bbox
    tid_data = dict()
    for tid in mot_list:
        tracklet = mot_list[tid]
        if len(tracklet) <= 1: continue
        frame_list = list(tracklet.keys())
        frame_list.sort()
        # zone_list = [tracklet[f]['zone'] for f in frame_list]
        feature_list = [tracklet[f]['feat'] for f in frame_list if (tracklet[f]['bbox'][3]-tracklet[f]['bbox'][1])*(tracklet[f]['bbox'][2]-tracklet[f]['bbox'][0])>2000]
        if len(feature_list)<2:
            feature_list = [tracklet[f]['feat'] for f in frame_list]
        io_time = [cur_bias + frame_list[0] / 10., cur_bias + frame_list[-1] / 10.]
        all_feat = np.array([feat for feat in feature_list])
        mean_feat = np.mean(all_feat, axis=0)
        tid_data[tid]={
            'cam': cid,
            'tid': tid,
            'mean_feat': mean_feat,
            'frame_list': frame_list,
            'tracklet': tracklet,
            'io_time': io_time
        }
    return tid_data




if __name__ == "__main__":
    cameras_bias={"c041":1.2,"c042":2.5,"c043":3,"c044":4}
    ret = parse_bias(cameras_bias)
    print(ret)
 


