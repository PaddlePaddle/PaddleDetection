
def subcam_list(cid_tid_dict,cid_tids):
    '''
    sub_cid_tids:[(41,42):[],(41,43):[]]
    '''
    sub_3_4 = dict()
    sub_4_3 = dict()
    for cid_tid in cid_tids:
        cid,tid = cid_tid
        tracklet = cid_tid_dict[cid_tid]
        zs,ze = get_dire(tracklet['zone_list'], cid)
        if zs in [3] and cid not in [46]: # 4 to 3
            if not cid+1 in sub_4_3:
                sub_4_3[cid+1] = []
            sub_4_3[cid + 1].append(cid_tid)
        if ze in [4] and cid not in [41]: # 4 to 3
            if not cid in sub_4_3:
                sub_4_3[cid] = []
            sub_4_3[cid].append(cid_tid)
        if zs in [4] and cid not in [41]: # 3 to 4
            if not cid-1 in sub_3_4:
                sub_3_4[cid-1] = []
            sub_3_4[cid - 1].append(cid_tid)
        if ze in [3] and cid not in [46]: # 3 to 4
            if not cid in sub_3_4:
                sub_3_4[cid] = []
            sub_3_4[cid].append(cid_tid)
    sub_cid_tids = dict()
    for i in sub_3_4:
        sub_cid_tids[(i,i+1)]=sub_3_4[i]
    for i in sub_4_3:
        sub_cid_tids[(i,i-1)]=sub_4_3[i]
    return sub_cid_tids

def subcam_list2(cid_tid_dict,cid_tids):
    sub_dict = dict()
    for cid_tid in cid_tids:
        cid, tid = cid_tid
        if cid not in [41]:
            if not cid in sub_dict:
                sub_dict[cid] = []
            sub_dict[cid].append(cid_tid)
        if cid not in [46]:
            if not cid+1 in sub_dict:
                sub_dict[cid+1] = []
            sub_dict[cid+1].append(cid_tid)
    return sub_dict

def intracam_ignore(st_mask, cid_tids):
    count = len(cid_tids)
    for i in range(count):
        for j in range(count):
            if cid_tids[i][0] == cid_tids[j][0]:
                st_mask[i, j] = 0.
    return st_mask
