import re

def parse_pt(mot_feature):
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

def show_res(map_tid):
    show_dict = dict()
    for cid_tid in map_tid:
        iid = map_tid[cid_tid]
        if iid in show_dict:
            show_dict[iid].append(cid_tid)
        else:
            show_dict[iid] = [cid_tid]
    for i in show_dict:
        print('ID{}:{}'.format(i,show_dict[i]))

def gen_res(output_dir_filename, scene_cluster, map_tid, mot_features):
    f_w = open(output_dir_filename, 'w')
    for cid in scene_cluster:
        for mot_feature in mot_features:
            img_rects = parse_pt(mot_feature)
            for fid in img_rects:
                tid_rects = img_rects[fid]
                fid = int(fid)+1 # frameId add from 1
                for tid_rect in tid_rects:
                    tid = tid_rect[0]
                    rect = tid_rect[1:]
                    cx = 0.5*rect[0] + 0.5*rect[2]
                    cy = 0.5*rect[1] + 0.5*rect[3]
                    w = rect[2] - rect[0]
                    # w = min(w*1.2,w+40)
                    h = rect[3] - rect[1]
                    # h = min(h*1.2,h+40)
                    rect[2] -= rect[0]
                    rect[3] -= rect[1]
                    rect[0] = max(0, rect[0])
                    rect[1] = max(0, rect[1])
                    x1, y1 = max(0, cx - 0.5*w), max(0, cy - 0.5*h)
                    # x2, y2 = min(width, cx + 0.5*w), min(height, cy + 0.5*h)
                    x2, y2 = cx + 0.5*w, cy + 0.5*h
                    w , h = x2-x1 , y2-y1
                    new_rect = list(map(int, [x1, y1, w, h]))
                    # new_rect = rect
                    rect = list(map(int, rect))
                    if (cid, tid) in map_tid:
                        new_tid = map_tid[(cid, tid)]
                        f_w.write(str(cid) + ' ' + str(new_tid) + ' ' + str(fid) + ' ' + ' '.join(map(str, new_rect)) + ' -1 -1' '\n')
    f_w.close()


