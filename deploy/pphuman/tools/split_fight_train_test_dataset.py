import os
import glob
import random
import fnmatch
import re
import sys

class_id = {"nofight": 0, "fight": 1}


def get_list(path, key_func=lambda x: x[-11:], rgb_prefix='img_', level=1):
    if level == 1:
        frame_folders = glob.glob(os.path.join(path, '*'))
    elif level == 2:
        frame_folders = glob.glob(os.path.join(path, '*', '*'))
    else:
        raise ValueError('level can be only 1 or 2')

    def count_files(directory):
        lst = os.listdir(directory)
        cnt = len(fnmatch.filter(lst, rgb_prefix + '*'))
        return cnt

    # check RGB
    video_dict = {}
    for f in frame_folders:
        cnt = count_files(f)
        k = key_func(f)
        if level == 2:
            k = k.split("/")[0]

        video_dict[f] = str(cnt) + " " + str(class_id[k])

    return video_dict


def fight_splits(video_dict, train_percent=0.8):
    videos = list(video_dict.keys())

    train_num = int(len(videos) * train_percent)

    train_list = []
    val_list = []

    random.shuffle(videos)

    for i in range(train_num):
        train_list.append(videos[i] + " " + str(video_dict[videos[i]]))
    for i in range(train_num, len(videos)):
        val_list.append(videos[i] + " " + str(video_dict[videos[i]]))

    print("train:", len(train_list), ",val:", len(val_list))

    with open("fight_train_list.txt", "w") as f:
        for item in train_list:
            f.write(item + "\n")

    with open("fight_val_list.txt", "w") as f:
        for item in val_list:
            f.write(item + "\n")


if __name__ == "__main__":
    frame_dir = sys.argv[1]  # "rawframes"
    level = sys.argv[2]  # 2
    train_percent = sys.argv[3]  # 0.8

    if level == 2:

        def key_func(x):
            return '/'.join(x.split('/')[-2:])
    else:

        def key_func(x):
            return x.split('/')[-1]

    video_dict = get_list(frame_dir, key_func=key_func, level=level)
    print("number:", len(video_dict))

    fight_splits(video_dict, train_percent)
