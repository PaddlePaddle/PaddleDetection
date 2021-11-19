# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import os.path as osp
import cv2
import random
import numpy as np
import argparse
import tqdm
import json


def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)


def bdd2mot_tracking(img_dir, label_dir, save_img_dir, save_label_dir):
    label_jsons = os.listdir(label_dir)
    for label_json in tqdm(label_jsons):
        with open(os.path.join(label_dir, label_json)) as f:
            labels_json = json.load(f)
            for label_json in labels_json:
                img_name = label_json['name']
                video_name = label_json['videoName']
                labels = label_json['labels']
                txt_string = ""
                for label in labels:
                    category = label['category']
                    x1 = label['box2d']['x1']
                    x2 = label['box2d']['x2']
                    y1 = label['box2d']['y1']
                    y2 = label['box2d']['y2']
                    width = x2 - x1
                    height = y2 - y1
                    x_center = (x1 + x2) / 2. / args.width
                    y_center = (y1 + y2) / 2. / args.height
                    width /= args.width
                    height /= args.height
                    identity = int(label['id'])
                    # [class] [identity] [x_center] [y_center] [width] [height]
                    txt_string += "{} {} {} {} {} {}\n".format(
                        attr_id_dict[category], identity, x_center, y_center,
                        width, height)

                fn_label = os.path.join(save_label_dir, img_name[:-4] + '.txt')
                source_img = os.path.join(img_dir, video_name, img_name)
                target_img = os.path.join(save_img_dir, img_name)
                with open(fn_label, 'w') as f:
                    f.write(txt_string)
                os.system('cp {} {}'.format(source_img, target_img))


def transBbox(bbox):
    # bbox --> cx cy w h
    bbox = list(map(lambda x: float(x), bbox))
    bbox[0] = (bbox[0] - bbox[2] / 2) * 1280
    bbox[1] = (bbox[1] - bbox[3] / 2) * 720
    bbox[2] = bbox[2] * 1280
    bbox[3] = bbox[3] * 720

    bbox = list(map(lambda x: str(x), bbox))
    return bbox


def genSingleImageMot(inputPath, classes=[]):
    labelPaths = glob.glob(inputPath + '/*.txt')
    labelPaths = sorted(labelPaths)
    allLines = []
    result = {}
    for labelPath in labelPaths:
        frame = str(int(labelPath.split('-')[-1].replace('.txt', '')))
        with open(labelPath, 'r') as labelPathFile:
            lines = labelPathFile.readlines()
            for line in lines:
                line = line.replace('\n', '')
                lineArray = line.split(' ')
                if len(classes) > 0:
                    if lineArray[0] in classes:
                        lineArray.append(frame)
                        allLines.append(lineArray)
                else:
                    lineArray.append(frame)
                    allLines.append(lineArray)
    resultMap = {}
    for line in allLines:
        if line[1] not in resultMap.keys():
            resultMap[line[1]] = []
        resultMap[line[1]].append(line)
    mot_gt = []
    id_idx = 0
    for rid in resultMap.keys():
        id_idx += 1
        for id_line in resultMap[rid]:
            mot_line = []
            mot_line.append(id_line[-1])
            mot_line.append(str(id_idx))
            id_line_temp = transBbox(id_line[2:6])
            mot_line.extend(id_line_temp)
            mot_line.append('1') # origin class: id_line[0]
            mot_line.append('1')  # permanent class  => 1
            mot_line.append('1')
            mot_gt.append(mot_line)

    result = list(map(lambda line: str.join(',', line), mot_gt))
    resultStr = str.join('\n', result)
    return resultStr


def writeGt(inputPath, outPath, classes=[]):
    singleImageResult = genSingleImageMot(inputPath, classes=classes)
    outPathFile = outPath + '/gt.txt'
    mkdir_if_missing(outPath)
    with open(outPathFile, 'w') as gtFile:
        gtFile.write(singleImageResult)


def genSeqInfo(seqInfoPath):
    name = seqInfoPath.split('/')[-2]
    img1Path = osp.join(str.join('/', seqInfoPath.split('/')[0:-1]), 'img1')
    seqLength = len(glob.glob(img1Path + '/*.jpg'))
    seqInfoStr = f'''[Sequence]\nname={name}\nimDir=img1\nframeRate=30\nseqLength={seqLength}\nimWidth=1280\nimHeight=720\nimExt=.jpg'''
    with open(seqInfoPath, 'w') as seqFile:
        seqFile.write(seqInfoStr)


def genMotGt(dataDir, classes=[]):
    seqLists = sorted(glob.glob(dataDir))
    for seqList in seqLists:
        inputPath = osp.join(seqList, 'img1')
        outputPath = seqList.replace('labels_with_ids', 'images')
        outputPath = osp.join(outputPath, 'gt')
        mkdir_if_missing(outputPath)
        print('processing...', outputPath)
        writeGt(inputPath, outputPath, classes=classes)
        seqList = seqList.replace('labels_with_ids', 'images')
        seqInfoPath = osp.join(seqList, 'seqinfo.ini')
        genSeqInfo(seqInfoPath)


def updateSeqInfo(dataDir, phase):
    seqPath = osp.join(dataDir, 'labels_with_ids', phase)
    seqList = glob.glob(seqPath + '/*')
    for seqName in seqList:
        print('seqName=>', seqName)
        seqName_img1_dir = osp.join(seqName, 'img1')
        txtLength = glob.glob(seqName_img1_dir + '/*.txt')
        name = seqName.split('/')[-1].replace('.jpg', '').replace('.txt', '')
        seqLength = len(txtLength)
        seqInfoStr = f'''[Sequence]\nname={name}\nimDir=img1\nframeRate=30\nseqLength={seqLength}\nimWidth=1280\nimHeight=720\nimExt=.jpg'''
        seqInfoPath = seqName_img1_dir.replace('labels_with_ids', 'images')
        seqInfoPath = seqInfoPath.replace('/img1', '')
        seqInfoPath = seqInfoPath + '/seqinfo.ini'
        with open(seqInfoPath, 'w') as seqFile:
            seqFile.write(seqInfoStr)


def VisualDataset(datasetPath, phase='train', seqName='', frameId=1):
    trainPath = osp.join(datasetPath, 'labels_with_ids', phase)
    seq1Paths = osp.join(trainPath, seqName)
    seq_img1_path = osp.join(seq1Paths, 'img1')
    label_with_idPath = osp.join(seq_img1_path, seqName + '-' + '%07d' %
                                 frameId) + '.txt'
    image_path = label_with_idPath.replace('labels_with_ids', 'images').replace(
        '.txt', '.jpg')

    seqInfoPath = str.join('/', image_path.split('/')[:-2])
    seqInfoPath = seqInfoPath + '/seqinfo.ini'
    seq_info = open(seqInfoPath).read()
    width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find(
        '\nimHeight')])
    height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find(
        '\nimExt')])

    with open(label_with_idPath, 'r') as label:
        allLines = label.readlines()
        images = cv2.imread(image_path)
        print('image_path => ', image_path)
        for line in allLines:
            line = line.split(' ')
            line = list(map(lambda x: float(x), line))
            c1, c2, w, h = line[2:6]
            x1 = c1 - w / 2
            x2 = c2 - h / 2
            x3 = c1 + w / 2
            x4 = c2 + h / 2
            cv2.rectangle(
                images, (int(x1 * width), int(x2 * height)),
                (int(x3 * width), int(x4 * height)), (255, 0, 0),
                thickness=2)
        cv2.imwrite('test.jpg', images)


def VisualGt(dataPath, phase='train'):
    seqList = sorted(glob.glob(osp.join(dataPath, 'images', phase) + '/*'))
    seqIndex = random.randint(0, len(seqList) - 1)
    seqPath = seqList[seqIndex]
    gt_path = osp.join(seqPath, 'gt', 'gt.txt')
    img_list_path = sorted(glob.glob(osp.join(seqPath, 'img1', '*.jpg')))
    imgIndex = random.randint(0, len(img_list_path))
    img_Path = img_list_path[imgIndex]

    frame_value = img_Path.split('/')[-1].replace('.jpg', '')
    frame_value = frame_value.split('-')[-1]
    frame_value = int(frame_value)
    seqNameStr = img_Path.split('/')[-1].replace('.jpg', '').replace('img', '')
    frame_value = int(seqNameStr.split('-')[-1])
    print('frame_value => ', frame_value)
    gt_value = np.loadtxt(gt_path, dtype=float, delimiter=',')
    gt_value = gt_value[gt_value[:, 0] == frame_value]

    get_list = gt_value.tolist()
    img = cv2.imread(img_Path)

    colors = [[255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 255, 0],
              [0, 255, 255], [0, 0, 255]]
    for seq, _id, pl, pt, w, h, _, bbox_class, _ in get_list:
        pl, pt, w, h = int(pl), int(pt), int(w), int(h)
        print('pl,pt,w,h => ', pl, pt, w, h)
        cv2.putText(img,
                    str(bbox_class), (pl, pt), cv2.FONT_HERSHEY_PLAIN, 2,
                    colors[int(bbox_class - 1)])
        cv2.rectangle(
            img, (pl, pt), (pl + w, pt + h),
            colors[int(bbox_class - 1)],
            thickness=2)
    cv2.imwrite('testGt.jpg', img)
    print(seqPath, frame_value)
    return seqPath.split('/')[-1], frame_value


def gen_image_list(dataPath, datType):
    inputPath = f'{dataPath}/labels_with_ids/{datType}'
    pathList = sorted(glob.glob(inputPath + '/*'))
    print(pathList)
    allImageList = []
    for pathSingle in pathList:
        imgList = sorted(glob.glob(osp.join(pathSingle, 'img1', '*.txt')))
        for imgPath in imgList:
            imgPath = imgPath.replace('labels_with_ids', 'images').replace(
                '.txt', '.jpg')
            allImageList.append(imgPath)
    with open(f'{dataPath}.{datType}', 'w') as image_list_file:
        allImageListStr = str.join('\n', allImageList)
        image_list_file.write(allImageListStr)


def formatOrigin(datapath, phase):
    label_with_idPath = osp.join(datapath, 'labels_with_ids', phase)
    print(label_with_idPath)
    for txtList in sorted(glob.glob(label_with_idPath + '/*.txt')):
        print(txtList)
        seqName = txtList.split('/')[-1]
        seqName = str.join('-', seqName.split('-')[0:-1]).replace('.txt', '')
        seqPath = osp.join(label_with_idPath, seqName, 'img1')
        mkdir_if_missing(seqPath)
        os.system(f'mv {txtList} {seqPath}')


def copyImg(fromRootPath, toRootPath, phase):
    fromPath = osp.join(fromRootPath, 'images', phase)
    toPathSeqPath = osp.join(toRootPath, 'labels_with_ids', phase)
    seqList = sorted(glob.glob(toPathSeqPath + '/*'))
    for seqPath in seqList:
        seqName = seqPath.split('/')[-1]
        imgTxtList = sorted(glob.glob(osp.join(seqPath, 'img1') + '/*.txt'))
        img_toPathSeqPath = osp.join(seqPath, 'img1')
        img_toPathSeqPath = img_toPathSeqPath.replace('labels_with_ids',
                                                      'images')
        mkdir_if_missing(img_toPathSeqPath)

        for imgTxt in imgTxtList:
            imgName = imgTxt.split('/')[-1].replace('.txt', '.jpg')
            imgfromPath = osp.join(fromPath, seqName, imgName)
            print(f'cp {imgfromPath} {img_toPathSeqPath}')
            os.system(f'cp {imgfromPath} {img_toPathSeqPath}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BDD100K to MOT format')
    parser.add_argument("--data_path", default='bdd100k')
    parser.add_argument("--phase", default='train')
    parser.add_argument("--classes", default='2,3,4,9,10')

    parser.add_argument("--img_dir", default="bdd100k/images/track/")
    parser.add_argument("--label_dir", default="bdd100k/labels/box_track_20/")
    parser.add_argument("--save_path", default="bdd100kmot_vehicle")
    parser.add_argument("--height", default=720)
    parser.add_argument("--width", default=1280)
    args = parser.parse_args()

    attr_dict = dict()
    attr_dict["categories"] = [{
        "supercategory": "none",
        "id": 0,
        "name": "pedestrian"
    }, {
        "supercategory": "none",
        "id": 1,
        "name": "rider"
    }, {
        "supercategory": "none",
        "id": 2,
        "name": "car"
    }, {
        "supercategory": "none",
        "id": 3,
        "name": "truck"
    }, {
        "supercategory": "none",
        "id": 4,
        "name": "bus"
    }, {
        "supercategory": "none",
        "id": 5,
        "name": "train"
    }, {
        "supercategory": "none",
        "id": 6,
        "name": "motorcycle"
    }, {
        "supercategory": "none",
        "id": 7,
        "name": "bicycle"
    }, {
        "supercategory": "none",
        "id": 8,
        "name": "other person"
    }, {
        "supercategory": "none",
        "id": 9,
        "name": "trailer"
    }, {
        "supercategory": "none",
        "id": 10,
        "name": "other vehicle"
    }]
    attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

    # create bdd100kmot_vehicle training set in MOT format
    print('Loading and converting training set...')
    train_img_dir = os.path.join(args.img_dir, 'train')
    train_label_dir = os.path.join(args.label_dir, 'train')
    save_img_dir = os.path.join(args.save_path, 'images', 'train')
    save_label_dir = os.path.join(args.save_path, 'labels_with_ids', 'train')
    if not os.path.exists(save_img_dir): os.makedirs(save_img_dir)
    if not os.path.exists(save_label_dir): os.makedirs(save_label_dir)
    bdd2mot_tracking(train_img_dir, train_label_dir, save_img_dir,
                     save_label_dir)

    # create bdd100kmot_vehicle validation set in MOT format
    print('Loading and converting validation set...')
    val_img_dir = os.path.join(args.img_dir, 'val')
    val_label_dir = os.path.join(args.label_dir, 'val')
    save_img_dir = os.path.join(args.save_path, 'images', 'val')
    save_label_dir = os.path.join(args.save_path, 'labels_with_ids', 'val')
    if not os.path.exists(save_img_dir): os.makedirs(save_img_dir)
    if not os.path.exists(save_label_dir): os.makedirs(save_label_dir)
    bdd2mot_tracking(val_img_dir, val_label_dir, save_img_dir, save_label_dir)

    # gen gt file
    dataPath = args.data_path
    phase = args.phase
    classes = args.classes.split(',')
    formatOrigin(osp.join(dataPath, 'bdd100kmot_vehicle'), phase)
    dataDir = osp.join(
        osp.join(dataPath, 'bdd100kmot_vehicle'), 'labels_with_ids',
        phase) + '/*'
    genMotGt(dataDir, classes=classes)
    copyImg(dataPath, osp.join(dataPath, 'bdd100kmot_vehicle'), phase)
    updateSeqInfo(osp.join(dataPath, 'bdd100kmot_vehicle'), phase)
    gen_image_list(osp.join(dataPath, 'bdd100kmot_vehicle'), phase)
    os.system(f'rm -r {dataPath}/bdd100kmot_vehicle/images/' + phase + '/*.jpg')
