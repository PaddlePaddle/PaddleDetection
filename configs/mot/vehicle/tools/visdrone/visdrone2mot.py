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
import argparse
import numpy as np
import random

# The object category indicates the type of annotated object, 
# (i.e., ignored regions(0), pedestrian(1), people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10),others(11))

# Extract single class or multi class
isExtractMultiClass = False
# The sequence is excluded because there are too few vehicles
exclude_seq = ["uav0000086_00000_v"]


def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)


def genGtFile(seqPath, outPath, classes=[]):
    id_idx = 0
    old_idx = -1
    with open(seqPath, 'r') as singleSeqFile:
        motLine = []
        allLines = singleSeqFile.readlines()
        for line in allLines:
            line = line.replace('\n', '')
            line = line.split(',')
            # exclude occlusion!='2'
            if line[-1] != '2' and line[7] in classes:
                if old_idx != int(line[1]):
                    id_idx += 1
                    old_idx = int(line[1])
                newLine = line[0:6]
                newLine[1] = str(id_idx)
                newLine.append('1')
                if (len(classes) > 1 and isExtractMultiClass):
                    class_index = str(classes.index(line[7]) + 1)
                    newLine.append(class_index)
                else:
                    newLine.append('1')  # use permenant class '1'
                newLine.append('1')
                motLine.append(newLine)
        mkdir_if_missing(outPath)
        gtFilePath = osp.join(outPath, 'gt.txt')
        with open(gtFilePath, 'w') as gtFile:
            motLine = list(map(lambda x: str.join(',', x), motLine))
            motLineStr = str.join('\n', motLine)
            gtFile.write(motLineStr)


def genSeqInfo(img1Path, seqName):
    imgPaths = glob.glob(img1Path + '/*.jpg')
    seqLength = len(imgPaths)
    if seqLength > 0:
        image1 = cv2.imread(imgPaths[0])
        imgHeight = image1.shape[0]
        imgWidth = image1.shape[1]
    else:
        imgHeight = 0
        imgWidth = 0
    seqInfoStr = f'''[Sequence]\nname={seqName}\nimDir=img1\nframeRate=30\nseqLength={seqLength}\nimWidth={imgWidth}\nimHeight={imgHeight}\nimExt=.jpg'''
    seqInfoPath = img1Path.replace('/img1', '')
    with open(seqInfoPath + '/seqinfo.ini', 'w') as seqFile:
        seqFile.write(seqInfoStr)


def copyImg(img1Path, gtTxtPath, outputFileName):
    with open(gtTxtPath, 'r') as gtFile:
        allLines = gtFile.readlines()
        imgList = []
        for line in allLines:
            imgIdx = int(line.split(',')[0])
            if imgIdx not in imgList:
                imgList.append(imgIdx)
                seqName = gtTxtPath.replace('./{}/'.format(outputFileName),
                                            '').replace('/gt/gt.txt', '')
                sourceImgPath = osp.join('./sequences', seqName,
                                         '{:07d}.jpg'.format(imgIdx))
                os.system(f'cp {sourceImgPath} {img1Path}')


def genMotLabels(datasetPath, outputFileName, classes=['2']):
    mkdir_if_missing(osp.join(datasetPath, outputFileName))
    annotationsPath = osp.join(datasetPath, 'annotations')
    annotationsList = glob.glob(osp.join(annotationsPath, '*.txt'))
    for annotationPath in annotationsList:
        seqName = annotationPath.split('/')[-1].replace('.txt', '')
        if seqName in exclude_seq:
            continue
        mkdir_if_missing(osp.join(datasetPath, outputFileName, seqName, 'gt'))
        mkdir_if_missing(osp.join(datasetPath, outputFileName, seqName, 'img1'))
        genGtFile(annotationPath,
                  osp.join(datasetPath, outputFileName, seqName, 'gt'), classes)
        img1Path = osp.join(datasetPath, outputFileName, seqName, 'img1')
        gtTxtPath = osp.join(datasetPath, outputFileName, seqName, 'gt/gt.txt')
        copyImg(img1Path, gtTxtPath, outputFileName)
        genSeqInfo(img1Path, seqName)


def deleteFileWhichImg1IsEmpty(mot16Path, dataType='train'):
    path = mot16Path
    data_images_train = osp.join(path, 'images', f'{dataType}')
    data_images_train_seqs = glob.glob(data_images_train + '/*')
    if (len(data_images_train_seqs) == 0):
        print('dataset is empty!')
    for data_images_train_seq in data_images_train_seqs:
        data_images_train_seq_img1 = osp.join(data_images_train_seq, 'img1')
        if len(glob.glob(data_images_train_seq_img1 + '/*.jpg')) == 0:
            print(f"os.system(rm -rf {data_images_train_seq})")
            os.system(f'rm -rf {data_images_train_seq}')


def formatMot16Path(dataPath, pathType='train'):
    train_path = osp.join(dataPath, 'images', pathType)
    mkdir_if_missing(train_path)
    os.system(f'mv {dataPath}/* {train_path}')


def VisualGt(dataPath, phase='train'):
    seqList = sorted(glob.glob(osp.join(dataPath, 'images', phase) + '/*'))
    seqIndex = random.randint(0, len(seqList) - 1)
    seqPath = seqList[seqIndex]
    gt_path = osp.join(seqPath, 'gt', 'gt.txt')
    img_list_path = sorted(glob.glob(osp.join(seqPath, 'img1', '*.jpg')))
    imgIndex = random.randint(0, len(img_list_path))
    img_Path = img_list_path[imgIndex]
    frame_value = int(img_Path.split('/')[-1].replace('.jpg', ''))
    gt_value = np.loadtxt(gt_path, dtype=int, delimiter=',')
    gt_value = gt_value[gt_value[:, 0] == frame_value]
    get_list = gt_value.tolist()
    img = cv2.imread(img_Path)
    colors = [[255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 255, 0],
              [0, 255, 255], [0, 0, 255]]
    for seq, _id, pl, pt, w, h, _, bbox_class, _ in get_list:
        cv2.putText(img,
                    str(bbox_class), (pl, pt), cv2.FONT_HERSHEY_PLAIN, 2,
                    colors[bbox_class - 1])
        cv2.rectangle(
            img, (pl, pt), (pl + w, pt + h),
            colors[bbox_class - 1],
            thickness=2)
    cv2.imwrite('testGt.jpg', img)


def VisualDataset(datasetPath, phase='train', seqName='', frameId=1):
    trainPath = osp.join(datasetPath, 'labels_with_ids', phase)
    seq1Paths = osp.join(trainPath, seqName)
    seq_img1_path = osp.join(seq1Paths, 'img1')
    label_with_idPath = osp.join(seq_img1_path, '%07d' % frameId) + '.txt'
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


def gen_image_list(dataPath, datType):
    inputPath = f'{dataPath}/images/{datType}'
    pathList = glob.glob(inputPath + '/*')
    pathList = sorted(pathList)
    allImageList = []
    for pathSingle in pathList:
        imgList = sorted(glob.glob(osp.join(pathSingle, 'img1', '*.jpg')))
        for imgPath in imgList:
            allImageList.append(imgPath)
    with open(f'{dataPath}.{datType}', 'w') as image_list_file:
        allImageListStr = str.join('\n', allImageList)
        image_list_file.write(allImageListStr)


def gen_labels_mot(MOT_data, phase='train'):
    seq_root = './{}/images/{}'.format(MOT_data, phase)
    label_root = './{}/labels_with_ids/{}'.format(MOT_data, phase)
    mkdir_if_missing(label_root)
    seqs = [s for s in os.listdir(seq_root)]
    print('seqs => ', seqs)
    tid_curr = 0
    tid_last = -1
    for seq in seqs:
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find(
            '\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find(
            '\nimExt')])

        gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

        seq_label_root = osp.join(label_root, seq, 'img1')
        mkdir_if_missing(seq_label_root)

        for fid, tid, x, y, w, h, mark, label, _ in gt:
            # if mark == 0 or not label == 1: 
            #     continue
            fid = int(fid)
            tid = int(tid)
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{:07d}.txt'.format(fid))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / seq_width, y / seq_height, w / seq_width,
                h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)


def parse_arguments():
    parser = argparse.ArgumentParser(description='input method')
    parser.add_argument("--transMot", type=bool, default=False)
    parser.add_argument("--genMot", type=bool, default=False)
    parser.add_argument("--formatMotPath", type=bool, default=False)
    parser.add_argument("--deleteEmpty", type=bool, default=False)
    parser.add_argument("--genLabelsMot", type=bool, default=False)
    parser.add_argument("--genImageList", type=bool, default=False)
    parser.add_argument("--visualImg", type=bool, default=False)
    parser.add_argument("--visualGt", type=bool, default=False)
    parser.add_argument("--data_name", type=str, default='visdrone_vehicle')
    parser.add_argument("--phase", type=str, default='train')
    parser.add_argument("--classes", type=str, default='4,5,6,9')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    classes = args.classes.split(',')
    datasetPath = './'
    dataName = args.data_name
    phase = args.phase
    if args.transMot:
        genMotLabels(datasetPath, dataName, classes)
        formatMot16Path(dataName, pathType=phase)
        mot16Path = f'./{dataName}'
        deleteFileWhichImg1IsEmpty(mot16Path, dataType=phase)
        gen_labels_mot(dataName, phase=phase)
        gen_image_list(dataName, phase)
    if args.genMot:
        genMotLabels(datasetPath, dataName, classes)
    if args.formatMotPath:
        formatMot16Path(dataName, pathType=phase)
    if args.deleteEmpty:
        mot16Path = f'./{dataName}'
        deleteFileWhichImg1IsEmpty(mot16Path, dataType=phase)
    if args.genLabelsMot:
        gen_labels_mot(dataName, phase=phase)
    if args.genImageList:
        gen_image_list(dataName, phase)
    if args.visualGt:
        VisualGt(f'./{dataName}', phase)
    if args.visualImg:
        seqName = 'uav0000137_00458_v'
        frameId = 43
        VisualDataset(
            f'./{dataName}', phase=phase, seqName=seqName, frameId=frameId)
