import glob
import os
import os.path as osp
import cv2
import random
import numpy as np
import argparse
# attention!!!
# 转换的流程是，先通过bbd2mot.py生成mot的数据，然后从mot的基础数据集上面
# 生成全量的labels_with_ids，然后生成gt(是筛选之后的gt),gt再生成筛选之后
# 的labels_with_ids

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)

def transBBOx(bbox):
    # bbox --> cx cy w h
    bbox = list(map(lambda x : float(x), bbox))

    bbox[0] = (bbox[0] - bbox[2]/2) * 1280
    bbox[1] = (bbox[1] - bbox[3]/2) * 720
    bbox[2] = bbox[2] * 1280
    bbox[3] = bbox[3] * 720

    bbox = list(map(lambda x: str(x), bbox))
    return bbox

def genSingleImageMot(inputPath, classes = []):
    labelPaths = glob.glob(inputPath+'/*.txt')
    labelPaths = sorted(labelPaths)
    allLines = []
    result = {}
    for labelPath in labelPaths:
        frame = str(int(labelPath.split('-')[-1].replace('.txt', '')))
        # print('loading => ', int(frame), labelPath)
        with open(labelPath, 'r') as labelPathFile:
            lines = labelPathFile.readlines()
            for line in lines:
                line = line.replace('\n','')
                lineArray = line.split(' ')
                if len(classes) > 0:
                    if lineArray[0] in classes:
                        #add frame
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
            mot_line.append(id_line[-1]) # frame
            # mot_line.append(rid) # id
            mot_line.append(str(id_idx))
            #bbox scale
            id_line_temp = transBBOx(id_line[2:6])
            mot_line.extend(id_line_temp) # bbox
            mot_line.append('1') # come into
            # mot_line.append(id_line[0]) # class 
            mot_line.append('1') # class => 1
            mot_line.append('1')  # visual
            mot_gt.append(mot_line) 
                
    result = list(map(lambda line:str.join(',', line),mot_gt))
    resultStr = str.join('\n', result)
    return resultStr

def writeGt(inputPath, outPath, classes=[]):
    singleImageResult = genSingleImageMot(inputPath, classes = classes)
    outPathFile = outPath+'/gt.txt'
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

def genMotGtForDemo(dataDir, classes = []):
    seqLists = sorted(glob.glob(dataDir))
    for seqList in seqLists:
        # print('processing...', seqList)
        inputPath = osp.join(seqList, 'img1')
        outputPath = seqList.replace('labels_with_ids', 'images')
        outputPath = osp.join(outputPath, 'gt')
        mkdir_if_missing(outputPath)
        print('processing...', outputPath)
        writeGt(inputPath, outputPath, classes = classes)
        seqList = seqList.replace('labels_with_ids', 'images')
        seqInfoPath = osp.join(seqList,'seqinfo.ini')
        genSeqInfo(seqInfoPath)


def updateSeqInfo(dataDir, phase):
    seqPath = osp.join(dataDir,'labels_with_ids', phase)
    seqList = glob.glob(seqPath+'/*')
    for seqName in seqList:
        print('seqName=>', seqName)
        seqName_img1_dir = osp.join(seqName,'img1')
        txtLength = glob.glob(seqName_img1_dir+'/*.txt')
        name = seqName.split('/')[-1].replace('.jpg','').replace('.txt','')
        seqLength = len(txtLength)
        seqInfoStr = f'''[Sequence]\nname={name}\nimDir=img1\nframeRate=30\nseqLength={seqLength}\nimWidth=1280\nimHeight=720\nimExt=.jpg'''
        seqInfoPath = seqName_img1_dir.replace('labels_with_ids','images')
        seqInfoPath = seqInfoPath.replace('/img1','')
        seqInfoPath = seqInfoPath + '/seqinfo.ini'
        with open(seqInfoPath, 'w') as seqFile:
            seqFile.write(seqInfoStr)


def VisualDataset(datasetPath, phase='train', seqName='', frameId=1):
    trainPath = osp.join(datasetPath, 'labels_with_ids', phase)
    seq1Paths = osp.join(trainPath, seqName)
    seq_img1_path = osp.join(seq1Paths,'img1')
    label_with_idPath = osp.join(seq_img1_path, seqName+'-'+'%07d' % frameId)+'.txt'
    image_path = label_with_idPath.replace('labels_with_ids', 'images').replace('.txt','.jpg')

    seqInfoPath = str.join('/', image_path.split('/')[:-2])
    seqInfoPath = seqInfoPath + '/seqinfo.ini'
    seq_info = open(seqInfoPath).read()
    width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    with open(label_with_idPath, 'r') as label:
        allLines = label.readlines()
        images = cv2.imread(image_path)
        print('image_path => ', image_path)
        for line in allLines:
            line = line.split(' ')
            line = list(map(lambda x: float(x), line))
            c1,c2,w,h = line[2:6]
            x1 = c1-w/2
            x2 = c2-h/2
            x3 = c1+w/2
            x4 = c2+h/2
            cv2.rectangle(images, (int(x1*width),int(x2*height)),(int(x3*width),int(x4*height)),(255,0,0),thickness=2)
        cv2.imwrite('test.jpg', images)

def VisualGt(dataPath, phase='train'):
    seqList = sorted(glob.glob(osp.join(dataPath,'images',phase)+'/*'))
    seqIndex = random.randint(0,len(seqList)-1)
    seqPath = seqList[seqIndex]
    gt_path = osp.join(seqPath,'gt','gt.txt')
    img_list_path = sorted(glob.glob(osp.join(seqPath,'img1','*.jpg')))
    imgIndex = random.randint(0,len(img_list_path))
    img_Path = img_list_path[imgIndex]
    #
    frame_value = img_Path.split('/')[-1].replace('.jpg','')
    frame_value = frame_value.split('-')[-1]
    frame_value = int(frame_value)
    seqNameStr = img_Path.split('/')[-1].replace('.jpg','').replace('img', '')
    frame_value = int(seqNameStr.split('-')[-1])
    print('frame_value => ', frame_value)
    gt_value = np.loadtxt(gt_path,dtype=float, delimiter=',')
    gt_value = gt_value[gt_value[:,0]==frame_value]

    get_list = gt_value.tolist()
    img = cv2.imread(img_Path)

    colors = [[255,0,0],[255,255,0],[255,0,255],[0,255,0],[0,255,255],[0,0,255]]
    for seq,_id,pl,pt,w,h,_,bbox_class,_ in get_list:
        pl,pt,w,h = int(pl), int(pt), int(w), int(h)
        print('pl,pt,w,h => ', pl,pt,w,h)
        cv2.putText(img,str(bbox_class),(pl,pt),cv2.FONT_HERSHEY_PLAIN,2,colors[int(bbox_class-1)])
        cv2.rectangle(img, (pl,pt),(pl+w,pt+h),colors[int(bbox_class-1)],thickness=2)
    cv2.imwrite('testGt.jpg',img)
    print(seqPath, frame_value)
    return seqPath.split('/')[-1], frame_value

def gen_image_list(dataPath,datType):
    inputPath = f'{dataPath}/labels_with_ids/{datType}'
    pathList = sorted(glob.glob(inputPath+'/*'))
    print(pathList)
    allImageList = []
    for pathSingle in pathList:
        imgList = sorted(glob.glob(osp.join(pathSingle, 'img1', '*.txt')))
        for imgPath in imgList:
            imgPath = imgPath.replace('labels_with_ids','images').replace('.txt','.jpg')
            allImageList.append(imgPath)
    with open(f'{dataPath}.{datType}', 'w') as image_list_file:
        allImageListStr = str.join('\n', allImageList)
        image_list_file.write(allImageListStr)

def formatOrigin(datapath, phase):
    label_with_idPath = osp.join(datapath, 'labels_with_ids', phase)
    print(label_with_idPath)
    for txtList in sorted(glob.glob(label_with_idPath+'/*.txt')):
        print(txtList)
        seqName = txtList.split('/')[-1]
        seqName = str.join('-', seqName.split('-')[0:-1]).replace('.txt','')
        seqPath = osp.join(label_with_idPath, seqName, 'img1')
        mkdir_if_missing(seqPath)
        # print(txtList,'--> ',seqPath)
        os.system(f'mv {txtList} {seqPath}')


def copyImg(fromRootPath, toRootPath, phase):
    fromPath = osp.join(fromRootPath, 'images', phase)
    toPathSeqPath = osp.join(toRootPath,'labels_with_ids',phase)
    seqList = sorted(glob.glob(toPathSeqPath+'/*'))
    for seqPath in seqList:
        seqName = seqPath.split('/')[-1]
        imgTxtList = sorted(glob.glob(osp.join(seqPath, 'img1')+'/*.txt'))
        img_toPathSeqPath = osp.join(seqPath, 'img1')
        img_toPathSeqPath = img_toPathSeqPath.replace('labels_with_ids', 'images')
        mkdir_if_missing(img_toPathSeqPath)

        for imgTxt in imgTxtList:
            imgName = imgTxt.split('/')[-1].replace('.txt','.jpg')
            imgfromPath = osp.join(fromPath,seqName,imgName)
            print(f'cp {imgfromPath} {img_toPathSeqPath}')
            os.system(f'cp {imgfromPath} {img_toPathSeqPath}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BDD100K to MOT format')
    parser.add_argument("--data_path", default='/paddle/dataset/bdd100kmot/bdd100k_small')
    parser.add_argument("--phase", default='train')
    parser.add_argument("--classes", default='2,3,4,9,10')
    args = parser.parse_args()

    dataPath = args.data_path
    phase = args.phase
    classes = args.classes.split(',')
    # print(dataPath, phase, classes)
    formatOrigin(osp.join(dataPath, 'bdd100k_vehicle'), phase) # fromat格式
    # classes = [ '2','3','4','9','10']
    dataDir = osp.join(osp.join(dataPath, 'bdd100k_vehicle'), 'labels_with_ids', phase)+'/*'
    genMotGtForDemo(dataDir, classes=classes)
    copyImg(dataPath, osp.join(dataPath, 'bdd100k_vehicle'), phase)
    updateSeqInfo(osp.join(dataPath, 'bdd100k_vehicle'), phase)
    gen_image_list(osp.join(dataPath, 'bdd100k_vehicle'), phase)
    # delete useless file
    os.system(f'rm -r {dataPath}/bdd100k_vehicle/images/'+phase+'/*.jpg')

















