# 将coco数据集转为voc格式

from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab,os,cv2,shutil
from lxml import etree, objectify
from tqdm import tqdm
import random
from PIL import Image
 
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# coco文件路径 
dataDir='path'
# 需要提前的标签
CK5cats=['你的标签']
 
CKdir=dataDir+'/voc'
CKimg_dir=CKdir+"/"+"images"
CKanno_dir=CKdir+"/"+"annotations"
 
def mkr(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
 
def showimg(coco,dataType,img,CK5Ids):
    global dataDir
    I = io.imread('%s/%s/%s' % (dataDir, dataType, img['file_name']))
    plt.imshow(I)
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=CK5Ids, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()

'''
把coco的json转为xml
''' 
def save_annotations(dataType,filename,objs):
    annopath=CKanno_dir+"/"+filename[:-3]+"xml"
    img_path=dataDir+"/"+dataType+"/"+filename
    dst_path=CKimg_dir+"/"+filename
    img=cv2.imread(img_path)
    im=Image.open(img_path)
    if im.mode!="RGB":
        print(filename+" not a RGB image")
        im.close()
        return
    im.close()
    shutil.copy(img_path, dst_path)
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('1'),
        E.filename(filename),
        E.source(
            E.database('CKdemo'),
            E.annotation('VOC'),
            E.image('CK')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose(),
            E.truncated("0"),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[2]),
                E.ymin(obj[3]),
                E.xmax(obj[4]),
                E.ymax(obj[5])
            )
        )
        anno_tree.append(anno_tree2)
    etree.ElementTree(anno_tree).write(annopath, pretty_print=True)
 
def showbycv(coco,dataType,img,classes,CK5Ids):
    global dataDir
    filename= img['file_name']
    filepath='%s/%s/%s' % (dataDir, dataType,filename)
    I = cv2.imread(filepath)
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=CK5Ids, iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs=[]
    for ann in anns:
        name=classes[ann['category_id']]
        if name in CK5cats:
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin=(int)(bbox[0])
                ymin=(int)(bbox[1])
                xmax=(int)(bbox[2]+bbox[0])
                ymax=(int)(bbox[3]+bbox[1])
                obj=[name,1.0,xmin,ymin,xmax,ymax]
                objs.append(obj)
                cv2.rectangle(I, (xmin,ymin),(xmax,ymax),(255,0,0))
                cv2.putText(I,name,(xmin,ymin),3,1,(0,0,255))
    save_annotations(dataType,filename,objs)
    # cv2.imshow("img",I)
    # cv2.waitKey(1)
 
def catid2name(coco):
    classes=dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']]=cat['name']
        #print(str(cat['id'])+":"+cat['name'])
    return classes
 
def get_CK5():
    mkr(CKimg_dir)
    mkr(CKanno_dir)
    dataTypes=['train','val']
    for dataType in dataTypes:
        annFile = '{}/annotations/{}.json'.format(dataDir, dataType)
        coco = COCO(annFile)
        CK5Ids = coco.getCatIds(catNms=CK5cats)
        classes=catid2name(coco)
        for srccat in CK5cats:
            print(dataType + ":" + srccat)
            catIds = coco.getCatIds(catNms=[srccat])
            imgIds = coco.getImgIds(catIds=catIds)
            #imgIds=imgIds[0:100]
            for imgId in tqdm(imgIds):
                img=coco.loadImgs(imgId)[0]
                showbycv(coco,dataType,img,classes,CK5Ids)
                #showimg(coco,dataType,img,CK5Ids)
 
#按照比例拆分训练\验证\测试的比例集合
def split_traintest(trainratio=0.7,valratio=0.2,testratio=0.1):
    dataset_dir=CKdir
    files=os.listdir(CKimg_dir)
    trains=[]
    vals=[]
    trainvals=[]
    tests=[]
    random.shuffle(files)
    for i in range(len(files)):
        filepath=CKimg_dir+"/"+files[i][:-3]+"jpg"
        if(i<trainratio*len(files)):
            trains.append(files[i])
            trainvals.append(files[i])
        elif i<(trainratio+valratio)*len(files):
            vals.append(files[i])
            trainvals.append(files[i])
        else:
            tests.append(files[i])
    #write txt files for yolo
    with open(dataset_dir+"/train.txt","w")as f:
        for line in trainvals:
            filename=line[:line.rfind(".")]
            line="./images/"+line+" ./annotations/"+filename+".xml"
            f.write(line+"\n")
    with open(dataset_dir+"/valid.txt","w") as f:
        for line in tests:
            filename=line[:line.rfind(".")]
            line="./images/"+line+" ./annotations/"+filename+".xml"
            f.write(line+"\n")
    # 写入labellist
    with open(dataset_dir+"/label_list.txt","w")as f:
        for label in CK5cats:
            f.write(label+"\n")
    print("spliting done")
 
if __name__=="__main__":
    get_CK5()
    split_traintest()