# convert VOC xml to COCO format json
import xml.etree.ElementTree as ET
import os
import json
import argparse


# create and init coco json, img set, and class set
def init_json():
    # create coco json
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []
    # voc classes
    voc_class = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    # init json categories
    image_set = set()
    class_set = dict()
    for cat_id, cat_name in enumerate(voc_class):
        cat_item = dict()
        cat_item['supercategory'] = 'none'
        cat_item['id'] = cat_id
        cat_item['name'] = cat_name
        coco['categories'].append(cat_item)
        class_set[cat_name] = cat_id
    return coco, class_set, image_set


def getImgItem(file_name, size, img_id):
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_item = dict()
    image_item['id'] = img_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    return image_item


def getAnnoItem(object_name, image_id, ann_id, category_id, bbox):
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_item['id'] = ann_id
    return annotation_item


def convert_voc_to_coco(txt_path, json_path, xml_path):

    # create and init coco json, img set, and class set
    coco_json, class_set, image_set = init_json()

    ### collect img and ann info into coco json
    # read img_name in txt, e.g., 000005 for voc2007, 2008_000002 for voc2012
    img_txt = open(txt_path, 'r')
    img_line = img_txt.readline().strip()

    # loop xml 
    img_id = 0
    ann_id = 0
    while img_line:
        print('img_id:', img_id)

        # find corresponding xml
        xml_name = img_line.split('Annotations/', 1)[1]
        xml_file = os.path.join(xml_path, xml_name)
        if not os.path.exists(xml_file):
            print('{} is not exists.'.format(xml_name))
            img_line = img_txt.readline().strip()
            continue

        # decode xml
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception(
                'xml {} root element should be annotation, rather than {}'.
                format(xml_name, root.tag))

        # init img and ann info
        bndbox = dict()
        size = dict()
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        # filename
        fileNameNode = root.find('filename')
        file_name = fileNameNode.text

        # img size
        sizeNode = root.find('size')
        if not sizeNode:
            raise Exception('xml {} structure broken at size tag.'.format(
                xml_name))
        for subNode in sizeNode:
            size[subNode.tag] = int(subNode.text)

        # add img into json
        if file_name not in image_set:
            img_id += 1
            format_img_id = int("%04d" % img_id)
            # print('line 120. format_img_id:', format_img_id)
            image_item = getImgItem(file_name, size, img_id)
            image_set.add(file_name)
            coco_json['images'].append(image_item)
        else:
            raise Exception(' xml {} duplicated image: {}'.format(xml_name,
                                                                  file_name))

        ### add objAnn into json
        objectAnns = root.findall('object')
        for objectAnn in objectAnns:
            bndbox['xmin'] = None
            bndbox['xmax'] = None
            bndbox['ymin'] = None
            bndbox['ymax'] = None

            #add obj category
            object_name = objectAnn.find('name').text
            if object_name not in class_set:
                raise Exception('xml {} Unrecognized category: {}'.format(
                    xml_name, object_name))
            else:
                current_category_id = class_set[object_name]

            #add obj bbox ann
            objectBboxNode = objectAnn.find('bndbox')
            for coordinate in objectBboxNode:
                if bndbox[coordinate.tag] is not None:
                    raise Exception('xml {} structure corrupted at bndbox tag.'.
                                    format(xml_name))
                bndbox[coordinate.tag] = int(float(coordinate.text))
            bbox = []
            # x
            bbox.append(bndbox['xmin'])
            # y
            bbox.append(bndbox['ymin'])
            # w
            bbox.append(bndbox['xmax'] - bndbox['xmin'])
            # h
            bbox.append(bndbox['ymax'] - bndbox['ymin'])
            ann_id += 1
            ann_item = getAnnoItem(object_name, img_id, ann_id,
                                   current_category_id, bbox)
            coco_json['annotations'].append(ann_item)

        img_line = img_txt.readline().strip()

    print('Saving json.')
    json.dump(coco_json, open(json_path, 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type', type=str, default='VOC2007_test', help="data type")
    parser.add_argument(
        '--base_path',
        type=str,
        default='dataset/voc/VOCdevkit',
        help="base VOC path.")
    args = parser.parse_args()

    # image info path
    txt_name = args.type + '.txt'
    json_name = args.type + '.json'
    txt_path = os.path.join(args.base_path, 'PseudoAnnotations', txt_name)
    json_path = os.path.join(args.base_path, 'PseudoAnnotations', json_name)

    # xml path
    xml_path = os.path.join(args.base_path,
                            args.type.split('_')[0], 'Annotations')

    print('txt_path:', txt_path)
    print('json_path:', json_path)
    print('xml_path:', xml_path)

    print('Converting {} to COCO json.'.format(args.type))
    convert_voc_to_coco(txt_path, json_path, xml_path)
    print('Finished.')
