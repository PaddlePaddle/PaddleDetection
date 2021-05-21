#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import yaml
import shutil
import requests
import tqdm
import hashlib
import base64
import binascii
import tarfile
import zipfile

from .voc_utils import create_list
from ppdet.core.workspace import BASE_KEY

from .logger import setup_logger
logger = setup_logger(__name__)

__all__ = [
    'get_weights_path', 'get_dataset_path', 'get_config_path',
    'download_dataset', 'create_voc_list'
]

WEIGHTS_HOME = osp.expanduser("~/.cache/paddle/weights")
DATASET_HOME = osp.expanduser("~/.cache/paddle/dataset")
CONFIGS_HOME = osp.expanduser("~/.cache/paddle/configs")

# dict of {dataset_name: (download_info, sub_dirs)}
# download info: [(url, md5sum)]
DATASETS = {
    'coco': ([
        (
            'http://images.cocodataset.org/zips/train2017.zip',
            'cced6f7f71b7629ddf16f17bbcfab6b2', ),
        (
            'http://images.cocodataset.org/zips/val2017.zip',
            '442b8da7639aecaf257c1dceb8ba8c80', ),
        (
            'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
            'f4bbac642086de4f52a3fdda2de5fa2c', ),
    ], ["annotations", "train2017", "val2017"]),
    'voc': ([
        (
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
            '6cd6e144f989b92b3379bac3b3de84fd', ),
        (
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
            'c52e279531787c972589f7e41ab4ae64', ),
        (
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
            'b6e924de25625d8de591ea690078ad9f', ),
    ], ["VOCdevkit/VOC2012", "VOCdevkit/VOC2007"]),
    'wider_face': ([
        (
            'https://dataset.bj.bcebos.com/wider_face/WIDER_train.zip',
            '3fedf70df600953d25982bcd13d91ba2', ),
        (
            'https://dataset.bj.bcebos.com/wider_face/WIDER_val.zip',
            'dfa7d7e790efa35df3788964cf0bbaea', ),
        (
            'https://dataset.bj.bcebos.com/wider_face/wider_face_split.zip',
            'a4a898d6193db4b9ef3260a68bad0dc7', ),
    ], ["WIDER_train", "WIDER_val", "wider_face_split"]),
    'fruit': ([(
        'https://dataset.bj.bcebos.com/PaddleDetection_demo/fruit.tar',
        'baa8806617a54ccf3685fa7153388ae6', ), ],
              ['Annotations', 'JPEGImages']),
    'roadsign_voc': ([(
        'https://paddlemodels.bj.bcebos.com/object_detection/roadsign_voc.tar',
        '8d629c0f880dd8b48de9aeff44bf1f3e', ), ], ['annotations', 'images']),
    'roadsign_coco': ([(
        'https://paddlemodels.bj.bcebos.com/object_detection/roadsign_coco.tar',
        '49ce5a9b5ad0d6266163cd01de4b018e', ), ], ['annotations', 'images']),
    'mot': (),
    'objects365': ()
}

DOWNLOAD_RETRY_LIMIT = 3

PPDET_WEIGHTS_DOWNLOAD_URL_PREFIX = 'https://paddledet.bj.bcebos.com/'


def parse_url(url):
    url = url.replace("ppdet://", PPDET_WEIGHTS_DOWNLOAD_URL_PREFIX)
    return url


def get_weights_path(url):
    """Get weights path from WEIGHTS_HOME, if not exists,
    download it from url.
    """
    url = parse_url(url)
    path, _ = get_path(url, WEIGHTS_HOME)
    return path


def get_config_path(url):
    """Get weights path from CONFIGS_HOME, if not exists,
    download it from url.
    """
    url = parse_url(url)
    path, _ = get_path(url, CONFIGS_HOME)
    _download_config(path, url, CONFIGS_HOME)

    return path


def _download_config(cfg_path, cfg_url, cur_dir):
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # download dependence base ymls
    if BASE_KEY in cfg:
        base_ymls = list(cfg[BASE_KEY])
        for base_yml in base_ymls:
            if base_yml.startswith("~"):
                base_yml = os.path.expanduser(base_yml)
                relpath = osp.relpath(base_yml, cfg_path)
            if not base_yml.startswith('/'):
                relpath = base_yml
                base_yml = os.path.join(os.path.dirname(cfg_path), base_yml)

            if osp.isfile(base_yml):
                logger.debug("Found _BASE_ config: {}".format(base_yml))
                continue

            # download to CONFIGS_HOME firstly
            base_yml_url = osp.join(osp.split(cfg_url)[0], relpath)
            path, _ = get_path(base_yml_url, CONFIGS_HOME)

            # move from CONFIGS_HOME to dst_path to restore config directory structure
            dst_path = osp.join(cur_dir, relpath)
            dst_dir = osp.split(dst_path)[0]
            if not osp.isdir(dst_dir):
                os.makedirs(dst_dir)
            shutil.move(path, dst_path)

            # perfrom download base yml recursively
            _download_config(dst_path, base_yml_url, osp.split(dst_path)[0])


def get_dataset_path(path, annotation, image_dir):
    """
    If path exists, return path.
    Otherwise, get dataset path from DATASET_HOME, if not exists,
    download it.
    """
    if _dataset_exists(path, annotation, image_dir):
        return path

    logger.info("Dataset {} is not valid for reason above, try searching {} or "
                "downloading dataset...".format(
                    osp.realpath(path), DATASET_HOME))

    data_name = os.path.split(path.strip().lower())[-1]
    for name, dataset in DATASETS.items():
        if data_name == name:
            logger.debug("Parse dataset_dir {} as dataset "
                         "{}".format(path, name))
            if name == 'objects365':
                raise NotImplementedError(
                    "Dataset {} is not valid for download automatically. "
                    "Please apply and download the dataset from "
                    "https://www.objects365.org/download.html".format(name))
            data_dir = osp.join(DATASET_HOME, name)

            if name == 'mot':
                if osp.exists(path) or osp.exists(data_dir):
                    return data_dir
                else:
                    raise NotImplementedError(
                        "Dataset {} is not valid for download automatically. "
                        "Please apply and download the dataset following docs/tutorials/PrepareMOTDataSet.md".
                        format(name))

            # For voc, only check dir VOCdevkit/VOC2012, VOCdevkit/VOC2007
            if name in ['voc', 'fruit', 'roadsign_voc']:
                exists = True
                for sub_dir in dataset[1]:
                    check_dir = osp.join(data_dir, sub_dir)
                    if osp.exists(check_dir):
                        logger.info("Found {}".format(check_dir))
                    else:
                        exists = False
                if exists:
                    return data_dir

            # voc exist is checked above, voc is not exist here
            check_exist = name != 'voc' and name != 'fruit' and name != 'roadsign_voc'
            for url, md5sum in dataset[0]:
                get_path(url, data_dir, md5sum, check_exist)

            # voc should create list after download
            if name == 'voc':
                create_voc_list(data_dir)
            return data_dir

    # not match any dataset in DATASETS
    raise ValueError(
        "Dataset {} is not valid and cannot parse dataset type "
        "'{}' for automaticly downloading, which only supports "
        "'voc' , 'coco', 'wider_face', 'fruit', 'roadsign_voc' and 'mot' currently".
        format(path, osp.split(path)[-1]))


def create_voc_list(data_dir, devkit_subdir='VOCdevkit'):
    logger.debug("Create voc file list...")
    devkit_dir = osp.join(data_dir, devkit_subdir)
    years = ['2007', '2012']

    # NOTE: since using auto download VOC
    # dataset, VOC default label list should be used, 
    # do not generate label_list.txt here. For default
    # label, see ../data/source/voc.py
    create_list(devkit_dir, years, data_dir)
    logger.debug("Create voc file list finished")


def map_path(url, root_dir):
    # parse path after download to decompress under root_dir
    fname = osp.split(url)[-1]
    zip_formats = ['.zip', '.tar', '.gz']
    fpath = fname
    for zip_format in zip_formats:
        fpath = fpath.replace(zip_format, '')
    return osp.join(root_dir, fpath)


def get_path(url, root_dir, md5sum=None, check_exist=True):
    """ Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.

    url (str): download url
    root_dir (str): root dir for downloading, it should be
                    WEIGHTS_HOME or DATASET_HOME
    md5sum (str): md5 sum of download package
    """
    # parse path after download to decompress under root_dir
    fullpath = map_path(url, root_dir)

    # For same zip file, decompressed directory name different
    # from zip file name, rename by following map
    decompress_name_map = {
        "VOCtrainval_11-May-2012": "VOCdevkit/VOC2012",
        "VOCtrainval_06-Nov-2007": "VOCdevkit/VOC2007",
        "VOCtest_06-Nov-2007": "VOCdevkit/VOC2007",
        "annotations_trainval": "annotations"
    }
    for k, v in decompress_name_map.items():
        if fullpath.find(k) >= 0:
            fullpath = osp.join(osp.split(fullpath)[0], v)

    if osp.exists(fullpath) and check_exist:
        if not osp.isfile(fullpath) or \
                _check_exist_file_md5(fullpath, md5sum, url):
            logger.debug("Found {}".format(fullpath))
            return fullpath, True
        else:
            os.remove(fullpath)

    fullname = _download(url, root_dir, md5sum)

    # new weights format which postfix is 'pdparams' not
    # need to decompress
    if osp.splitext(fullname)[-1] not in ['.pdparams', '.yml']:
        _decompress(fullname)

    return fullpath, False


def download_dataset(path, dataset=None):
    if dataset not in DATASETS.keys():
        logger.error("Unknown dataset {}, it should be "
                     "{}".format(dataset, DATASETS.keys()))
        return
    dataset_info = DATASETS[dataset][0]
    for info in dataset_info:
        get_path(info[0], path, info[1], False)
    logger.debug("Download dataset {} finished.".format(dataset))


def _dataset_exists(path, annotation, image_dir):
    """
    Check if user define dataset exists
    """
    if not osp.exists(path):
        logger.warning("Config dataset_dir {} is not exits, "
                       "dataset config is not valid".format(path))
        return False

    if annotation:
        annotation_path = osp.join(path, annotation)
        if not osp.isfile(annotation_path):
            logger.warning("Config annotation {} is not a "
                           "file, dataset config is not "
                           "valid".format(annotation_path))
            return False
    if image_dir:
        image_path = osp.join(path, image_dir)
        if not osp.isdir(image_path):
            logger.warning("Config image_dir {} is not a "
                           "directory, dataset config is not "
                           "valid".format(image_path))
            return False
    return True


def _download(url, path, md5sum=None):
    """
    Download from url, save to path.

    url (str): download url
    path (str): download to given path
    """
    if not osp.exists(path):
        os.makedirs(path)

    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    retry_cnt = 0

    while not (osp.exists(fullname) and _check_exist_file_md5(fullname, md5sum,
                                                              url)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError("Download from {} failed. "
                               "Retry limit reached".format(url))

        logger.info("Downloading {} from {}".format(fname, url))

        
        # NOTE: windows path join may incur \, which is invalid in url
        if sys.platform == "win32":
            url = url.replace('\\', '/')

        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code "
                               "{}!".format(url, req.status_code))

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get('content-length')
        with open(tmp_fullname, 'wb') as f:
            if total_size:
                for chunk in tqdm.tqdm(
                        req.iter_content(chunk_size=1024),
                        total=(int(total_size) + 1023) // 1024,
                        unit='KB'):
                    f.write(chunk)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)
        return fullname


def _check_exist_file_md5(filename, md5sum, url):
    # if md5sum is None, and file to check is weights file, 
    # read md5um from url and check, else check md5sum directly
    return _md5check_from_url(filename, url) if md5sum is None \
            and filename.endswith('pdparams') \
            else _md5check(filename, md5sum)


def _md5check_from_url(filename, url):
    # For weights in bcebos URLs, MD5 value is contained
    # in request header as 'content_md5'
    req = requests.get(url, stream=True)
    content_md5 = req.headers.get('content-md5')
    req.close()
    if not content_md5 or _md5check(
            filename,
            binascii.hexlify(base64.b64decode(content_md5.strip('"'))).decode(
            )):
        return True
    else:
        return False


def _md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    logger.debug("File {} md5 checking...".format(fullname))
    md5 = hashlib.md5()
    with open(fullname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        logger.warning("File {} md5 check failed, {}(calc) != "
                       "{}(base)".format(fullname, calc_md5sum, md5sum))
        return False
    return True


def _decompress(fname):
    """
    Decompress for zip and tar file
    """
    logger.info("Decompressing {}...".format(fname))

    # For protecting decompressing interupted,
    # decompress to fpath_tmp directory firstly, if decompress
    # successed, move decompress files to fpath and delete
    # fpath_tmp and remove download compress file.
    fpath = osp.split(fname)[0]
    fpath_tmp = osp.join(fpath, 'tmp')
    if osp.isdir(fpath_tmp):
        shutil.rmtree(fpath_tmp)
        os.makedirs(fpath_tmp)

    if fname.find('tar') >= 0:
        with tarfile.open(fname) as tf:
            tf.extractall(path=fpath_tmp)
    elif fname.find('zip') >= 0:
        with zipfile.ZipFile(fname) as zf:
            zf.extractall(path=fpath_tmp)
    else:
        raise TypeError("Unsupport compress file type {}".format(fname))

    for f in os.listdir(fpath_tmp):
        src_dir = osp.join(fpath_tmp, f)
        dst_dir = osp.join(fpath, f)
        _move_and_merge_tree(src_dir, dst_dir)

    shutil.rmtree(fpath_tmp)
    os.remove(fname)


def _move_and_merge_tree(src, dst):
    """
    Move src directory to dst, if dst is already exists,
    merge src to dst
    """
    if not osp.exists(dst):
        shutil.move(src, dst)
    elif osp.isfile(src):
        shutil.move(src, dst)
    else:
        for fp in os.listdir(src):
            src_fp = osp.join(src, fp)
            dst_fp = osp.join(dst, fp)
            if osp.isdir(src_fp):
                if osp.isdir(dst_fp):
                    _move_and_merge_tree(src_fp, dst_fp)
                else:
                    shutil.move(src_fp, dst_fp)
            elif osp.isfile(src_fp) and \
                    not osp.isfile(dst_fp):
                shutil.move(src_fp, dst_fp)
