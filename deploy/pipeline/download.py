# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os, sys
import os.path as osp
import hashlib
import requests
import shutil
import tqdm
import time
import tarfile
import zipfile
from paddle.utils.download import _get_unique_endpoints

PPDET_WEIGHTS_DOWNLOAD_URL_PREFIX = 'https://paddledet.bj.bcebos.com/'

DOWNLOAD_RETRY_LIMIT = 3

WEIGHTS_HOME = osp.expanduser("~/.cache/paddle/infer_weights")

MODEL_URL_MD5_DICT = {
    'https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_det_infer.tar.gz':
    '1b8eae0f098635699bd4e8bccf3067a7',
    'https://bj.bcebos.com/v1/paddledet/models/pipeline/ch_PP-OCRv3_rec_infer.tar.gz':
    '64fa0e0701efd93c7db52a9b685b3de6',
    "https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_ppvehicle.zip":
    "3859d1a26e0c498285c2374b1a347013",
    "https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_s_36e_ppvehicle.zip":
    "4ed58b546be2a76d8ccbb138f64874ac",
    "https://bj.bcebos.com/v1/paddledet/models/pipeline/dark_hrnet_w32_256x192.zip":
    "a20d5f6ca087bff0e9f2b18df45a36f2",
    "https://bj.bcebos.com/v1/paddledet/models/pipeline/PPLCNet_x1_0_person_attribute_945_infer.zip":
    "1dfb161bf12bbc1365b2ed6866674483",
    "https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.zip":
    "5d4609142501258608bf0a1445eedaba",
    "https://bj.bcebos.com/v1/paddledet/models/pipeline/STGCN.zip":
    "cf1c3c4bae90b975accb954d13129ea4",
    "https://bj.bcebos.com/v1/paddledet/models/pipeline/ppyoloe_crn_s_80e_smoking_visdrone.zip":
    "4cd12ae55be8f0eb2b90c08ac3b48218",
    "https://bj.bcebos.com/v1/paddledet/models/pipeline/PPHGNet_tiny_calling_halfbody.zip":
    "cf86b87ace97540dace6ef08e62b584a",
    "https://bj.bcebos.com/v1/paddledet/models/pipeline/reid_model.zip":
    "fdc4dac38393b8e2b5921c1e1fdd5315"
}


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://') \
            or path.startswith('https://') \
            or path.startswith('ppdet://')


def parse_url(url):
    url = url.replace("ppdet://", PPDET_WEIGHTS_DOWNLOAD_URL_PREFIX)
    return url


def map_path(url, root_dir, path_depth=1):
    # parse path after download to decompress under root_dir
    assert path_depth > 0, "path_depth should be a positive integer"
    dirname = url
    for _ in range(path_depth):
        dirname = osp.dirname(dirname)
    fpath = osp.relpath(url, dirname)

    zip_formats = ['.zip', '.tar', '.gz']
    for zip_format in zip_formats:
        fpath = fpath.replace(zip_format, '')
    return osp.join(root_dir, fpath)


def _md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    md5 = hashlib.md5()
    with open(fullname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        return False
    return True


def _check_exist_file_md5(filename, md5sum, url):
    return _md5check(filename, md5sum)


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


def _download_dist(url, path, md5sum=None):
    env = os.environ
    if 'PADDLE_TRAINERS_NUM' in env and 'PADDLE_TRAINER_ID' in env:
        trainer_id = int(env['PADDLE_TRAINER_ID'])
        num_trainers = int(env['PADDLE_TRAINERS_NUM'])
        if num_trainers <= 1:
            return _download(url, path, md5sum)
        else:
            fname = osp.split(url)[-1]
            fullname = osp.join(path, fname)
            lock_path = fullname + '.download.lock'

            if not osp.isdir(path):
                os.makedirs(path)

            if not osp.exists(fullname):
                from paddle.distributed import ParallelEnv
                unique_endpoints = _get_unique_endpoints(ParallelEnv()
                                                         .trainer_endpoints[:])
                with open(lock_path, 'w'):  # touch    
                    os.utime(lock_path, None)
                if ParallelEnv().current_endpoint in unique_endpoints:
                    _download(url, path, md5sum)
                    os.remove(lock_path)
                else:
                    while os.path.exists(lock_path):
                        time.sleep(0.5)
            return fullname
    else:
        return _download(url, path, md5sum)


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


def _decompress(fname):
    """
    Decompress for zip and tar file
    """

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
    elif fname.find('.txt') >= 0:
        return
    else:
        raise TypeError("Unsupport compress file type {}".format(fname))

    for f in os.listdir(fpath_tmp):
        src_dir = osp.join(fpath_tmp, f)
        dst_dir = osp.join(fpath, f)
        _move_and_merge_tree(src_dir, dst_dir)

    shutil.rmtree(fpath_tmp)
    os.remove(fname)


def _decompress_dist(fname):
    env = os.environ
    if 'PADDLE_TRAINERS_NUM' in env and 'PADDLE_TRAINER_ID' in env:
        trainer_id = int(env['PADDLE_TRAINER_ID'])
        num_trainers = int(env['PADDLE_TRAINERS_NUM'])
        if num_trainers <= 1:
            _decompress(fname)
        else:
            lock_path = fname + '.decompress.lock'
            from paddle.distributed import ParallelEnv
            unique_endpoints = _get_unique_endpoints(ParallelEnv()
                                                     .trainer_endpoints[:])
            # NOTE(dkp): _decompress_dist always performed after
            # _download_dist, in _download_dist sub-trainers is waiting
            # for download lock file release with sleeping, if decompress
            # prograss is very fast and finished with in the sleeping gap
            # time, e.g in tiny dataset such as coco_ce, spine_coco, main
            # trainer may finish decompress and release lock file, so we
            # only craete lock file in main trainer and all sub-trainer
            # wait 1s for main trainer to create lock file, for 1s is
            # twice as sleeping gap, this waiting time can keep all
            # trainer pipeline in order
            # **change this if you have more elegent methods**
            if ParallelEnv().current_endpoint in unique_endpoints:
                with open(lock_path, 'w'):  # touch    
                    os.utime(lock_path, None)
                _decompress(fname)
                os.remove(lock_path)
            else:
                time.sleep(1)
                while os.path.exists(lock_path):
                    time.sleep(0.5)
    else:
        _decompress(fname)


def get_path(url, root_dir=WEIGHTS_HOME, md5sum=None, check_exist=True):
    """ Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.
    url (str): download url
    root_dir (str): root dir for downloading
    md5sum (str): md5 sum of download package
    """
    # parse path after download to decompress under root_dir
    fullpath = map_path(url, root_dir)

    # For same zip file, decompressed directory name different
    # from zip file name, rename by following map
    decompress_name_map = {"ppTSM_fight": "ppTSM", }
    for k, v in decompress_name_map.items():
        if fullpath.find(k) >= 0:
            fullpath = osp.join(osp.split(fullpath)[0], v)

    if osp.exists(fullpath) and check_exist:
        if not osp.isfile(fullpath) or \
                _check_exist_file_md5(fullpath, md5sum, url):
            return fullpath, True
        else:
            os.remove(fullpath)

    fullname = _download_dist(url, root_dir, md5sum)

    # new weights format which postfix is 'pdparams' not
    # need to decompress
    if osp.splitext(fullname)[-1] not in ['.pdparams', '.yml']:
        _decompress_dist(fullname)

    return fullpath, False


def get_weights_path(url):
    """Get weights path from WEIGHTS_HOME, if not exists,
    download it from url.
    """
    url = parse_url(url)
    md5sum = None
    if url in MODEL_URL_MD5_DICT.keys():
        md5sum = MODEL_URL_MD5_DICT[url]
    path, _ = get_path(url, WEIGHTS_HOME, md5sum)
    return path


def auto_download_model(model_path):
    # auto download
    if is_url(model_path):
        weight = get_weights_path(model_path)
        return weight
    return None


if __name__ == "__main__":
    model_path = "https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip"
    auto_download_model(model_path)
