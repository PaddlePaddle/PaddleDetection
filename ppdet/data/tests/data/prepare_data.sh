#!/bin/bash

#function:
#   prepare coco data for testing

root=$(dirname `readlink -f ${BASH_SOURCE}[0]`)
cwd=`pwd`

if [[ $cwd != $root ]];then
    pushd $root 2>&1 1>/dev/null
fi

test_coco_python2_url="http://filecenter.matrix.baidu.com/api/v1/file/wanglong03/coco.test.python2.zip/20190603095315/download"
test_coco_python3_url="http://filecenter.matrix.baidu.com/api/v1/file/wanglong03/coco.test.python3.zip/20190603095447/download"

if [[ $1 = "python2" ]];then
    test_coco_data_url=${test_coco_python2_url}
    coco_zip_file="coco.test.python2.zip"
else
    test_coco_data_url=${test_coco_python3_url}
    coco_zip_file="coco.test.python3.zip"
fi
echo "download testing coco from url[${test_coco_data_url}]"
coco_root_dir=${coco_zip_file/.zip/}

# clear already exist file or directory
rm -rf ${coco_root_dir} ${coco_zip_file}

wget ${test_coco_data_url} -O ${coco_zip_file}
if [ -e $coco_zip_file ];then
    echo "succeed to download ${coco_zip_file}, so unzip it"
    unzip ${coco_zip_file} >/dev/null 2>&1
fi

if [ -e ${coco_root_dir} ];then
    rm -rf coco.test
    ln -s ${coco_root_dir} coco.test
    echo "succeed to generate coco data in[${coco_root_dir}] for testing"
    exit 0
else
    echo "failed to generate coco data"
    exit 1
fi
