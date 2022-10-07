#!/usr/bin/env bash


help() {
  echo "Usage: $0 [option...] download|conduct|fulll"
  echo "download   download coco dataset"
  echo "conduct    conduct data split for semi supervised training and evaluation"
  echo "option:"
  echo " -r, --root [PATH]    select the root path of dataset. The default dataset root is ssod/data"
}
download() {
  mkdir -p coco
  cd coco
  for split in train2017 val2017 unlabeled2017;
    do
      wget http://images.cocodataset.org/zips/${split}.zip;
      unzip ${split}.zip
    done
  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip annotations_trainval2017.zip
  wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip
  unzip image_info_unlabeled2017.zip
  cd ..
}
conduct() {
  OFFSET=$RANDOM
  for percent in 1 5 10; do
      for fold in 1 2 3 4 5; do
          python tools/dataset/semi_coco.py --percent ${percent} --seed ${fold} --data-dir "${data_root}"/coco --seed-offset ${OFFSET}
      done
  done
}

data_root=data
ROOT=$(dirname "$0")/../..

cd "${ROOT}"

case $1 in
  -r | --root)
    data_root=$2
    shift 2
    ;;
esac
mkdir -p ${data_root}
case $1 in
  download)
    cd ${data_root}
    download
    ;;
  conduct)
    conduct
    ;;
  full)
    cd ${data_root}
    download
    cd ..
    conduct
    ;;
  *)
    help
    exit 0
    ;;
esac
