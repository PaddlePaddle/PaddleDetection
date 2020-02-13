#!/bin/bash

abort(){
    echo "Run unittest failed" 1>&2
    echo "Please check your code" 1>&2
    echo "  1. you can run unit tests by 'bash .travis/unittest.sh' locally" 1>&2
    echo "  2. you can add python requirements in .travis/requirements.txt if you use new requirements in unit tests" 1>&2
    exit 1
}

unittest(){
    if [ $? != 0 ]; then
        exit 1
    fi
    find "./ppdet" -name 'tests' -type d -print0 | \
        xargs -0 -I{} -n1 bash -c \
        'python -m unittest discover -v -s {}'
}

trap 'abort' 0
set -e

# install travis python dependencies exclude pycocotools
if [ -f ".travis/requirements.txt" ]; then
    pip install -r .travis/requirements.txt
fi

# install pycocotools
if [ `pip list | grep pycocotools | wc -l` -eq 0 ]; then
  # install git if needed
  if [ -n  `which git` ]; then
    apt-get update
    apt-get install -y git
  fi;
  git clone https://github.com/cocodataset/cocoapi.git
  cd cocoapi/PythonAPI
  make install
  python setup.py install --user
  cd ../..
  rm -rf cocoapi
fi

export PYTHONPATH=`pwd`:$PYTHONPATH

unittest .

trap : 0
