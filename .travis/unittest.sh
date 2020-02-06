#!/bin/bash

abort(){
    echo "Run unittest failed" 1>&2
    echo "Please check your code" 1>&2
    exit 1
}

log(){
  echo "Run unittest failed" 1>&2
  echo "\t1. you can run 'bash .travis/unittest.sh locally.'" 1<&2
  echo "\t2. you can add python requirement in .travis/requirements.txt for ImportError" 1<&2
}

unittest(){
    if [ $? != 0 ]; then
        exit 1
    fi
    find "./ppdet/modeling" -name 'tests' -type d -print0 | \
        xargs -0 -I{} -n1 bash -c \
        'python -m unittest discover -v -s {}'
}

# install travis python dependencies
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

trap 'abort' 0
set -e

# # install travis python dependencies
# if [ -f ".travis/requirements.txt" ]; then
#     pip install -r .travis/requirements.txt
# fi
export PYTHONPATH=`pwd`:$PYTHONPATH

unittest .

trap : 0
