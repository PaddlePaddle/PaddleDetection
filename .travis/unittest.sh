#!/bin/bash

abort(){
    echo "Run unittest failed" 1>&2
    echo "Please check your code" 1>&2
    exit 1
}

unittest(){
    if [ $? != 0 ]; then
        exit 1
    fi
    find "./ppdet/modeling" -name 'tests' -type d -print0 | \
        xargs -0 -I{} -n1 bash -c \
        'python -m unittest discover -v -s {}'
}

trap 'abort' 0
set -e

# install travis python dependencies
if [ -f ".travis/requirements.txt" ]; then
    pip install -r .travis/requirements.txt
fi
export PYTHONPATH=`pwd`:$PYTHONPATH

unittest .

trap : 0
