#!/usr/bin/env bash

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#=================================================
#                   Utils
#=================================================


DIST_DIR="dist"
BUILD_DIR="build"
EGG_DIR="paddledet.egg-info"

CFG_DIR="configs"
TEST_DIR=".tests"

function python_version_check() {
  PY_MAIN_VERSION=`python -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1}'`
  PY_SUB_VERSION=`python -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $2}'`
  echo -e "find python version ${PY_MAIN_VERSION}.${PY_SUB_VERSION}"
  if [ $PY_MAIN_VERSION -ne "3" -o $PY_SUB_VERSION -lt "5" ]; then
    echo -e "please use Python >= 3.5 !"
    exit 1
  fi
}

function init() {
    echo -e "removing building directory..."
    rm -rf $DIST_DIR $BUILD_DIR $EGG_DIR $TEST_DIR
    if [ `pip list | grep paddledet | wc -l` -gt 0  ]; then
      echo -e "uninstalling paddledet..."
      pip uninstall -y paddledet
    fi
}

function build_and_install() {
  echo -e "building paddledet wheel..."
  python setup.py sdist bdist_wheel
  if [$? -ne 0]; then
    echo -e "build paddledet wheel failed!"
    exit 1
  fi

  echo -e "build paddldet wheel success, installing paddledet..."
  cd $DIST_DIR
  echo -e "find wheel `find . -name 'paddledet*.whl'`"
  find . -name "paddledet*.whl" | xargs pip install
  if [ $? -ne 0 ]; then
    cd ..
    echo -e "install paddledet wheel failed!"
    exit 1
  fi
  echo -e "paddledet compile and install success"
  cd ..
}

function unittest() {
  if [ -d $TEST_DIR ]; then
    rm -rf $TEST_DIR
  fi;

  # NOTE: perform unittests under TEST_DIR to
  #       make sure installed paddledet is used
  mkdir $TEST_DIR
  cp -r $CFG_DIR $TEST_DIR
  cd $TEST_DIR

  if [ $? != 0  ]; then
    exit 1
  fi
  find "../ppdet" -name 'tests' -type d -print0 | \
      xargs -0 -I{} -n1 bash -c \
      'python -m unittest discover -v -s {}'

  # clean TEST_DIR
  cd ..
  rm -rf $TEST_DIR
}

function cleanup() {
  if [ -d $TEST_DIR ]; then
    rm -rf $TEST_DIR
  fi

  rm -rf $BUILD_DIR $EGG_DIR
  pip uninstall -y paddledet
}

function abort() {
  echo "build wheel and unittest failed! please check your code" 1>&2

  cur_dir=`basename "$pwd"`
  if [ cur_dir==$TEST_DIR -o cur_dir==$DIST_DIR ]; then
    cd ..
  fi

  rm -rf $BUILD_DIR $EGG_DIR $DIST_DIR $TEST_DIR
  pip uninstall -y paddledet
}

python_version_check

trap 'abort' 0
set -e

init
build_and_install
unittest
cleanup

echo -e "paddledet wheel compiled and check success!"
echo -e "wheel saved under ./dist"

trap : 0
