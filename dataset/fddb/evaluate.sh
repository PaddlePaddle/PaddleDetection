#!/usr/bin/env bash

RESULT_FILE=$1
OUTPUT_DIR=$2

./evaluation/evaluate -a ./FDDB-folds/fddb_annotFile.txt \
	   -d $RESULT_FILE -f 0 \
	   -i ./ -l ./FDDB-folds/filePath.txt \
           -z .jpg \
	   -r $OUTPUT_DIR
