#!/bin/sh

CUT=$1
IDX=$2
RATIO=$3

cd ckd_1year
python main.py $CUT $IDX $RATIO
cd ..

make cut_point=$CUT
make dpm
