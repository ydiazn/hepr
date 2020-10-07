#!/bin/bash

config=$1
data=$2
output=$3

python optimization/run.py --config=$config --output=$output --data=$data &&
python optimization/performance/run.py --config=$config --data=$data --output=$output
