#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python evaluate.py --data real_test\
  --model model/model.pth\
  --num_structure_points 256\
  --result_dir results/real
