#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python train.py --dataset CAMERA+Real\
  --result_dir results/real/