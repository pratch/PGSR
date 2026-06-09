#!/bin/bash

dataset=$1 # e.g., 'deepblending_pgsr'
scene=$2   # e.g., 'playroom'

python train.py -s /projects/datasets/$dataset/$scene -m output_$dataset/$scene --max_abs_split_points 0 --opacity_cull_threshold 0.05
python render.py -m output_$dataset/$scene --max_depth 10.0 --voxel_size 0.01