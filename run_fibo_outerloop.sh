#!/bin/bash
# /ist-nas/users/pratchp/conda_envs/pgsr/bin/python run_fibo_outerloop.py \
#   --ply splats/pgsrlego.ply \
#   --n-cams 30 \
#   --output-root output_multicams_tsdf/pgsrlego_truepgsrdepth \
#   --opencv-y-down

# /ist-nas/users/pratchp/conda_envs/pgsr/bin/python run_fibo_outerloop.py \
#   --ply splats/pgsrlego.ply \
#   --n-cams 30 \
#   --output-root output_multicams_tsdf/pgsrlego_truepgsrdepth_median \
#   --median_depth \
#   --opencv-y-down

/ist-nas/users/pratchp/conda_envs/pgsr/bin/python run_fibo_outerloop.py \
  --ply splats/pgsrlego.ply \
  --n-cams 30 \
  --output-root output_multicams_tsdf/pgsrlego_truepgsrdepth_blenddepth \
  --blend_depth \
  --opencv-y-down