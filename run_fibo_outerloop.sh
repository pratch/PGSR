#!/bin/bash
/ist-nas/users/pratchp/conda_envs/pgsr/bin/python run_fibo_outerloop.py \
  --ply splats/pgsrlego.ply \
  --n-cams 30 \
  --output-root output_multicams_tsdf/pgsrlego \
  --opencv-y-down
