#!/bin/bash
# /ist-nas/users/pratchp/conda_envs/pgsr/bin/python run_generate_depth_maps_fibo.py \
#   --ply splats/pgsrlego.ply \
#   --n-cams 30 \
#   --output-root output_multicams_tsdf/pgsrlego_truepgsrdepth \
#   --opencv-y-down

# /ist-nas/users/pratchp/conda_envs/pgsr/bin/python run_generate_depth_maps_fibo.py \
#   --ply splats/pgsrlego.ply \
#   --n-cams 30 \
#   --output-root output_multicams_tsdf/pgsrlego_truepgsrdepth_median \
#   --median_depth \
#   --opencv-y-down

# /ist-nas/users/pratchp/conda_envs/pgsr/bin/python run_generate_depth_maps_fibo.py \
#   --ply splats/pgsrlego.ply \
#   --n-cams 30 \
#   --output-root output_multicams_tsdf/pgsrlego_truepgsrdepth_blenddepth \
#   --blend_depth \
#   --opencv-y-down

# try vanilla
# /ist-nas/users/pratchp/conda_envs/pgsr/bin/python run_generate_depth_maps_fibo.py \
#   --ply splats/vanillalego.ply \
#   --n-cams 30 \
#   --output-root output_multicams_tsdf/vanillalego_truepgsrdepth \
#   --opencv-y-down \
#   --scene vanillalego

# try aomlion, fox
# /ist-nas/users/pratchp/conda_envs/pgsr/bin/python run_generate_depth_maps_fibo.py \
#   --ply splats/aomlion1.ply \
#   --n-cams 30 \
#   --output-root output_multicams_tsdf/aomlion1_truepgsrdepth \
#   --opencv-y-down \
#   --scene aomlion1

/ist-nas/users/pratchp/conda_envs/pgsr/bin/python run_generate_depth_maps_fibo.py \
  --ply splats/fox.ply \
  --n-cams 30 \
  --output-root output_multicams_tsdf/fox_truepgsrdepth \
  --opencv-y-down \
  --scene fox