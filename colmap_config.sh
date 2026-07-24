#!/usr/bin/env bash
# COLMAP configuration variables
# Edit these to change options passed to COLMAP commands.
# This file is sourced by colmap_reconstruction.sh

# Feature extractor extra args (space-separated)
FEATURE_EXTRACTOR_ARGS=(
  --ImageReader.camera_model SIMPLE_RADIAL
  --ImageReader.single_camera 1
  --FeatureExtraction.use_gpu 1
  --SiftExtraction.max_image_size 3200
  --SiftExtraction.max_num_features 8192
)

# Sequential matcher args
SEQUENTIAL_MATCHER_ARGS=(
  --FeatureMatching.use_gpu 1
  --FeatureMatching.guided_matching 1
  --SequentialMatching.overlap 10
  --SequentialMatching.loop_detection 1
)

# Mapper args
MAPPER_ARGS=(
  --Mapper.min_num_matches 15
  --Mapper.multiple_models 1
  --Mapper.ba_use_gpu 0
)

# Image undistorter extra args
IMAGE_UNDISTORTER_ARGS=(
  --output_type COLMAP
  --copy_policy copy
  --max_image_size 2000
)

# PatchMatch stereo args
PATCH_MATCH_ARGS=(
  --PatchMatchStereo.max_image_size 2000
  --PatchMatchStereo.geom_consistency 1
)

# Stereo fusion args
STEREO_FUSION_ARGS=(
  --input_type geometric
  --output_type PLY
)

# Poisson mesher args (empty by default)
POISSON_MESHER_ARGS=( )

# End of config
