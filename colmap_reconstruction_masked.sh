#!/bin/bash

# COLMAP Reconstruction Pipeline
# This script runs the complete COLMAP reconstruction workflow in sequence
#
# Usage: ./colmap_reconstruction_masked.sh <project_dir>
# Example: ./colmap_reconstruction_masked.sh south-building
#          ./colmap_reconstruction_masked.sh ferrari_lego

set -e  # Exit on error

# Error handler
trap 'echo ""; echo "❌ Error: Command failed on line $LINENO"; echo "Failed command: $BASH_COMMAND"; exit 1' ERR

# Check if project_dir argument is provided
if [ -z "$1" ]; then
    echo "Error: Project directory name required"
    echo "Usage: $0 <project_dir>"
    echo "Example: $0 south-building"
    exit 1
fi

PROJECT_NAME="$1"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$BASE_DIR/$PROJECT_NAME"
PROJECT_RECON_DIR="$PROJECT_DIR/recon_masked"

# Validate project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory not found: $PROJECT_DIR"
    exit 1
fi

# Start from a clean masked reconstruction workspace so databases, sparse
# models, and dense products from earlier runs cannot be mixed with this run.
if [ -d "$PROJECT_RECON_DIR" ]; then
  echo "Clearing previous masked reconstruction: $PROJECT_RECON_DIR"
  rm -rf -- "$PROJECT_RECON_DIR"
fi

# Create necessary directories
mkdir -p "$PROJECT_RECON_DIR/sparse"
mkdir -p "$PROJECT_RECON_DIR/dense"

# Source config file (optional)
CONFIG_FILE="$BASE_DIR/colmap_config.sh"
if [ -f "$CONFIG_FILE" ]; then
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
  echo "Loaded config: $CONFIG_FILE"
else
  echo "Warning: Config file not found: $CONFIG_FILE. Using builtin defaults."
  FEATURE_EXTRACTOR_ARGS=(--ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1 --FeatureExtraction.use_gpu 1 --SiftExtraction.max_image_size 3200 --SiftExtraction.max_num_features 8192)
  SEQUENTIAL_MATCHER_ARGS=(--FeatureMatching.use_gpu 1 --FeatureMatching.guided_matching 1 --SequentialMatching.overlap 10 --SequentialMatching.loop_detection 1)
  MAPPER_ARGS=(--Mapper.min_num_matches 15 --Mapper.multiple_models 1 --Mapper.ba_use_gpu 0)
  IMAGE_UNDISTORTER_ARGS=(--output_type COLMAP --copy_policy copy --max_image_size 2000)
  PATCH_MATCH_ARGS=(--PatchMatchStereo.max_image_size 2000 --PatchMatchStereo.geom_consistency 1)
  STEREO_FUSION_ARGS=(--input_type geometric --output_type PLY)
  POISSON_MESHER_ARGS=()
fi

echo "Starting COLMAP reconstruction pipeline..."
echo "Project: $PROJECT_NAME"
echo "Project directory: $PROJECT_DIR"
echo "Recon directory: $PROJECT_RECON_DIR"
echo ""

# 1) Feature extraction
echo "=========================================="
echo "Step 1: Feature Extraction"
echo "=========================================="
colmap feature_extractor \
  --database_path "$PROJECT_RECON_DIR/database.db" \
  --image_path "$PROJECT_DIR/images" \
  --ImageReader.mask_path "$PROJECT_DIR/masks" \
  "${FEATURE_EXTRACTOR_ARGS[@]}"
echo "✓ Feature extraction complete"
echo ""
read -p "Press Enter to continue to next step..."

# 2) Sequential matching
echo "=========================================="
echo "Step 2: Sequential Matching"
echo "=========================================="
colmap sequential_matcher \
  --database_path "$PROJECT_RECON_DIR/database.db" \
  "${SEQUENTIAL_MATCHER_ARGS[@]}"
echo "✓ Sequential matching complete"
echo ""
read -p "Press Enter to continue to next step..."

# 3) Sparse reconstruction
echo "=========================================="
echo "Step 3: Sparse Reconstruction (Mapping)"
echo "=========================================="
colmap mapper \
  --database_path "$PROJECT_RECON_DIR/database.db" \
  --image_path "$PROJECT_DIR/images" \
  --output_path "$PROJECT_RECON_DIR/sparse" \
  "${MAPPER_ARGS[@]}"
echo "✓ Sparse reconstruction complete"
echo ""
read -p "Press Enter to continue to next step..."

# 4) Image undistortion
echo "=========================================="
echo "Step 4: Image Undistortion"
echo "=========================================="
colmap image_undistorter \
  --image_path "$PROJECT_DIR/images" \
  --input_path "$PROJECT_RECON_DIR/sparse/0" \
  --output_path "$PROJECT_RECON_DIR/dense" \
  "${IMAGE_UNDISTORTER_ARGS[@]}"
echo "✓ Image undistortion complete"
echo ""
read -p "Press Enter to continue to next step..."

# 5) PatchMatch stereo
echo "=========================================="
echo "Step 5: PatchMatch Stereo"
echo "=========================================="
colmap patch_match_stereo \
  --workspace_path "$PROJECT_RECON_DIR/dense" \
  --workspace_format COLMAP \
  "${PATCH_MATCH_ARGS[@]}"
echo "✓ PatchMatch stereo complete"
echo ""
read -p "Press Enter to continue to next step..."

# 6) Stereo fusion
echo "=========================================="
echo "Step 6: Stereo Fusion"
echo "=========================================="
colmap stereo_fusion \
  --workspace_path "$PROJECT_RECON_DIR/dense" \
  --workspace_format COLMAP \
  "${STEREO_FUSION_ARGS[@]}" \
  --output_path "$PROJECT_RECON_DIR/dense/fused.ply"
echo "✓ Stereo fusion complete"
echo ""
read -p "Press Enter to continue to next step..."

# 7) Poisson meshing
echo "=========================================="
echo "Step 7: Poisson Meshing"
echo "=========================================="
colmap poisson_mesher \
  --input_path "$PROJECT_RECON_DIR/dense/fused.ply" \
  --output_path "$PROJECT_RECON_DIR/dense/mesh_poisson.ply" \
  "${POISSON_MESHER_ARGS[@]}"
echo "✓ Poisson meshing complete"
echo ""

echo "=========================================="
echo "✓ COLMAP reconstruction pipeline complete!"
echo "=========================================="
echo "Output files:"
echo "  - Sparse model: $PROJECT_RECON_DIR/sparse/0"
echo "  - Dense point cloud: $PROJECT_RECON_DIR/dense/fused.ply"
echo "  - Mesh: $PROJECT_RECON_DIR/dense/mesh_poisson.ply"
