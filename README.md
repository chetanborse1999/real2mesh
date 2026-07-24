# Video-to-3D Reconstruction

This project converts a video into raw and object-masked 3D meshes:

```text
video → JPEG frames → SAM2 object masks → COLMAP → PLY meshes
```

The complete walkthrough uses the example
[`ferrari_lego` video](https://drive.google.com/file/d/1b5y2e2xu1iF_H6hQpi_Z1J8YOyzPqMqD/view?usp=sharing).
Download it, create a `ferrari_lego` directory in the repository, and save the
video as:

```text
ferrari_lego/ferrari_lego.mp4
```

You can replace it with any other video. Use a different dataset directory and
video filename as needed, then substitute those paths for `ferrari_lego` in
the commands below.

Commands in this guide are run in a terminal from `~/real2mesh`. The example
project is named `ferrari_lego`; replace that name for another dataset.

## Requirements

- Linux with Conda, Miniconda, or Miniforge
- Python 3.12.11
- A graphical desktop for box selection and mesh viewing
- An NVIDIA GPU, working driver, and CUDA-enabled COLMAP for dense
  reconstruction
- At least 20 GB of available disk space

SAM2 can run on CPU, although the large model is slow. COLMAP PatchMatch
stereo normally requires CUDA. Confirm GPU access with:

```bash
nvidia-smi
```

## File structure

Before running the workflow:

```text
~/real2mesh/
├── requirements.txt
├── extract_frames.py
├── segment_images_sam2.py
├── colmap_config.sh
├── colmap_reconstruction_raw.sh
├── colmap_reconstruction_masked.sh
├── view_mesh.py
├── sam2/
│   ├── sam2/configs/sam2.1/sam2.1_hiera_l.yaml
│   └── checkpoints/sam2.1_hiera_large.pt
└── ferrari_lego/
    └── ferrari_lego.mp4
```

The workflow creates:

```text
ferrari_lego/
├── images/          # Extracted JPEGs
├── masks/           # One SAM2 mask per JPEG
├── recon/           # Raw COLMAP results
└── recon_masked/    # Masked COLMAP results
```

The segmentation filename uses underscores: `segment_images_sam2.py`.

## Install the Conda environment

Create and activate the environment:

```bash
conda create -n real2mesh3d \
  python=3.12.11 \
  pip=25.2 \
  setuptools=78.1.1 \
  wheel=0.45.1 \
  -y
conda activate real2mesh3d
cd ~/real2mesh
```

Activate `real2mesh3d` again whenever a new terminal is opened.

Install CUDA-enabled COLMAP, FFmpeg, and OpenCV:

```bash
conda install -c conda-forge \
  colmap=3.13.0=cuda_129h1e19e12_4 \
  ffmpeg=8.1.2=gpl_h54862ce_904 \
  opencv=5.0.0=qt6_h12f39d8_603 \
  -y
colmap -h
```

The COLMAP header should state `with CUDA`.

Install PyTorch. For CUDA 12.8:

```bash
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only SAM2:

```bash
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

Clone the tested SAM2 revision if `~/real2mesh/sam2` does not exist:

```bash
cd ~/real2mesh
git clone https://github.com/facebookresearch/sam2.git sam2
git -C sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
```

Install the pinned Python dependencies and local SAM2 package:

```bash
cd ~/real2mesh
SAM2_BUILD_CUDA=0 python -m pip install \
  --no-build-isolation \
  -r requirements.txt
```

Download checkpoints:

```bash
cd ~/real2mesh/sam2/checkpoints
./download_ckpts.sh
cd ~/real2mesh
```

Make the pipelines executable:

```bash
chmod +x colmap_reconstruction_raw.sh colmap_reconstruction_masked.sh
```

## COLMAP configuration

Both pipelines read `colmap_config.sh`. It contains camera, SIFT, matching,
mapping, image-size, fusion, and meshing options.

GPU feature extraction and matching are enabled by default. Set these options
to `0` in the config to run those two stages on CPU:

```text
--FeatureExtraction.use_gpu 0
--FeatureMatching.use_gpu 0
```

This does not make PatchMatch stereo CPU-compatible.

## Run the workflow

Start every session with:

```bash
conda activate real2mesh3d
cd ~/real2mesh
```

### 1. Extract frames

```bash
python extract_frames.py \
  --video_file ferrari_lego/ferrari_lego.mp4 \
  --frame_rate 2 \
  --output_dir ferrari_lego/images
```

This produces `frame_000000.jpg`, `frame_000001.jpg`, and so on.

**Destructive:** after validating the video, extraction empties `images/`
before writing new frames.

### 2. Segment one object

GPU:

```bash
python segment_images_sam2.py \
  ferrari_lego/images ferrari_lego/masks \
  --checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda
```

An image window opens. Draw a box around the object and press Enter or Space.
SAM2 tracks that object and writes one mask per frame. Use `--device cpu` if PyTorch reports no CUDA. 

Add `--prompt-frame 10` if the object is clearer in frame index 10. Tracking
then runs forward and backward.

White mask pixels retain COLMAP features; black pixels exclude them. Segmentation does not remove stale masks. If the images changed, reset masks:

```bash
rm -rf ferrari_lego/masks
mkdir -p ferrari_lego/masks
```

### 3. Run raw reconstruction

```bash
./colmap_reconstruction_raw.sh ferrari_lego
```

Press Enter when the script pauses between stages.

**Destructive:** the script deletes only `ferrari_lego/recon` at startup so
old databases and depth maps cannot contaminate a new run.

### 4. Run masked reconstruction

```bash
./colmap_reconstruction_masked.sh ferrari_lego
```

**Destructive:** this script deletes only `ferrari_lego/recon_masked`.
It does not modify `recon`, images, or masks.

### 5. View meshes

Raw mesh:

```bash
python view_mesh.py --mesh ferrari_lego/recon/dense/mesh_poisson.ply
```

Masked mesh:

```bash
python view_mesh.py \
  --mesh ferrari_lego/recon_masked/dense/mesh_poisson.ply
```

## Outputs

| Result | Raw path | Masked path |
|---|---|---|
| Database | `recon/database.db` | `recon_masked/database.db` |
| Sparse model | `recon/sparse/0/` | `recon_masked/sparse/0/` |
| Dense cloud | `recon/dense/fused.ply` | `recon_masked/dense/fused.ply` |
| Mesh | `recon/dense/mesh_poisson.ply` | `recon_masked/dense/mesh_poisson.ply` |
