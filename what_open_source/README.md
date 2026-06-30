# Visuo-Tactile Depth Fusion Framework

A multi-modal object classification framework that fuses **RGB images**, **depth maps**, and **point clouds** for 9-class recognition.

Built on top of [PointMamba](https://github.com/LMD0311/PointMamba) (NeurIPS 2024).

This repository provides **evaluation-only** code for reviewers to reproduce test-set results. All model weights are bundled in a single checkpoint (`fusion.pth`).

## Highlights

- **Fusion model**: ViT (RGB + depth) + PointMambaScan (point cloud) + fusion decoder
- **9-class classification**: A, C, D, E, F, G, I, J, K

## Repository Structure

```
├── eval.py                     # Evaluation entry point
├── models/                     # Fusion decoder, ViT encoder & PointMambaScan
├── datasets/FusionDataset.py   # Multi-modal dataset loader
├── utils/fusion_utils.py       # Eval helpers
├── mamba/                      # mamba-ssm source (install locally)
├── scripts/install_deps.sh     # Dependency install helper
├── checkpoints/fusion.pth      # Download and place here
└── data/test/                  # Download test set here
```

## Installation

This project is built on top of [PointMamba](https://github.com/LMD0311/PointMamba) (NeurIPS 2024). The dependency chain follows the [PointMamba installation guide](https://github.com/LMD0311/PointMamba/blob/main/docs/USAGE.md): PyTorch → basic packages → PointNet++ → KNN_CUDA → Mamba.

PointMamba-related code is **bundled in this repo** (`models/`, `mamba/`). You do **not** need to clone PointMamba separately.

### Requirements

| Item | Recommendation |
|------|----------------|
| OS | Linux (PointMamba tested on Ubuntu 20.04) |
| GPU | NVIDIA GPU with CUDA |
| Python | 3.10 (PointMamba upstream uses 3.9) |
| PyTorch | 2.x + torchvision (CUDA build) |

### 1. Clone this repository

```bash
git clone <YOUR_REPO_URL>
cd fusion_open_source
```

### 2. Create conda environment & install PyTorch

Same workflow as [PointMamba § Installation](https://github.com/LMD0311/PointMamba/blob/main/docs/USAGE.md#installation):

```bash
conda create -n vtdfusion python=3.10 -y
conda activate vtdfusion

# Adjust CUDA version to match your driver (example: cu118)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Python dependencies & Mamba

```bash
bash scripts/install_deps.sh
```

This script installs `requirements.txt`, `causal-conv1d`, and the bundled `mamba/` package.

If `mamba-ssm` compilation fails, retry:

```bash
pip install -e mamba/ --no-build-isolation
```

### 4. Install PointNet++ ops (required)

PointMambaScan uses furthest-point sampling via `pointnet2_ops`. Install as in the [PointMamba guide](https://github.com/LMD0311/PointMamba/blob/main/docs/USAGE.md#installation):

```bash
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

### 5. Install KNN_CUDA (required)

PointMambaScan grouping uses GPU kNN:

```bash
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

If the wheel fails on your Python/CUDA version, build from [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) or refer to [PointMamba Issues](https://github.com/LMD0311/PointMamba/issues).

### 6. Verify installation

```bash
python -c "
from mamba_ssm.modules.mamba_simple import Mamba
from knn_cuda import KNN
from pointnet2_ops import pointnet2_utils
print('All dependencies OK')
"
```

## Download (Google Drive)

**Test set and checkpoint** are hosted on Google Drive (not included in this repository due to size):

[Google Drive — dataset & checkpoint](https://drive.google.com/drive/folders/1x_mntoVmkfxibr1g-NgdgJ08PAUcrYcP?usp=drive_link)

| Download | Place at |
|----------|----------|
| Test set | `data/test/` |
| `fusion.pth` | `checkpoints/fusion.pth` |

```bash
mkdir -p checkpoints data/test
# Copy downloaded files into the paths above
```

`fusion.pth` contains all weights needed for evaluation: fusion decoder, PointMambaScan, RGB ViT, and depth ViT. No separate PointMamba checkpoint is required.

## Dataset

Download `data/test/` from [Google Drive](#download-google-drive), then verify the layout below.

```
data/test/
└── A1/                          # Class letter + instance id
    ├── camera0/{sid}.png        # RGB (3 views)
    ├── camera2/{sid}.png
    ├── camera4/{sid}.png
    ├── depth0/{sid}.npy         # Depth (3 views)
    ├── depth2/{sid}.npy
    ├── depth4/{sid}.npy
    ├── point0/{sid}.txt         # Point cloud, 1024 points (3 views)
    ├── point2/{sid}.txt
    └── point4/{sid}.txt
```

Classes: **A, C, D, E, F, G, I, J, K** (B and H excluded).

## Checkpoint

| File | Description |
|------|-------------|
| `checkpoints/fusion.pth` | Full fusion model (required for eval) |

The checkpoint stores four state dicts: `fusion_decoder`, `point_model`, `rgb_vit_model`, and `depth_vit_model`.

## Evaluation

```bash
python eval.py --test_root data/test
```

This will print overall/per-class accuracy and save a confusion matrix heatmap to `results/confusion_matrix_heatmap.png`.

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt_fusion` | `checkpoints/fusion.pth` | Fusion checkpoint path |
| `--test_root` | `data/test` | Test set root |
| `--output_dir` | `results` | Confusion matrix output directory |
| `--batch_size` | `16` | Batch size (reduce if OOM) |
| `--gpu_id` | `-1` | CUDA device id; `-1` auto-selects the most free GPU |
| `--device` | `cuda` / `cpu` | Device override |

Default `batch_size` is 16 (effective 48 images per step due to 3-view stacking). Reduce if OOM.

## Acknowledgement

This project is based on [PointMamba](https://github.com/LMD0311/PointMamba), [Mamba](https://github.com/state-spaces/mamba), and related point cloud analysis codebases.

## License

Apache-2.0 (see [LICENSE](LICENSE))
