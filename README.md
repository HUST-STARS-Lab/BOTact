# BOTact

BOTact is a visuo-tactile perception framework for object understanding and localization. The repository is organized into two complementary modules:

- **WhatNet** (`whatnet/`): multi-modal object classification with RGB images, depth maps, and point clouds.
- **WhereNet** (`wherenet/`): tactile depth reconstruction and template-based 3D localization with BUFFER-X registration.

Together, the two modules answer two questions for tactile manipulation:

1. **What is the object?** Recognize the object class from multi-view visual/tactile-derived modalities.
2. **Where is the object?** Reconstruct the contact geometry and estimate the pose of a template point cloud.

## Repository Structure

```text
BOTact/
|-- README.md                 # Project-level overview
|-- whatnet/                  # Object classification module
|   |-- eval.py               # Evaluation entry point
|   |-- models/               # Fusion decoder, ViT encoder, PointMambaScan
|   |-- datasets/             # Multi-modal dataset loader
|   |-- mamba/                # Bundled mamba-ssm source
|   `-- README.md             # WhatNet setup and evaluation guide
`-- wherenet/                 # Object localization and registration module
    |-- main.py               # Offline reconstruction and registration entry point
    |-- models/               # BUFFER-X model and registration modules
    |-- reconstruction/       # Segmentation, depth estimation, UI, runtime pipeline
    |-- config/               # BUFFER-X and dataset configuration
    |-- cpp_wrappers/         # C++/CUDA wrappers
    `-- README.md             # WhereNet setup and demo guide
```

## Pipeline

```text
Input RGB/depth/point-cloud samples
        |
        v
WhatNet: RGB ViT + Depth ViT + PointMambaScan
        |
        v
Object class prediction

Input tactile image + background image + template PCD
        |
        v
WhereNet: segmentation -> depth estimation -> point-cloud reconstruction -> BUFFER-X registration
        |
        v
Object pose / registered template point cloud
```

## Modules

### WhatNet: Object Classification

`whatnet/` provides evaluation-only code for 9-class object recognition. It fuses:

- RGB images from three camera views
- depth maps from three views
- point clouds from three views

The classification model combines ViT encoders, PointMambaScan, and a fusion decoder. See [whatnet/README.md](whatnet/README.md) for dependency installation, dataset layout, checkpoint placement, and evaluation commands.

Quick start:

```bash
cd whatnet
bash scripts/install_deps.sh
python eval.py --test_root data/test
```

Required runtime assets:

| Asset | Place at |
|-------|----------|
| Test set | `whatnet/data/test/` |
| Fusion checkpoint | `whatnet/checkpoints/fusion.pth` |

### WhereNet: Object Localization

`wherenet/` reconstructs a target point cloud from a tactile image and registers a template point cloud to it. The offline pipeline runs:

```text
background-difference segmentation
-> learned depth estimation
-> depth-to-point-cloud reconstruction
-> BUFFER-X registration
```

See [wherenet/README.md](wherenet/README.md) for installation, checkpoint paths, demo data, and command-line arguments.

Quick start:

```bash
cd wherenet
bash install.sh
python main.py \
  --image demo/m3.png \
  --background demo/m3_bg.png \
  --model-pcd demo/GB_M3X4.pcd \
  --output-dir outputs/m3_demo
```

Required runtime assets:

| Asset | Place at |
|-------|----------|
| Depth checkpoint | `wherenet/checkpoints/depth_mlp_7x7.pth` |
| BUFFER-X descriptor checkpoint | `wherenet/checkpoints/registration/Desc/best.pth` |
| BUFFER-X pose checkpoint | `wherenet/checkpoints/registration/Pose/best.pth` |
| Template point clouds | `wherenet/demo/` or custom `--model-pcd` path |

## Installation

The two modules have separate dependency stacks. Create separate environments to avoid CUDA extension conflicts.

### WhatNet environment

```bash
cd whatnet
conda create -n botact_what python=3.10 -y
conda activate botact_what
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
bash scripts/install_deps.sh
```

### WhereNet environment

```bash
cd wherenet
conda create -n botact_where python=3.10 -y
conda activate botact_where
bash install.sh
```

Both modules assume a Linux/CUDA environment for the point-cloud and registration operators.

## Data and Checkpoints

Large datasets, checkpoints, and generated outputs are not committed to this repository. Download the released assets and place them under each module as documented in the module README files.

Common ignored paths include:

```text
whatnet/data/
whatnet/checkpoints/
wherenet/data/
wherenet/checkpoints/
wherenet/snapshot/
wherenet/outputs/
```

## Quick Commands

Evaluate WhatNet:

```bash
cd whatnet
python eval.py --test_root data/test
```

Run WhereNet offline localization:

```bash
cd wherenet
python main.py \
  --image demo/m3.png \
  --background demo/m3_bg.png \
  --model-pcd demo/GB_M3X4.pcd
```

## Outputs

WhatNet writes classification metrics and a confusion matrix heatmap under `whatnet/results/`.

WhereNet writes per-run reconstruction and registration outputs under `wherenet/outputs/`, including:

- `segmented.png`
- `depth.npy`
- `depth_visualization.png`
- `reconstructed_target.pcd`
- `registered_model.pcd`
- `registered_merged.pcd`
- `registration_transform.txt`

## Acknowledgement

BOTact builds on PointMamba, Mamba, PointNet++ ops, KNN_CUDA, BUFFER-X-style point-cloud registration, Open3D, and torch-batch-svd.

## License

Please refer to the repository-level license and bundled third-party license files in each module.
