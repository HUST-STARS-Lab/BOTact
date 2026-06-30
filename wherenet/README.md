# BOTact Object Localization and Registration Framework

An offline tactile-to-3D localization pipeline that reconstructs a target contact point cloud from a single tactile image and registers a CAD/template point cloud to it with **BUFFER-X**.

This repository provides code for:

- background-difference tactile segmentation
- learned depth estimation from segmented tactile images
- depth-to-point-cloud reconstruction
- BUFFER-X point-cloud registration and pose estimation

## Highlights

- **End-to-end offline pipeline**: image segmentation -> depth estimation -> point-cloud reconstruction -> registration
- **Template-based localization**: aligns a model/template `.pcd` file to the reconstructed target point cloud
- **BUFFER-X registration**: uses multi-scale patch embedding and hierarchical correspondence matching
- **Demo-ready interface**: includes a command-line entry point and a PyQt/Open3D reconstruction UI prototype

## Repository Structure

```text
wherenet/
|-- main.py                         # Offline reconstruction and registration entry point
|-- config/                         # Dataset and BUFFER-X configuration files
|-- models/                         # BUFFER-X model and registration modules
|-- reconstruction/                 # Segmentation, depth estimation, UI, and runtime pipeline
|-- cpp_wrappers/                   # C++/CUDA neighbor and grid-subsampling wrappers
|-- loss/                           # Descriptor and pose loss utilities
|-- utils/                          # Point-cloud, SE(3), timing, and result helpers
|-- demo/                           # Example tactile images and template point clouds
|-- torch-batch-svd/                # Bundled batch SVD extension source
|-- install.sh                      # Dependency install helper
`-- checkpoints/                    # Download and place model weights here
```

## Installation

### Requirements

| Item            | Recommendation                     |
| --------------- | ---------------------------------- |
| OS              | Linux / Ubuntu 20.04+              |
| GPU             | NVIDIA GPU with CUDA               |
| Python          | 3.9 or 3.10                        |
| PyTorch         | CUDA build, e.g. cu118             |
| System packages | gcc, g++, cmake, ninja, Eigen, TBB |

### 1. Clone this repository

```bash
git clone <YOUR_REPO_URL>
cd BOTact/where_open_source
```

### 2. Create conda environment

```bash
conda create -n botact python=3.10 -y
conda activate botact
```

### 3. Install dependencies

```bash
bash install.sh
```

The script installs PyTorch, Open3D, NumPy 1.26.3, PointNet++ ops, KNN_CUDA, Python utility packages, C++ wrappers, and `torch-batch-svd`.

If you prefer manual installation, install the packages in `install.sh` and then compile the wrappers:

```bash
cd cpp_wrappers
sh compile_wrappers.sh
cd ..
```

## Download Checkpoints

Model weights are not included in this repository due to size. Place the downloaded checkpoints in the following paths:

| File                              | Place at                                   | Used by                   |
| --------------------------------- | ------------------------------------------ | ------------------------- |
| `depth_mlp_7x7.pth`             | `checkpoints/depth_mlp_7x7.pth`          | tactile depth estimation  |
| `best.pth` for descriptor stage | `checkpoints/registration/Desc/best.pth` | BUFFER-X descriptor stage |
| `best.pth` for pose stage       | `checkpoints/registration/Pose/best.pth` | BUFFER-X pose stage       |
| `vit_best_8classes.pth`         | `checkpoints/vit_best_8classes.pth`      | optional UI classifier    |

The registration loader also searches fallback paths:

```text
snapshot/kitti/<stage>/best.pth
snapshot/threedmatch/<stage>/best.pth
```

## Demo Data

The demo folder is expected to contain paired tactile images and background images, plus template point clouds:

```text
demo/
|-- m3.png
|-- m3_bg.png
|-- m4.png
|-- m4_bg.png
|-- GB_M3X4.pcd
`-- GB_M4X6.pcd
```

If `.pcd` files are missing after cloning, download or copy the template point clouds into `demo/` or pass another template path with `--model-pcd`.

## Offline Reconstruction and Registration

Run the M3 demo:

```bash
python main.py \
  --image demo/m3.png \
  --background demo/m3_bg.png \
  --model-pcd demo/GB_M3X4.pcd \
  --output-dir outputs/m3_demo
```

Run the M4 demo:

```bash
python main.py \
  --image demo/m4.png \
  --background demo/m4_bg.png \
  --model-pcd demo/GB_M4X6.pcd \
  --output-dir outputs/m4_demo
```

The script prints five pipeline stages:

```text
[1/5] Reading image
[2/5] Running background-difference segmentation
[3/5] Running depth estimation
[4/5] Reconstructing target point cloud
[5/5] Running BUFFER-X registration
```

### Outputs

Each run writes outputs to `outputs/<image_stem>_<timestamp>/` unless `--output-dir` is provided.

| Output                         | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| `segmented.png`              | Background-difference tactile segmentation result |
| `depth.npy`                  | Predicted dense depth map                         |
| `depth_visualization.png`    | Depth map visualization                           |
| `reconstructed_target.pcd`   | Reconstructed tactile target point cloud          |
| `registered_model.pcd`       | Template point cloud after registration           |
| `registered_merged.pcd`      | Registered template plus target point cloud       |
| `registration_transform.txt` | Estimated 4x4 transformation matrix               |

### CLI Arguments

| Argument                  | Default                 | Description                                                  |
| ------------------------- | ----------------------- | ------------------------------------------------------------ |
| `--image`               | `demo.png`            | Input tactile image                                          |
| `--background`          | `background.png`      | Background image for segmentation                            |
| `--model-pcd`           | required                | Template/model point cloud used as registration source       |
| `--output-dir`          | auto timestamped folder | Output directory                                             |
| `--min-register-points` | `1000`                | Skip registration if too few target points are reconstructed |

## Optional UI

The `reconstruction/app.py` module contains an experimental PyQt/Open3D UI for live-style reconstruction and visualization.

```bash
python reconstruction/app.py
```

The UI expects the same runtime assets as the command-line pipeline: depth checkpoint, registration checkpoints, and template point clouds.

## Notes

- `main.py` uses CUDA automatically when available and falls back to CPU.
- Background images should be captured under the same camera and lighting setup as the tactile input.
- The depth estimator resizes/uses a 640x480 segmentation map and converts valid depth pixels to an ASCII `.pcd`.
- Registration checkpoints are loaded stage by stage according to `cfg.train.all_stage = ["Desc", "Pose"]`.

## Acknowledgement

This project builds on BUFFER-X-style point-cloud registration, PointNet++ ops, KNN_CUDA, Open3D, and torch-batch-svd.

## License

Please refer to the repository-level license and bundled third-party license files.