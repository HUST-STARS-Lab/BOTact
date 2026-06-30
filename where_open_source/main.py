"""Offline reconstruction entry point.

Given one RGB image and one model point cloud, this script runs:
background-difference segmentation -> depth estimation -> point-cloud reconstruction -> registration.

python main.py --image /mnt/data/yycdata/explore/code_public/demo/m6.png --background /mnt/data/yycdata/explore/code_public/demo/m6_bg.png --model-pcd /mnt/data/yycdata/explore/code_public/demo/GB_M6X8.pcd
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RECON_DIR = PROJECT_ROOT / "reconstruction"
if str(RECON_DIR) not in sys.path:
    sys.path.insert(0, str(RECON_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline segmentation, depth estimation, point-cloud reconstruction, "
            "and BUFFER-X registration."
        )
    )
    parser.add_argument(
        "--image",
        default="demo.png",
        help="Input image path. Default: demo.png",
    )
    parser.add_argument(
        "--model-pcd",
        required=True,
        help="Template/model point-cloud path used as the registration source.",
    )
    parser.add_argument(
        "--background",
        default="background.png",
        help=(
            "Background image path for background-difference segmentation. "
            "Default: background.png"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: outputs/<image_stem>_<timestamp>",
    )
    parser.add_argument(
        "--min-register-points",
        type=int,
        default=1000,
        help="Minimum reconstructed target points required before registration.",
    )
    return parser.parse_args()


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def save_depth_outputs(depth_map: np.ndarray, out_dir: Path) -> tuple[Path, Path]:
    depth_npy = out_dir / "depth.npy"
    depth_png = out_dir / "depth_visualization.png"
    np.save(depth_npy, depth_map)

    nonzero = depth_map[depth_map > 0]
    if nonzero.size:
        hi = float(np.percentile(nonzero, 99))
        lo = float(np.percentile(nonzero, 1))
        denom = max(hi - lo, 1e-6)
        vis = np.clip((depth_map - lo) / denom * 255.0, 0, 255).astype(np.uint8)
    else:
        vis = np.zeros_like(depth_map, dtype=np.uint8)
    cv2.imwrite(str(depth_png), vis)
    return depth_npy, depth_png


def save_registration_outputs(
    src_transformed: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transform: np.ndarray,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    registered_model = out_dir / "registered_model.pcd"
    registered_merged = out_dir / "registered_merged.pcd"
    transform_txt = out_dir / "registration_transform.txt"

    o3d.io.write_point_cloud(str(registered_model), src_transformed, write_ascii=True)
    merged = src_transformed + target
    o3d.io.write_point_cloud(str(registered_merged), merged, write_ascii=True)
    np.savetxt(transform_txt, transform, fmt="%.8f")
    return registered_model, registered_merged, transform_txt


def main() -> int:
    args = parse_args()

    try:
        global cv2, np, o3d, torch
        global BufferXRegistrar, DepthEstimator, MODEL_DEPTH_PATH, Segmenter, ensure_dirs

        import cv2
        import numpy as np
        import open3d as o3d
        import torch

        from bufferx_registrar import BufferXRegistrar
        from depth_estimator import DepthEstimator
        from runtime_config import MODEL_DEPTH_PATH, ensure_dirs
        from segmenter import Segmenter
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Please install the project environment first "
            f"and retry. Original error: {exc}"
        ) from exc

    ensure_dirs()

    image_path = Path(args.image).resolve()
    model_pcd_path = Path(args.model_pcd).resolve()
    background_path = Path(args.background).resolve()

    if not model_pcd_path.exists():
        raise FileNotFoundError(f"Model point cloud not found: {model_pcd_path}")
    if not background_path.exists():
        raise FileNotFoundError(f"Background image not found: {background_path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else PROJECT_ROOT / "outputs" / f"{image_path.stem}_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Reading image: {image_path}")
    image_bgr = read_bgr(image_path)
    bg_bgr = read_bgr(background_path)

    print(f"[2/5] Running background-difference segmentation: {background_path}")
    segmenter = Segmenter(img_width=640, img_height=480)
    segmented = segmenter.segment(image_bgr, bg_bgr)

    segmented_path = out_dir / "segmented.png"
    cv2.imwrite(str(segmented_path), segmented)

    print("[3/5] Running depth estimation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_estimator = DepthEstimator(device=device, model_path=MODEL_DEPTH_PATH, patch_size=7)
    depth_map = depth_estimator.estimate_depth(segmented)
    depth_npy, depth_png = save_depth_outputs(depth_map, out_dir)

    print("[4/5] Reconstructing target point cloud...")
    points = depth_estimator.depth_to_pointcloud(depth_map)
    target_pcd_path = out_dir / "reconstructed_target.pcd"
    DepthEstimator.save_ascii_pcd(points, str(target_pcd_path))
    print(f"      Reconstructed points: {len(points)}")

    if len(points) < args.min_register_points:
        print(
            "      Registration skipped: "
            f"point count {len(points)} < {args.min_register_points}"
        )
        print_outputs(out_dir, segmented_path, depth_npy, depth_png, target_pcd_path)
        return 0

    print("[5/5] Running BUFFER-X registration...")
    registrar = BufferXRegistrar(device=device)
    src_transformed, target, transform, _times = registrar.register(
        str(model_pcd_path),
        str(target_pcd_path),
    )
    registered_model, registered_merged, transform_txt = save_registration_outputs(
        src_transformed,
        target,
        transform,
        out_dir,
    )

    print_outputs(
        out_dir,
        segmented_path,
        depth_npy,
        depth_png,
        target_pcd_path,
        registered_model,
        registered_merged,
        transform_txt,
    )
    return 0


def print_outputs(out_dir: Path, *paths: Path) -> None:
    print("\nDone. Outputs:")
    print(f"  output_dir: {out_dir}")
    for path in paths:
        print(f"  {path.name}: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
