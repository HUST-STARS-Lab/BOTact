"""Shared helpers for fusion evaluation."""

import os

import torch
import torch.nn as nn
from easydict import EasyDict

from models import build_model_from_cfg
from models.point_mamba_scan import serialization_func, apply_OrderScale

# 9-class recognition: B and H are excluded.
CLASS_NAMES = ["A", "C", "D", "E", "F", "G", "I", "J", "K"]
NUM_CLASSES = len(CLASS_NAMES)

VIT_FEAT_DIM = 768
DEFAULT_FUSION_CKPT = "checkpoints/fusion.pth"


def build_point_mamba(device: torch.device, ckpt_path: str = None) -> nn.Module:
    """Build PointMambaScan; optionally load a standalone PointMamba checkpoint."""
    point_cfg = EasyDict({
        "NAME": "PointMambaScan",
        "trans_dim": 384,
        "depth": 12,
        "cls_dim": 40,
        "group_size": 32,
        "num_group": 64,
        "encoder_dims": 384,
        "rms_norm": False,
        "drop_path": 0.1,
        "drop_out": 0.0,
        "use_cls_token": False,
        "max_head": False,
        "avg_head": True,
    })
    model = build_model_from_cfg(point_cfg).to(device)

    if ckpt_path:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"PointMamba checkpoint not found: {ckpt_path}")
        try:
            model.load_model_from_ckpt(ckpt_path)
            print(f"[INFO] Loaded PointMamba checkpoint from {ckpt_path}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load PointMamba checkpoint: {exc}") from exc

    return model


def extract_point_sequence(model: nn.Module, pts: torch.Tensor) -> torch.Tensor:
    """Extract PointMambaScan sequence features. pts: (B_total, 1024, 3)."""
    neighborhood, center = model.group_divider(pts)
    group_input_tokens = model.encoder(neighborhood)
    pos = model.pos_embed(center)

    _, _, _, group_input_tokens_forward, pos_forward = serialization_func(
        center, group_input_tokens, pos, "hilbert",
    )
    _, _, _, group_input_tokens_backward, pos_backward = serialization_func(
        center, group_input_tokens, pos, "hilbert-trans",
    )

    group_input_tokens_forward = apply_OrderScale(
        group_input_tokens_forward, model.OrderScale_gamma_1, model.OrderScale_beta_1,
    )
    group_input_tokens_backward = apply_OrderScale(
        group_input_tokens_backward, model.OrderScale_gamma_2, model.OrderScale_beta_2,
    )

    tokens = torch.cat([group_input_tokens_forward, group_input_tokens_backward], dim=1)
    pos_all = torch.cat([pos_forward, pos_backward], dim=1)
    return model.blocks(tokens, pos_all)  # (B_total, 128, 384)


def normalize_depth_for_vit(depth_stack: torch.Tensor) -> torch.Tensor:
    """Min-max normalize depth maps and map to pseudo-RGB for ViT input."""
    if depth_stack.dim() == 3:
        depth_stack = depth_stack.unsqueeze(1)

    batch_total = depth_stack.size(0)
    depth_flat = depth_stack.view(batch_total, -1)
    d_min = depth_flat.min(dim=1, keepdim=True)[0].view(batch_total, 1, 1, 1)
    d_max = depth_flat.max(dim=1, keepdim=True)[0].view(batch_total, 1, 1, 1)
    depth_norm = (depth_stack - d_min) / (d_max - d_min + 1e-6)
    depth_rgb = depth_norm.repeat(1, 3, 1, 1)
    # Use channel-0 ImageNet stats on all three repeated channels.
    return (depth_rgb - 0.485) / 0.229


def pick_most_free_gpu() -> int:
    """Return the GPU index with the most free memory."""
    if not torch.cuda.is_available():
        return 0
    best_id, best_free = 0, -1
    for gpu_id in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(gpu_id)
        if free > best_free:
            best_free, best_id = free, gpu_id
    return best_id


def resolve_device(device: str, gpu_id: int) -> torch.device:
    """Resolve runtime device; gpu_id=-1 picks the GPU with most free memory."""
    if not torch.cuda.is_available() or device == "cpu":
        return torch.device("cpu")
    if device.startswith("cuda:"):
        return torch.device(device)
    if gpu_id < 0:
        gpu_id = pick_most_free_gpu()
    return torch.device(f"cuda:{gpu_id}")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_names_for_eval(num_classes: int) -> list:
    """Return display names aligned with training class order."""
    if num_classes == NUM_CLASSES:
        return CLASS_NAMES.copy()
    return [f"Class{i + 1}" for i in range(num_classes)]
