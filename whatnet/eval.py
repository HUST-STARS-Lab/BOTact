import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets.FusionDataset import FusionDataset
from models.fusion_decoder import FusionDecoder
from models.visual_encoder import build_visual_encoder
from utils.fusion_utils import (
    CLASS_NAMES,
    DEFAULT_FUSION_CKPT,
    VIT_FEAT_DIM,
    build_point_mamba,
    class_names_for_eval,
    extract_point_sequence,
    normalize_depth_for_vit,
    resolve_device,
    set_seed,
)

FUSION_CKPT_KEYS = ("fusion_decoder", "point_model", "rgb_vit_model", "depth_vit_model")


@torch.no_grad()
def evaluate_detailed(
    point_model,
    rgb_encoder,
    depth_encoder,
    fusion_decoder,
    loader,
    device,
    num_classes: int = 9,
):
    """Compute overall/per-class accuracy and confusion matrix."""
    fusion_decoder.eval()
    point_model.eval()
    rgb_encoder.eval()
    depth_encoder.eval()

    total_correct = 0
    total_samples = 0
    cls_total = torch.zeros(num_classes, dtype=torch.long)
    cls_correct = torch.zeros(num_classes, dtype=torch.long)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(loader, desc="[Eval-Test]", ncols=100)
    for sample_dict, labels in pbar:
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        img_stack = torch.cat(
            [sample_dict["camera0"], sample_dict["camera2"], sample_dict["camera4"]],
            dim=0,
        ).to(device, non_blocking=True)

        depth_stack = torch.cat(
            [sample_dict["depth0"], sample_dict["depth2"], sample_dict["depth4"]],
            dim=0,
        ).to(device, non_blocking=True)
        depth_rgb_stack = normalize_depth_for_vit(depth_stack)

        pc_stack = torch.cat(
            [sample_dict["point0"], sample_dict["point2"], sample_dict["point4"]],
            dim=0,
        ).to(device, non_blocking=True)

        seq_feat_stack = extract_point_sequence(point_model, pc_stack)
        img_feat_stack = rgb_encoder(img_stack)
        depth_feat_stack = depth_encoder(depth_rgb_stack)

        seq_feat = (
            seq_feat_stack.view(3, batch_size, 128, 384)
            .permute(1, 0, 2, 3)
            .reshape(batch_size, -1, 384)
        )
        img_feat_flat = img_feat_stack.view(3, batch_size, -1).permute(1, 0, 2).reshape(batch_size, -1)
        depth_feat_flat = (
            depth_feat_stack.view(3, batch_size, -1).permute(1, 0, 2).reshape(batch_size, -1)
        )

        logits = fusion_decoder(seq_feat, img_feat_flat, depth_feat_flat)
        preds = logits.argmax(dim=1)

        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        for true_label, pred_label in zip(labels.view(-1), preds.view(-1)):
            t_i = int(true_label.item())
            p_i = int(pred_label.item())
            if 0 <= t_i < num_classes:
                cls_total[t_i] += 1
                if t_i == p_i:
                    cls_correct[t_i] += 1
                if 0 <= p_i < num_classes:
                    confusion[t_i, p_i] += 1

        acc_batch = (preds == labels).float().mean().item() * 100.0
        pbar.set_postfix(acc=f"{acc_batch:.2f}%")

    overall_acc = 100.0 * total_correct / max(1, total_samples)
    return overall_acc, cls_total, cls_correct, confusion


def save_confusion_matrix_heatmap(
    confusion: torch.Tensor,
    class_names: list,
    output_path: str,
) -> None:
    """Save a row-normalized confusion matrix heatmap."""
    num_classes = len(class_names)
    confusion_np = confusion.numpy().astype(float)
    row_sums = confusion_np.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    confusion_percent = (confusion_np / row_sums) * 100.0

    base_cmap = plt.cm.Blues
    colors = base_cmap(np.linspace(0.15, 0.9, 256))
    light_blues = LinearSegmentedColormap.from_list("Blues_light", colors)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion_percent, cmap=light_blues, vmin=0.0, vmax=100.0)
    ax.set_xlabel("Prediction", fontsize=16, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=16, fontweight="bold")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    cbar.ax.tick_params(labelsize=12)

    for i in range(num_classes):
        for j in range(num_classes):
            val = confusion_percent[i, j]
            ax.text(j, i, f"{int(round(val))}", ha="center", va="center", color="k", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(42)

    parser = argparse.ArgumentParser(description="Evaluate fusion model on test set")
    parser.add_argument("--gpu_id", type=int, default=-1,
                        help="CUDA device id; -1 auto-selects the most free GPU")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cuda / cpu / cuda:N")
    parser.add_argument("--ckpt_fusion", type=str, default=DEFAULT_FUSION_CKPT,
                        help="Fusion checkpoint (contains all model weights)")
    parser.add_argument("--test_root", type=str, default="data/test",
                        help="Test set root directory")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for confusion matrix plot")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=len(CLASS_NAMES),
                        help="Number of classes (default: 9)")
    args = parser.parse_args()

    device = resolve_device(args.device, args.gpu_id)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"[INFO] Eval device: {device} (current cuda: {torch.cuda.current_device()})")

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    depth_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x),
        transforms.Resize((224, 224)),
    ])
    point_transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x),
    ])

    test_dataset = FusionDataset(
        root=args.test_root, subset="test",
        img_transform=img_transform, depth_transform=depth_transform,
        point_transform=point_transform, shuffle=False,
        classes=CLASS_NAMES,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
    )
    print(f"[INFO] Test set size: {len(test_dataset)}")

    if not os.path.exists(args.ckpt_fusion):
        raise FileNotFoundError(f"Fusion checkpoint not found: {args.ckpt_fusion}")

    try:
        ckpt = torch.load(args.ckpt_fusion, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt_fusion, map_location="cpu")

    missing_keys = [key for key in FUSION_CKPT_KEYS if key not in ckpt]
    if missing_keys:
        raise KeyError(
            f"Fusion checkpoint is missing required keys: {missing_keys}. "
            "Download the full fusion.pth from Google Drive."
        )

    print(f"[INFO] Loaded fusion checkpoint from {args.ckpt_fusion}")
    print(f"[INFO] Best test acc in checkpoint: {ckpt.get('test_acc', 'N/A')}%")

    point_model = build_point_mamba(device)

    print("[INFO] Building RGB ViT encoder...")
    rgb_encoder, vit_dim = build_visual_encoder(device)

    print("[INFO] Building depth ViT encoder...")
    depth_encoder, _ = build_visual_encoder(device)
    assert vit_dim == VIT_FEAT_DIM

    fusion_decoder = FusionDecoder(
        trans_dim=384,
        img_dim=vit_dim * 3,
        depth_dim=vit_dim * 3,
        cls_dim=args.num_classes,
        point_pool="avg",
        normalize=True,
    ).to(device)

    fusion_decoder.load_state_dict(ckpt["fusion_decoder"])
    point_model.load_state_dict(ckpt["point_model"])
    rgb_encoder.load_state_dict(ckpt["rgb_vit_model"])
    depth_encoder.load_state_dict(ckpt["depth_vit_model"])

    fusion_decoder.to(device)
    point_model.to(device)
    rgb_encoder.to(device)
    depth_encoder.to(device)

    overall_acc, cls_total, cls_correct, confusion = evaluate_detailed(
        point_model, rgb_encoder, depth_encoder, fusion_decoder,
        test_loader, device, num_classes=args.num_classes,
    )

    class_names = class_names_for_eval(args.num_classes)

    print("\n========== Eval Results ==========")
    print(f"Overall accuracy: {overall_acc:.2f}%  ({cls_total.sum().item()} samples)")

    print("\nPer-class accuracy:")
    for class_idx in range(args.num_classes):
        total_c = cls_total[class_idx].item()
        correct_c = cls_correct[class_idx].item()
        acc_c = 100.0 * correct_c / total_c if total_c > 0 else 0.0
        name = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
        print(f"  Class {class_idx} ({name}): {acc_c:.2f}%  ({correct_c}/{total_c})")

    print("\nMisclassification details (true -> predicted: count):")
    for true_idx in range(args.num_classes):
        total_t = cls_total[true_idx].item()
        if total_t == 0:
            continue
        name_t = class_names[true_idx] if true_idx < len(class_names) else str(true_idx)
        row = confusion[true_idx]
        wrong_indices = [p for p in range(args.num_classes) if p != true_idx and row[p] > 0]
        if not wrong_indices:
            continue
        print(f"  True class {true_idx} ({name_t}):")
        for pred_idx in wrong_indices:
            cnt = row[pred_idx].item()
            name_p = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
            print(f"    -> Pred {pred_idx} ({name_p}): {cnt}")

    try:
        out_path = os.path.join(args.output_dir, "confusion_matrix_heatmap.png")
        save_confusion_matrix_heatmap(confusion, class_names, out_path)
        print(f"\n[INFO] Confusion matrix heatmap saved to: {out_path}")
    except Exception as exc:
        print(f"[WARN] Failed to save confusion matrix heatmap: {exc}")


if __name__ == "__main__":
    main()
