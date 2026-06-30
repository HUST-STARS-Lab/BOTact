import cv2
import numpy as np
import torch
import torch.nn as nn


class DepthMLP(nn.Module):
    """Depth network copied from depth.py to guarantee architecture consistency."""

    def __init__(self, input_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.model(x)


class DepthEstimator:
    """Depth prediction and depth-to-pointcloud conversion."""

    def __init__(self, device: torch.device, model_path: str, patch_size: int = 7):
        self.device = device
        self.patch_size = patch_size
        self.half_size = patch_size // 2

        self.depth_threshold = 1.6
        self.neighbor_kernel_size = 3
        self.neighbor_min_points = 9
        self.px_to_mm = 0.03
        self.z_scale = 3.0

        input_dim = self.patch_size * self.patch_size
        self.model = DepthMLP(input_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def estimate_depth(self, seg_gray: np.ndarray) -> np.ndarray:
        h, w = seg_gray.shape
        seg_norm = seg_gray.astype(np.float32) / 255.0
        bg_mask = seg_gray > 250

        x = torch.from_numpy(seg_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, padding=self.half_size)
        patches = patches.permute(0, 2, 1).contiguous().view(-1, self.patch_size * self.patch_size)

        preds = []
        batch_size = 8192
        with torch.no_grad():
            for i in range(0, patches.size(0), batch_size):
                batch = patches[i : i + batch_size]
                pred = self.model(batch)
                preds.append(pred.cpu().numpy())

        depth_flat = np.concatenate(preds, axis=0)
        depth_map = depth_flat.reshape(h, w)
        depth_map[bg_mask] = 0
        depth_map[depth_map < 0] = 0
        return depth_map.astype(np.float32)

    def depth_to_pointcloud(self, depth_map: np.ndarray) -> np.ndarray:
        initial_mask = depth_map > self.depth_threshold

        if self.neighbor_kernel_size % 2 == 0:
            raise ValueError("NEIGHBOR_KERNEL_SIZE must be odd.")

        kernel = np.ones((self.neighbor_kernel_size, self.neighbor_kernel_size), np.uint8)
        neighbor_count = cv2.filter2D(initial_mask.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        valid_mask = (neighbor_count >= self.neighbor_min_points) & initial_mask

        v_coords, u_coords = np.where(valid_mask)
        # if len(v_coords) == 0:
        #     raise RuntimeError("No valid point reconstructed. Try changing threshold or lighting.")

        x = u_coords * self.px_to_mm
        y = v_coords * self.px_to_mm
        z = depth_map[v_coords, u_coords] * self.z_scale
        return np.column_stack((x, y, z)).astype(np.float32)

    @staticmethod
    def save_ascii_pcd(points: np.ndarray, save_path: str) -> None:
        num_points = points.shape[0]
        header = (
            "# .PCD v0.7 - Point Cloud Data\n"
            "VERSION 0.7\n"
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
            f"WIDTH {num_points}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {num_points}\n"
            "DATA ascii\n"
        )
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(header)
            np.savetxt(f, points, fmt="%.6f")
