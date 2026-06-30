import copy
import os

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn

from config import make_cfg
from models.BUFFERX import BufferX
from runtime_config import PROJECT_ROOT


class BufferXRegistrar:
    """Load BufferX and run registration inference."""

    def __init__(self, device: torch.device):
        self.device = device
        self.cfg = make_cfg("KITTI", root_dir="/tmp")
        self.cfg[self.cfg.data.dataset] = self.cfg.copy()
        self.cfg.stage = "test"
        # Runtime UI needs real inference latency; enable timing in test mode.
        self.cfg.test.enable_timing = True
        self.model = self._load_registration_model()

    def _resolve_stage_ckpt(self, stage: str) -> str:
        candidates = [
            os.path.join(PROJECT_ROOT, "checkpoints", "registration", stage, "best.pth"),
            os.path.join(PROJECT_ROOT, "snapshot", "kitti", stage, "best.pth"),
            os.path.join(PROJECT_ROOT, "snapshot", "threedmatch", stage, "best.pth"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"Registration stage checkpoint not found for stage={stage}.\n"
            f"Searched:\n- " + "\n- ".join(candidates)
        )

    def _load_registration_model(self):
        model = BufferX(self.cfg)
        for stage in self.cfg.train.all_stage:
            stage_ckpt = self._resolve_stage_ckpt(stage)
            state_dict = torch.load(stage_ckpt, map_location=self.device)
            staged = {k: v for k, v in state_dict.items() if stage in k}
            if not staged:
                raise RuntimeError(
                    f"Checkpoint loaded but no parameters matched stage '{stage}': {stage_ckpt}"
                )
            model_state = model.state_dict()
            model_state.update(staged)
            model.load_state_dict(model_state)

        model = model.to(self.device)
        if self.device.type == "cuda":
            model = nn.DataParallel(model, device_ids=[torch.cuda.current_device()])
        model.eval()
        return model

    def _preprocess_for_bufferx(self, src_raw: o3d.geometry.PointCloud, tgt_raw: o3d.geometry.PointCloud):
        num_fps = self.cfg.patch.num_fps
        num_radius_est = self.cfg.patch.num_points_radius_estimate
        min_fds_pts = max(num_fps, num_radius_est) + 1

        voxel_fds = self.cfg.data.downsample
        voxel_sds = self.cfg.data.voxel_size_0

        def process_single(pcd: o3d.geometry.PointCloud):
            pcd_fds = pcd.voxel_down_sample(voxel_size=voxel_fds)
            pts_fds = np.asarray(pcd_fds.points, dtype=np.float32)
            if pts_fds.size == 0:
                raise RuntimeError("Point cloud is empty after first downsample.")
            np.random.shuffle(pts_fds)

            max_fds = 30000
            if pts_fds.shape[0] > max_fds:
                pts_fds = pts_fds[np.random.choice(pts_fds.shape[0], max_fds, replace=False)]
            if pts_fds.shape[0] < min_fds_pts:
                pts_fds = pts_fds[np.random.choice(pts_fds.shape[0], min_fds_pts, replace=True)]

            pcd_sds = pcd_fds.voxel_down_sample(voxel_size=voxel_sds)
            pts_sds = np.asarray(pcd_sds.points, dtype=np.float32)
            if pts_sds.size == 0:
                pts_sds = pts_fds.copy()

            max_pts = self.cfg.data.max_numPts
            if pts_sds.shape[0] > max_pts:
                pts_sds = pts_sds[np.random.choice(pts_sds.shape[0], max_pts, replace=False)]

            return pts_fds, pts_sds

        src_fds, src_sds = process_single(src_raw)
        tgt_fds, tgt_sds = process_single(tgt_raw)

        return {
            "src_fds_pcd": torch.tensor(src_fds, dtype=torch.float32),
            "tgt_fds_pcd": torch.tensor(tgt_fds, dtype=torch.float32),
            "src_sds_pcd": torch.tensor(src_sds, dtype=torch.float32),
            "tgt_sds_pcd": torch.tensor(tgt_sds, dtype=torch.float32),
            "relt_pose": torch.eye(4, dtype=torch.float32),
            "src_id": "live/src",
            "tgt_id": "live/tgt",
            "scene_name": "live",
            "voxel_sizes": torch.tensor([voxel_sds], dtype=torch.float32),
            "dataset_names": [self.cfg.data.dataset],
            "sphericity": torch.tensor([0.0], dtype=torch.float32),
            "is_aligned_to_global_z": False,
        }

    def register(self, template_pcd_path: str, target_pcd_path: str):
        if not os.path.exists(template_pcd_path):
            raise FileNotFoundError(f"Template PCD not found: {template_pcd_path}")
        if not os.path.exists(target_pcd_path):
            raise FileNotFoundError(f"Target PCD not found: {target_pcd_path}")

        src_raw = o3d.io.read_point_cloud(template_pcd_path)
        tgt_raw = o3d.io.read_point_cloud(target_pcd_path)
        if src_raw.is_empty() or tgt_raw.is_empty():
            raise RuntimeError("Input point cloud is empty.")

        data_source = self._preprocess_for_bufferx(src_raw, tgt_raw)

        with torch.no_grad():
            trans_est, times, *_ = self.model(data_source)

        if trans_est is None:
            trans = np.eye(4, dtype=np.float64)
        elif isinstance(trans_est, torch.Tensor):
            trans = trans_est.detach().cpu().numpy().astype(np.float64)
        else:
            trans = np.asarray(trans_est, dtype=np.float64)

        src_transformed = copy.deepcopy(src_raw)
        src_transformed.transform(trans)
        return src_transformed, tgt_raw, trans, times
