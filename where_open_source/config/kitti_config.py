from .outdoor_config import OutdoorBaseConfig
from pathlib import Path


class KITTIConfig(OutdoorBaseConfig):
    def __init__(self, root_dir=Path("../datasets")):
        super().__init__()
        self._C.data.dataset = "KITTI"
        self._C.data.root = root_dir / "kitti"
        self._C.test.pdist = 10

        self._C.train.pretrain_model = ""
        self._C.train.all_stage = ["Desc", "Pose"]

        self._C.test.experiment_id = "threedmatch"

        # Gaussian-Wasserstein-Mahalanobis inference parameters
        # Score: S_ij = alpha * D_desc + beta * W2 + gamma * d_M
        self._C.match.alpha_desc = 1.0
        self._C.match.beta_wasserstein = 1.0
        self._C.match.gamma_mahalanobis = 1.0
        self._C.match.topk_correspondences = 512

        # Mahalanobis inlier search after pose estimation
        self._C.match.use_mahalanobis_inlier = True
        # Chi-square threshold for 3 DoF at ~95%
        self._C.match.mahalanobis_inlier_th = 7.815


def make_cfg(root_dir):
    return KITTIConfig(root_dir).get_cfg()
