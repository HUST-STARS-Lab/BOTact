import torch
import torch.nn as nn


class FusionDecoder(nn.Module):
    """Multi-modal fusion decoder.

    - Point features: pooled PointMamba sequence output
    - Image features: RGB ViT output (3 views concatenated)
    - Depth features: depth ViT output (3 views concatenated)

    Vision and depth features are concatenated inside the decoder.
    """

    def __init__(
        self,
        trans_dim: int = 384,
        img_dim: int = 2304,
        depth_dim: int = 2304,
        cls_dim: int = 9,
        point_pool: str = "avg",
        normalize: bool = True,
        dropout_prob: float = 0.5,
        feat_dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.trans_dim = trans_dim
        self.img_dim = img_dim
        self.depth_dim = depth_dim
        self.cls_dim = cls_dim
        self.point_pool = point_pool
        self.normalize = normalize

        if self.normalize:
            self.point_norm = nn.LayerNorm(self.trans_dim)
            self.img_norm = nn.LayerNorm(self.img_dim)
            self.depth_norm = nn.LayerNorm(self.depth_dim)
        else:
            self.point_norm = nn.Identity()
            self.img_norm = nn.Identity()
            self.depth_norm = nn.Identity()

        self.p_drop = nn.Dropout(p=feat_dropout_prob)
        self.img_drop = nn.Dropout(p=feat_dropout_prob)
        self.depth_drop = nn.Dropout(p=feat_dropout_prob)
        self.feat_drop = nn.Dropout(p=feat_dropout_prob)

        input_dim = self.trans_dim + self.img_dim + self.depth_dim

        self.cls_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(32, self.cls_dim),
        )

    def _pool_point_feat(self, x_point: torch.Tensor) -> torch.Tensor:
        """Pool point features [B, L, C] -> [B, C]."""
        if x_point.dim() == 3:
            _, _, c = x_point.shape
            if c != self.trans_dim:
                raise ValueError(f"Point feature dim mismatch: got {c}, expected {self.trans_dim}")

            if self.point_pool == "avg":
                return x_point.mean(dim=1)
            if self.point_pool == "max":
                return x_point.max(dim=1)[0]
            raise ValueError(f"Unsupported point_pool: {self.point_pool}")

        if x_point.dim() == 2:
            if x_point.shape[1] != self.trans_dim:
                raise ValueError(
                    f"Point feature dim mismatch: got {x_point.shape[1]}, expected {self.trans_dim}"
                )
            return x_point

        raise ValueError("x_point must be [B, L, C] or [B, C]")

    def _check_img_feat(self, x_img: torch.Tensor) -> torch.Tensor:
        """Validate RGB ViT features."""
        if x_img.dim() != 2 or x_img.size(1) != self.img_dim:
            raise ValueError(
                f"Image feature must be [B, {self.img_dim}], got {list(x_img.shape)}"
            )
        return x_img

    def _check_depth_feat(self, x_depth: torch.Tensor) -> torch.Tensor:
        """Validate depth ViT features."""
        if x_depth.dim() != 2 or x_depth.size(1) != self.depth_dim:
            raise ValueError(
                f"Depth feature must be [B, {self.depth_dim}], got {list(x_depth.shape)}"
            )
        return x_depth

    def forward(
        self,
        x_point: torch.Tensor,
        x_img: torch.Tensor,
        x_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_point: [B, L, trans_dim] or [B, trans_dim]
            x_img:   [B, img_dim], RGB ViT features (3 views concatenated)
            x_depth: [B, depth_dim], depth ViT features (3 views concatenated)
        """
        p = self._pool_point_feat(x_point)
        p = self.p_drop(self.point_norm(p))

        img = self.img_drop(self.img_norm(self._check_img_feat(x_img)))
        depth = self.depth_drop(self.depth_norm(self._check_depth_feat(x_depth)))

        feat = self.feat_drop(torch.cat([p, img, depth], dim=1))
        return self.cls_head(feat)
