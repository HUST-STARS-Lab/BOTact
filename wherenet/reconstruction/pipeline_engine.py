import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from reconstruction.classifier_vit import build_visual_encoder, SimpleClassifier
from segmenter import Segmenter
from depth_estimator import DepthEstimator
from bufferx_registrar import BufferXRegistrar

from runtime_config import (
    CLASS_NAME_MAPPING,
    MODEL_CLS_PATH,
    MODEL_DEPTH_PATH,
    ensure_dirs,
)

class PipelineEngine:
    """Reusable inference engine to avoid reloading models on each click."""

    PATCH_SIZE = 7
    HALF_SIZE = PATCH_SIZE // 2

    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    PX_TO_MM = 0.03

    DEPTH_THRESHOLD = 1.6
    NEIGHBOR_KERNEL_SIZE = 3
    NEIGHBOR_MIN_POINTS = 9
    Z_SCALE = 3.0

    def __init__(self):
        ensure_dirs()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vit_model = None
        self.classifier = None
        self.idx_to_class = None
        self.transform = None

        self.segmenter = Segmenter(img_width=self.IMG_WIDTH, img_height=self.IMG_HEIGHT)
        self.depth_estimator = None
        self.registrar = None

        self._load_classifier()
        self._load_depth_model()
        self._load_registration_model()

    def _load_classifier(self) -> None:
        vit_model, vit_dim = build_visual_encoder(self.device)
        try:
            checkpoint = torch.load(MODEL_CLS_PATH, map_location=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load classifier checkpoint: {MODEL_CLS_PATH}\n"
                f"Reason: {exc}"
            ) from exc
        num_classes_meta = checkpoint.get("classes", 8)
        num_classes = len(num_classes_meta) if isinstance(num_classes_meta, (list, tuple)) else int(num_classes_meta)

        classifier = SimpleClassifier(vit_dim=vit_dim, num_classes=num_classes).to(self.device)
        vit_model.load_state_dict(checkpoint["vit_model"])
        classifier.load_state_dict(checkpoint["classifier"])
        vit_model.eval()
        classifier.eval()

        self.vit_model = vit_model
        self.classifier = classifier

        if "class_to_idx" in checkpoint:
            self.idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
        else:
            self.idx_to_class = {i: c for i, c in enumerate(["A", "B", "C", "D", "E", "F", "G", "H"])}

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _load_depth_model(self) -> None:
        try:
            self.depth_estimator = DepthEstimator(
                device=self.device,
                model_path=MODEL_DEPTH_PATH,
                patch_size=self.PATCH_SIZE,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load depth checkpoint: {MODEL_DEPTH_PATH}\n"
                f"Reason: {exc}"
            ) from exc

    def _load_registration_model(self) -> None:
        try:
            self.registrar = BufferXRegistrar(device=self.device)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize BufferX registrar:\nReason: {exc}") from exc

    def classify(self, bgr_img: np.ndarray):
        print("Running classification...")
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).convert("RGB")
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.vit_model(img_tensor)
            logits = self.classifier(feat)
            pred_idx = logits.argmax(1).item()
            pred_cls = self.idx_to_class[pred_idx]
            conf = torch.softmax(logits, dim=1)[0, pred_idx].item() * 100.0

        return pred_cls, CLASS_NAME_MAPPING.get(pred_cls, pred_cls), conf

    def segment(self, fg_bgr: np.ndarray, bg_bgr: np.ndarray) -> np.ndarray:
        print("Running segmentation...")
        return self.segmenter.segment(fg_bgr, bg_bgr)

    def estimate_depth(self, seg_gray: np.ndarray) -> np.ndarray:
        print("Running depth estimation...")
        return self.depth_estimator.estimate_depth(seg_gray)

    def depth_to_pointcloud(self, depth_map: np.ndarray) -> np.ndarray:
        print("Running point-cloud reconstruction...")
        return self.depth_estimator.depth_to_pointcloud(depth_map)

    @staticmethod
    def save_ascii_pcd(points: np.ndarray, save_path: str) -> None:
        DepthEstimator.save_ascii_pcd(points, save_path)

    def register(self, template_pcd_path: str, target_pcd_path: str):
        """Run BUFFER-X registration and return transformed source + target clouds."""
        print("Running BUFFER-X registration...")
        return self.registrar.register(template_pcd_path, target_pcd_path)
