import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

CLASS_NAME_MAPPING = {
    "A": "M3*0.35mm screw",
    "B": "M4*0.5mm screw",
    "C": "M6*0.75mm screw",
    "D": "M8*1mm screw",
    "E": "M2 nut",
    "F": "M3 nut",
    "G": "M4 nut",
    "H": "M6 nut",
}


CLASS_TO_TEMPLATE_PCD = {
    "A": os.path.join(PROJECT_ROOT, "data", "complete", "m3", "GB_M3X4.pcd"),
    "B": os.path.join(PROJECT_ROOT, "data", "complete", "m4", "GB_M4X6.pcd"),
    "C": os.path.join(PROJECT_ROOT, "data", "complete", "m6", "GB_M6X8.pcd"),
    "D": os.path.join(PROJECT_ROOT, "data", "complete", "m8", "GB_M8X16.pcd"),
    "E": None,
    "F": None,
    "G": None,
    "H": None,
}


MODEL_CLS_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "vit_best_8classes.pth")
MODEL_DEPTH_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "depth_mlp_7x7.pth")
LIVE_ROOT = os.path.join(PROJECT_ROOT, "data", "live")
RAW_DIR = os.path.join(LIVE_ROOT, "raw")
PRESEG_DIR = os.path.join(LIVE_ROOT, "presegmented")
DEPTH_DIR = os.path.join(LIVE_ROOT, "depth")
PCD_DIR = os.path.join(LIVE_ROOT, "pcd")


CAMERA_ID_STAGE1 = 10
CAMERA_ID_STAGE2 = 0


def ensure_dirs() -> None:
    """Create runtime output directories once."""
    for d in (RAW_DIR, PRESEG_DIR, DEPTH_DIR, PCD_DIR):
        os.makedirs(d, exist_ok=True)
