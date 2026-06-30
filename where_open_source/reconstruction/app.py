"""Qt5 UI - Tactile Capture and Full Reconstruction System (single-window, three-column layout)

Three-column layout:
    Col-1  Camera A - live RGB preview + ViT classification result
    Col-2  Camera B - depth estimation view + segmentation controls
    Col-3  3D reconstruction / registration rendering + metrics

Dependencies: PyQt5, opencv-python, numpy, open3d, matplotlib, Pillow
"""

from __future__ import annotations

import copy
import sys
import os
import threading
import time
from datetime import datetime
from typing import Optional

os.environ["TORCH_CUDA_ARCH_LIST"] = "6.1"
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from PyQt5.QtCore import Qt, QThread, QTimer, QLibraryInfo, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSizePolicy, QSplitter,
    QTextEdit, QVBoxLayout, QWidget,
)

from runtime_config import (
    CLASS_TO_TEMPLATE_PCD, PCD_DIR, RAW_DIR,
)
from pipeline_engine import PipelineEngine
import camera_stream as cam

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)


# ══════════════════════════════════════════════════════════════════════════════
#  全局样式表
# ══════════════════════════════════════════════════════════════════════════════

_STYLESHEET = """
* {
    font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
    font-size: 13px;
}
QMainWindow, QWidget {
    background: #f0efe8;
    color: #1a1a18;
}

/* ── 卡片 ── */
QFrame#card {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 14px;
}

/* ── 顶栏 ── */
QFrame#topbar {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 12px;
}

/* ── 列标题条 ── */
QFrame#colhead {
    background: #fafaf8;
    border-bottom: 1px solid rgba(0,0,0,0.06);
    border-top-left-radius: 14px;
    border-top-right-radius: 14px;
}

/* ── 相机视图 ── */
QLabel#camview {
    background: #16181f;
    color: rgba(255,255,255,0.25);
    border-radius: 10px;
}
QLabel#camview2 {
    background: #0e1520;
    color: rgba(255,255,255,0.25);
    border-radius: 10px;
}

/* ── 重建结果视图 ── */
QLabel#resultview {
    background: #0a0d14;
    color: rgba(255,255,255,0.2);
    border-radius: 10px;
}

/* ── 日志 ── */
QTextEdit#log {
    background: #f7f6f2;
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 8px;
    color: #444441;
    font-size: 11px;
    padding: 6px 8px;
    selection-background-color: #9FE1CB;
}

/* ── 徽章 ── */
QLabel#tag_green  { color:#0F6E56; background:#E1F5EE; border-radius:10px; padding:2px 10px; font-size:11px; font-weight:500; }
QLabel#tag_blue   { color:#185FA5; background:#E6F1FB; border-radius:10px; padding:2px 10px; font-size:11px; font-weight:500; }
QLabel#tag_purple { color:#534AB7; background:#EEEDFE; border-radius:10px; padding:2px 10px; font-size:11px; font-weight:500; }
QLabel#tag_warn   { color:#854F0B; background:#FAEEDA; border-radius:10px; padding:2px 10px; font-size:11px; font-weight:500; }
QLabel#tag_red    { color:#A32D2D; background:#FCEBEB; border-radius:10px; padding:2px 10px; font-size:11px; font-weight:500; }

/* ── 文字类 ── */
QLabel#sectiontitle { font-size:12px; font-weight:500; color:#1a1a18; }
QLabel#mutedlabel   { font-size:11px; color:#888780; }

/* ── 分割线 ── */
QFrame#divider { background:rgba(0,0,0,0.06); max-height:1px; border:none; }

/* ── 按钮基底 ── */
QPushButton {
    border: 1px solid rgba(0,0,0,0.12);
    border-radius: 8px;
    padding: 7px 16px;
    background: #ffffff;
    color: #1a1a18;
    font-size: 12px;
}
QPushButton:hover   { background:#f0efea; }
QPushButton:pressed { background:#e6e5e0; }
QPushButton:disabled { color:rgba(0,0,0,0.22); background:#ebebea; border-color:transparent; }

QPushButton#btnPrimary {
    background:#1D9E75; color:#fff; border:none; font-weight:500;
}
QPushButton#btnPrimary:hover   { background:#0F6E56; }
QPushButton#btnPrimary:pressed { background:#085041; }
QPushButton#btnPrimary:disabled { background:rgba(29,158,117,0.32); color:rgba(255,255,255,0.5); }

QPushButton#btnWarn {
    background:#EF9F27; color:#fff; border:none; font-weight:500;
}
QPushButton#btnWarn:hover   { background:#BA7517; }
QPushButton#btnWarn:pressed { background:#854F0B; }

QPushButton#btnDanger {
    color:#A32D2D; border-color:rgba(163,45,45,0.25);
}
QPushButton#btnDanger:hover   { background:#FCEBEB; }
QPushButton#btnDanger:pressed { background:#F7C1C1; }

"""


# ══════════════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════════════

def ndarray_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    """将 numpy 图像数组转换为可在 Qt QLabel 上显示的 QPixmap。"""
    arr = np.ascontiguousarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w, ch = arr.shape
    fmt = QImage.Format_RGB888 if ch == 3 else QImage.Format_RGBA8888
    return QPixmap.fromImage(QImage(arr.tobytes(), w, h, ch * w, fmt).copy())


def fit_pixmap(px: QPixmap, max_w: int, max_h: int) -> QPixmap:
    """按比例缩放图片，适配目标区域并保持平滑显示。"""
    return px.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def divider() -> QFrame:
    """创建分割线控件。"""
    f = QFrame()
    f.setObjectName("divider")
    f.setFixedHeight(1)
    return f


def badge(text: str, kind: str = "green", parent: QWidget | None = None) -> QLabel:
    """创建状态徽章标签（支持不同颜色主题）。"""
    lbl = QLabel(text, parent)
    lbl.setObjectName({"green": "tag_green", "blue": "tag_blue",
                        "purple": "tag_purple", "warn": "tag_warn",
                        "red": "tag_red"}.get(kind, "tag_green"))
    lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return lbl


def metric_row(layout: QVBoxLayout, key: str, val: str) -> QLabel:
    """在指标区域添加一行“键值对”并返回值控件便于后续更新。"""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    k = QLabel(key)
    k.setObjectName("mutedlabel")
    v = QLabel(val)
    v.setStyleSheet("font-size:12px; font-weight:500;")
    row.addWidget(k)
    row.addStretch()
    row.addWidget(v)
    layout.addLayout(row)
    return v


def set_badge_kind(lbl: QLabel, text: str, kind: str):
    """动态更换徽章颜色并刷新样式。"""
    lbl.setText(text)
    lbl.setObjectName({"green": "tag_green", "blue": "tag_blue",
                        "purple": "tag_purple", "warn": "tag_warn",
                        "red": "tag_red"}.get(kind, "tag_green"))
    lbl.setStyle(lbl.style())


# ══════════════════════════════════════════════════════════════════════════════
#  顶部状态栏
# ══════════════════════════════════════════════════════════════════════════════

class TopBar(QFrame):
    def __init__(self, parent: QWidget | None = None):
        """构建顶部状态栏，显示系统标题与运行状态。"""
        super().__init__(parent)
        self.setObjectName("topbar")
        self.setFixedHeight(50)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 16, 0)

        dot = QLabel()
        dot.setFixedSize(9, 9)
        dot.setStyleSheet("background:#1D9E75; border-radius:5px;")

        col = QVBoxLayout()
        col.setSpacing(1)
        t1 = QLabel("Visual Perception · 3D Reconstruction System")
        t1.setStyleSheet("font-size:14px; font-weight:500; color:#1a1a18;")
        t2 = QLabel("Dual Cameras  ·  Real-Time Inference  ·  Three-Column View")
        t2.setStyleSheet("font-size:11px; color:#888780;")
        col.addWidget(t1)
        col.addWidget(t2)

        lay.addWidget(dot)
        lay.addSpacing(8)
        lay.addLayout(col)
        lay.addStretch()

        self.lbl_status = badge("● Running", "green")
        self.lbl_gpu    = badge("GPU --", "warn")
        self.lbl_stage  = badge("Stage-1", "blue")
        for w in (self.lbl_status, self.lbl_gpu, self.lbl_stage):
            lay.addWidget(w)
            lay.addSpacing(4)


# ══════════════════════════════════════════════════════════════════════════════
#  工作线程
# ══════════════════════════════════════════════════════════════════════════════

class Stage1Worker(QThread):
    """Stage-1 后台线程：抓拍并执行分类推理。"""
    finished = pyqtSignal(str, str, str, float)
    error    = pyqtSignal(str)

    def __init__(self, engine: PipelineEngine, frame: np.ndarray):
        """注入推理引擎与待分类帧。"""
        super().__init__()
        self.engine = engine
        self.frame  = frame

    def run(self):
        """执行分类流程并通过信号把结果回传主线程。"""
        try:
            stamp    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            raw_path = os.path.join(RAW_DIR, f"stage1_{stamp}.png")
            cv2.imwrite(raw_path, self.frame)
            pred_cls, pred_name, conf = self.engine.classify(self.frame)
            summary = (
                f"[Stage-1  {stamp}]\n"
                f"Classification complete: {pred_name}  (class={pred_cls},  confidence={conf:.2f}%)\n"
                f"Saved: {raw_path}"
            )
            self.finished.emit(summary, pred_cls, pred_name, conf)
        except Exception as exc:
            self.error.emit(str(exc))


class Stage2Worker(QThread):
    """Stage-2 后台线程：实时重建、节流配准和性能统计。"""
    log_line   = pyqtSignal(str)
    render_pcd = pyqtSignal(object, object, object)
    empty_plot = pyqtSignal(str)
    pose_time  = pyqtSignal(float)
    pts_count  = pyqtSignal(int)
    fps_update = pyqtSignal(float, float)
    finished   = pyqtSignal(str)
    error      = pyqtSignal(str)

    def __init__(self, engine: PipelineEngine,
                 bg: np.ndarray,
                 pred_cls: str, pred_name: str):
        """注入推理引擎、背景帧与 Stage-1 分类结果。"""
        super().__init__()
        self.engine    = engine
        self._bg_lock  = threading.Lock()
        self.bg        = bg.copy()
        self.pred_cls  = pred_cls
        self.pred_name = pred_name

    def update_background(self, bg: np.ndarray):
        """在重建运行中热更新背景帧，使新背景立即生效。"""
        with self._bg_lock:
            self.bg = bg.copy()

    def run(self):
        """循环执行实时处理：分割 -> 深度 -> 点云 -> 条件配准。"""
        try:
            # 输出文件路径与运行参数初始化。
            pcd_path = os.path.join(PCD_DIR, "stage2_realtime_latest.pcd")
            template_path = CLASS_TO_TEMPLATE_PCD.get(self.pred_cls)
            frame_count = 0
            min_pts_for_registration = 1000

            # 若当前类别无模板点云，直接标记为跳过配准。
            if template_path is None:
                self.log_line.emit("Registration skipped: no template point cloud configured for this class.")
                self.pose_time.emit(-1.0)
                self.empty_plot.emit("No template point cloud for this class. Registration skipped.")

            self.log_line.emit("Realtime reconstruction started.")

            # 主循环：直到外部请求中断线程。
            while not self.isInterruptionRequested():
                t0 = time.perf_counter()
                fg = cam.current_frame_stage2
                if fg is None:
                    self.msleep(30)
                    continue

                # 执行实时分割、深度估计与点云重建。
                frame_count += 1
                fg = fg.copy()
                with self._bg_lock:
                    bg = self.bg.copy()
                seg = self.engine.segment(fg, bg)
                depth_map = self.engine.estimate_depth(seg)
                points = self.engine.depth_to_pointcloud(depth_map)
                pts_num = len(points)

                # 保存最新重建点云，并通知 UI 更新点数。
                self.engine.save_ascii_pcd(points, pcd_path)
                self.pts_count.emit(pts_num)

                # 统计重建 FPS。
                recon_dt = max(time.perf_counter() - t0, 1e-6)
                recon_fps = 1.0 / recon_dt
                reg_fps = -1.0

                # 点云点数不足时跳过配准，不刷新配准结果视图。
                if (
                    template_path is not None
                    and pts_num >= min_pts_for_registration
                    and frame_count % 2 == 0
                ):
                    treg = time.perf_counter()
                    src_tf, tgt, _trans, times = self.engine.register(template_path, pcd_path)
                    self.render_pcd.emit(src_tf, tgt, _trans)
                    pose_t = 0.0
                    if isinstance(times, (list, tuple)):
                        pose_t = float(times[1]) + float(times[2]) if len(times) > 2 else (
                            float(times[1]) if len(times) > 1 else 0.0)
                    self.pose_time.emit(pose_t)
                    # 统计配准 FPS。
                    reg_dt = max(time.perf_counter() - treg, 1e-6)
                    reg_fps = 1.0 / reg_dt

                # 将实时性能指标回传到界面。
                self.fps_update.emit(recon_fps, reg_fps)

                # 控制日志频率，避免刷屏。
                if frame_count == 1 or frame_count % 10 == 0:
                    if reg_fps >= 0:
                        self.log_line.emit(
                            f"Realtime frame {frame_count}: {pts_num} pts, recon_fps={recon_fps:.2f}, reg_fps={reg_fps:.2f}"
                        )
                    else:
                        if pts_num < min_pts_for_registration:
                            self.log_line.emit(
                                f"Realtime frame {frame_count}: {pts_num} pts, recon_fps={recon_fps:.2f}, reg=skip(points<{min_pts_for_registration})"
                            )
                        else:
                            self.log_line.emit(
                                f"Realtime frame {frame_count}: {pts_num} pts, recon_fps={recon_fps:.2f}, reg=skip(this frame)"
                            )

                # 适当休眠，给 UI 刷新与相机线程留出调度时间。
                self.msleep(30)

            # 线程退出时给出总结日志。
            self.finished.emit(
                f"[Stage-2 Realtime Stopped]\n"
                f"Class (from Stage-1): {self.pred_name} ({self.pred_cls})\n"
                f"Processed frames: {frame_count}\n"
                f"Latest point cloud: {pcd_path}"
            )
        except Exception as exc:
            # 任意异常都通过 error 信号交给主线程处理。
            self.error.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════════════
#  三列子面板
# ══════════════════════════════════════════════════════════════════════════════

class ColHeader(QFrame):
    """Column header: index badge + title + right-side status badge."""
    def __init__(self, number: str, title: str, tag_text: str,
                 tag_kind: str = "blue", parent: QWidget | None = None):
        """创建三列通用标题头部。"""
        super().__init__(parent)
        self.setObjectName("colhead")
        self.setFixedHeight(44)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 14, 0)

        num = QLabel(number)
        num.setFixedSize(22, 22)
        num.setAlignment(Qt.AlignCenter)
        num.setStyleSheet(
            "background:#1a1a18; color:#fff; border-radius:11px;"
            "font-size:11px; font-weight:500;"
        )
        t = QLabel(title)
        t.setStyleSheet("font-size:13px; font-weight:500; color:#1a1a18;")
        self.tag = badge(tag_text, tag_kind)

        lay.addWidget(num)
        lay.addSpacing(8)
        lay.addWidget(t)
        lay.addStretch()
        lay.addWidget(self.tag)


# ── Col-1: 分类视图 ───────────────────────────────────────────────────────────

class ClassifyColumn(QFrame):
    """第一列：相机A预览与分类结果展示。"""
    CAM_W = 400
    CAM_H = 300

    def __init__(self, parent: QWidget | None = None):
        """构建分类列 UI 组件。"""
        super().__init__(parent)
        self.setObjectName("card")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = ColHeader("1", "Classification", "Pending", "blue")
        root.addWidget(self.header)
        root.addWidget(divider())

        body = QVBoxLayout()
        body.setContentsMargins(14, 12, 14, 14)
        body.setSpacing(10)

        # 相机标题行
        cam_h = QHBoxLayout()
        lbl_cam = QLabel("Camera A")
        lbl_cam.setObjectName("sectiontitle")
        self.fps_lbl = QLabel("30 fps")
        self.fps_lbl.setObjectName("mutedlabel")
        cam_h.addWidget(lbl_cam)
        cam_h.addStretch()
        cam_h.addWidget(self.fps_lbl)
        body.addLayout(cam_h)

        # 相机画面
        self.lbl_camera = QLabel("Opening camera...")
        self.lbl_camera.setObjectName("camview")
        self.lbl_camera.setFixedSize(self.CAM_W, self.CAM_H)
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        body.addWidget(self.lbl_camera)

        # 分类结果徽章
        badge_row = QHBoxLayout()
        badge_row.setSpacing(6)
        self.badge_cls  = badge("Class -", "blue")
        self.badge_conf = badge("Confidence -", "green")
        badge_row.addWidget(self.badge_cls)
        badge_row.addWidget(self.badge_conf)
        badge_row.addStretch()
        body.addLayout(badge_row)

        body.addWidget(divider())

        sec = QLabel("Classification Result")
        sec.setObjectName("sectiontitle")
        body.addWidget(sec)

        self.m_name  = metric_row(body, "Target Name", "-")
        self.m_cls   = metric_row(body, "Class ID", "-")
        self.m_conf2 = metric_row(body, "Confidence", "-")
        self.m_name.setStyleSheet("font-size:20px; font-weight:700; color:#0F6E56;")
        self.m_cls.setStyleSheet("font-size:16px; font-weight:600; color:#1a1a18;")
        self.m_conf2.setStyleSheet("font-size:16px; font-weight:600; color:#1a1a18;")

        body.addWidget(divider())

        self.btn_capture = QPushButton("▶  Capture and Classify")
        self.btn_capture.setObjectName("btnPrimary")
        self.btn_capture.setFixedHeight(38)
        body.addWidget(self.btn_capture)

        body.addStretch()

        log_lbl = QLabel("Log")
        log_lbl.setObjectName("sectiontitle")
        body.addWidget(log_lbl)
        self.txt_log = QTextEdit()
        self.txt_log.setObjectName("log")
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(110)
        body.addWidget(self.txt_log)

        root.addLayout(body)

    def update_camera(self, rgb: np.ndarray):
        """刷新相机A画面。"""
        px = ndarray_to_qpixmap(rgb)
        self.lbl_camera.setPixmap(fit_pixmap(px, self.CAM_W, self.CAM_H))
        self.lbl_camera.setText("")

    def set_result(self, name: str, cls: str, conf: float):
        """更新分类结果徽章与指标。"""
        self.badge_cls.setText(name)
        self.badge_conf.setText(f"Confidence {conf:.1f}%")
        self.m_name.setText(name)
        self.m_cls.setText(cls)
        self.m_conf2.setText(f"{conf:.2f}%")
        set_badge_kind(self.header.tag, "Classified ✓", "green")

    def log(self, text: str):
        """追加日志到第一列日志窗。"""
        self.txt_log.append(text)


# ── Col-2: 深度视图 ───────────────────────────────────────────────────────────

class DepthColumn(QFrame):
    """第二列：相机B预览、背景控制与重建指标。"""
    CAM_W = 400
    CAM_H = 300

    def __init__(self, parent: QWidget | None = None):
        """构建深度列 UI 组件。"""
        super().__init__(parent)
        self.setObjectName("card")
        self.background_frame: Optional[np.ndarray] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = ColHeader("2", "Depth View", "Pending", "purple")
        root.addWidget(self.header)
        root.addWidget(divider())

        body = QVBoxLayout()
        body.setContentsMargins(14, 12, 14, 14)
        body.setSpacing(10)

        cam_h = QHBoxLayout()
        lbl_cam = QLabel("Camera B")
        lbl_cam.setObjectName("sectiontitle")
        self.fps_lbl = QLabel("30 fps")
        self.fps_lbl.setObjectName("mutedlabel")
        cam_h.addWidget(lbl_cam)
        cam_h.addStretch()
        cam_h.addWidget(self.fps_lbl)
        body.addLayout(cam_h)

        self.lbl_camera = QLabel("Waiting for camera feed...")
        self.lbl_camera.setObjectName("camview2")
        self.lbl_camera.setFixedSize(self.CAM_W, self.CAM_H)
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        body.addWidget(self.lbl_camera)

        badge_row = QHBoxLayout()
        badge_row.setSpacing(6)
        self.badge_bg  = badge("Background Unlocked", "warn")
        self.badge_seg = badge("Not Segmented", "purple")
        badge_row.addWidget(self.badge_bg)
        badge_row.addWidget(self.badge_seg)
        badge_row.addStretch()
        body.addLayout(badge_row)

        body.addWidget(divider())

        sec = QLabel("Depth Metrics")
        sec.setObjectName("sectiontitle")
        body.addWidget(sec)

        self.m_depth = metric_row(body, "Depth Status", "-")
        self.m_pts   = metric_row(body, "Point Count", "-")
        self.m_fps   = metric_row(body, "Recon FPS", "-")

        body.addWidget(divider())

        self.btn_reset_bg = QPushButton("Reset Background Frame")
        self.btn_reset_bg.setObjectName("btnWarn")
        self.btn_reset_bg.setFixedHeight(34)
        body.addWidget(self.btn_reset_bg)

        body.addStretch()

        log_lbl = QLabel("Log")
        log_lbl.setObjectName("sectiontitle")
        body.addWidget(log_lbl)
        self.txt_log = QTextEdit()
        self.txt_log.setObjectName("log")
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(110)
        body.addWidget(self.txt_log)

        root.addLayout(body)

    def update_camera(self, rgb: np.ndarray):
        """刷新相机B画面，并在首次可用时锁定背景帧。"""
        px = ndarray_to_qpixmap(rgb)
        self.lbl_camera.setPixmap(fit_pixmap(px, self.CAM_W, self.CAM_H))
        self.lbl_camera.setText("")
        if self.background_frame is None and cam.current_frame_stage2 is not None:
            self.background_frame = cam.current_frame_stage2.copy()
            set_badge_kind(self.badge_bg, "Background Locked", "green")

    def reset_background(self) -> bool:
        """将当前相机B画面重新设为背景帧。"""
        if cam.current_frame_stage2 is None:
            return False
        self.background_frame = cam.current_frame_stage2.copy()
        set_badge_kind(self.badge_bg, "Background Reset", "green")
        self.log("Background frame reset to current view.")
        return True

    def set_processing(self, on: bool):
        """设置第二列处于实时处理中状态。"""
        if on:
            set_badge_kind(self.header.tag, "Realtime", "blue")
            set_badge_kind(self.badge_seg,  "Realtime Running",  "blue")
            self.m_depth.setText("Streaming")
            self.m_fps.setText("-")

    def set_stopping(self):
        """设置第二列处于停止中的过渡状态。"""
        set_badge_kind(self.header.tag, "Stopping...", "warn")
        set_badge_kind(self.badge_seg, "Stopping...", "warn")

    def set_live_pts(self, pts: int):
        """实时更新点数指标。"""
        self.m_depth.setText("Streaming")
        self.m_pts.setText(f"{pts:,} pts")

    def set_live_fps(self, fps: float):
        """实时更新重建 FPS 指标。"""
        self.m_fps.setText(f"{fps:.2f} Hz" if fps > 0 else "-")

    def set_done(self, pts: int):
        """Stage-2 完成后更新最终状态。"""
        set_badge_kind(self.header.tag, "Done ✓", "green")
        set_badge_kind(self.badge_seg,  "Segmented",  "green")
        self.m_depth.setText("Valid")
        self.m_pts.setText(f"{pts:,} pts")

    def log(self, text: str):
        """追加日志到第二列日志窗。"""
        self.txt_log.append(text)


# ── Col-3: 重建结果视图 ───────────────────────────────────────────────────────

class ReconColumn(QFrame):
    """第三列：点云配准可视化与配准指标展示。"""
    RES_W = 400
    RES_H = 300

    def __init__(self, parent: QWidget | None = None):
        """构建重建结果列 UI 组件。"""
        super().__init__(parent)
        self.setObjectName("card")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = ColHeader("3", "3D Reconstruction", "Waiting", "purple")
        root.addWidget(self.header)
        root.addWidget(divider())

        body = QVBoxLayout()
        body.setContentsMargins(14, 12, 14, 14)
        body.setSpacing(10)

        res_h = QHBoxLayout()
        lbl_res = QLabel("Point Cloud Registration")
        lbl_res.setObjectName("sectiontitle")
        self.badge_pts = badge("— pts", "purple")
        res_h.addWidget(lbl_res)
        res_h.addStretch()
        res_h.addWidget(self.badge_pts)
        body.addLayout(res_h)

        self.lbl_result = QLabel("Registration result will appear here")
        self.lbl_result.setObjectName("resultview")
        self.lbl_result.setFixedSize(self.RES_W, self.RES_H)
        self.lbl_result.setAlignment(Qt.AlignCenter)
        body.addWidget(self.lbl_result)

        # 图例
        legend = QHBoxLayout()
        legend.setSpacing(18)
        for color, text in (("#E24B4A", "Reconstructed cloud (source)"),
                     ("#378ADD", "Template cloud (target)")):
            row = QHBoxLayout()
            row.setSpacing(5)
            dot = QLabel()
            dot.setFixedSize(9, 9)
            dot.setStyleSheet(f"background:{color}; border-radius:5px;")
            lbl = QLabel(text)
            lbl.setObjectName("mutedlabel")
            row.addWidget(dot)
            row.addWidget(lbl)
            legend.addLayout(row)
        legend.addStretch()
        body.addLayout(legend)

        body.addWidget(divider())

        sec = QLabel("Registration Metrics")
        sec.setObjectName("sectiontitle")
        body.addWidget(sec)

        self.m_class  = metric_row(body, "Target Class",  "-")
        self.m_pts    = metric_row(body, "Total Points", "-")
        self.m_time   = metric_row(body, "Registration Time",   "-")
        self.m_reg_fps = metric_row(body, "Registration FPS", "-")
        self.m_status = metric_row(body, "Registration Status",   "-")

        body.addWidget(divider())
        rot_sec = QLabel("Rotation Matrix")
        rot_sec.setObjectName("sectiontitle")
        body.addWidget(rot_sec)
        self.m_rot = QLabel("-")
        self.m_rot.setStyleSheet(
            "font-family:'Consolas','Menlo','Monospace';"
            "font-size:16px; font-weight:600; color:#0F6E56;"
            "background:#EAF7F2; border:1px solid rgba(15,110,86,0.25);"
            "border-radius:8px; padding:8px;"
        )
        self.m_rot.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.m_rot.setWordWrap(False)
        body.addWidget(self.m_rot)

        body.addStretch()

        log_lbl = QLabel("Log")
        log_lbl.setObjectName("sectiontitle")
        body.addWidget(log_lbl)
        self.txt_log = QTextEdit()
        self.txt_log.setObjectName("log")
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(110)
        body.addWidget(self.txt_log)

        root.addLayout(body)

    def show_result(self, px: QPixmap):
        """显示最新配准渲染图。"""
        self.lbl_result.setPixmap(fit_pixmap(px, self.RES_W, self.RES_H))
        self.lbl_result.setText("")

    def set_metrics(self, name: str, cls: str, total_pts: int, pose_t: float):
        """更新第三列配准相关指标。"""
        self.m_class.setText(f"{name}  ({cls})")
        self.m_pts.setText(f"{total_pts:,} pts")
        if pose_t >= 0:
            self.m_time.setText(f"{pose_t:.4f} s")
        self.badge_pts.setText(f"{total_pts:,} pts")

    def set_done(self):
        """设置配准完成状态。"""
        set_badge_kind(self.header.tag, "Done ✓", "green")
        self.m_status.setText("Completed")

    def set_skipped(self):
        """设置配准跳过状态。"""
        set_badge_kind(self.header.tag, "Skipped", "warn")
        self.m_status.setText("No template, skipped")

    def set_rotation_matrix(self, trans: Optional[np.ndarray]):
        """更新旋转矩阵显示区域。"""
        if trans is None:
            self.m_rot.setText("-")
            return
        arr = np.asarray(trans, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3:
            self.m_rot.setText("-")
            return
        rot = arr[:3, :3]
        text = "\n".join(" ".join(f"{v: .4f}" for v in row) for row in rot)
        self.m_rot.setText(text)

    def log(self, text: str):
        """追加日志到第三列日志窗。"""
        self.txt_log.append(text)


# ══════════════════════════════════════════════════════════════════════════════
#  主窗口
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """主窗口：负责 UI 组织、线程调度与三列联动。"""
    def __init__(self):
        """初始化窗口、状态变量与定时刷新。"""
        super().__init__()
        self.setWindowTitle("Tactile Capture · Visual Reconstruction System")
        self.resize(1440, 780)

        self.engine: Optional[PipelineEngine] = None
        self.s1_worker: Optional[Stage1Worker] = None
        self.s2_worker: Optional[Stage2Worker] = None

        self.selected_cls:  Optional[str]   = None
        self.selected_name: Optional[str]   = None
        self.selected_conf: Optional[float] = None
        self._last_pts: int = 0
        self._last_pose_t: float = -1.0
        self._pending_auto_stage2: bool = False

        self._build_ui()
        self._start_init()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_frames)
        self.timer.start(30)

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        """搭建主界面布局并连接基础按钮信号。"""
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        self.topbar = TopBar()
        root.addWidget(self.topbar)

        # 三列 QSplitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)
        splitter.setStyleSheet(
            "QSplitter::handle { background:#e6e5df; border-radius:4px; }"
        )

        self.col1 = ClassifyColumn()
        self.col2 = DepthColumn()
        self.col3 = ReconColumn()
        splitter.addWidget(self.col1)
        splitter.addWidget(self.col2)
        splitter.addWidget(self.col3)
        splitter.setSizes([480, 480, 480])
        root.addWidget(splitter, stretch=1)

        # 底部工具条
        bar = QHBoxLayout()
        guide = QLabel(
            "Workflow: complete classification in Column 1  →  run reconstruction in Column 2  →  view results in Column 3"
        )
        guide.setObjectName("mutedlabel")
        btn_clear = QPushButton("Clear All Logs")
        btn_clear.setObjectName("btnDanger")
        btn_clear.setFixedHeight(30)
        btn_clear.clicked.connect(self._clear_all_logs)
        btn_close = QPushButton("Close App")
        btn_close.setObjectName("btnWarn")
        btn_close.setFixedHeight(30)
        btn_close.clicked.connect(self.close)
        bar.addWidget(guide)
        bar.addStretch()
        bar.addWidget(btn_clear)
        bar.addWidget(btn_close)
        root.addLayout(bar)

        # 信号连接
        self.col1.btn_capture.clicked.connect(self._start_stage1)
        self.col2.btn_reset_bg.clicked.connect(self._reset_background)

    # ── 初始化 ───────────────────────────────────────────────────────────────

    def _start_init(self):
        """异步启动模型初始化，避免阻塞 UI。"""
        self.col1.log("Loading models, please wait...")
        threading.Thread(target=self._init_engine, daemon=True).start()

    def _init_engine(self):
        """加载推理引擎并启动相机采集线程。"""
        try:
            self.engine = PipelineEngine()
            QTimer.singleShot(0, lambda: self.col1.log("Model initialization complete."))
            QTimer.singleShot(0, lambda: self.col1.btn_capture.setEnabled(True))
            cam.start_camera_threads()
        except Exception as exc:
            QTimer.singleShot(0, lambda: self.col1.log(f"[Error] {exc}"))
            QTimer.singleShot(0, lambda: QMessageBox.critical(
                self, "Error", f"Model initialization failed:\n{exc}"))

    # ── 帧刷新 ───────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _refresh_frames(self):
        """定时读取最新相机帧并刷新两列预览图。"""
        if cam.current_frame_stage1 is not None:
            rgb = cv2.cvtColor(cam.current_frame_stage1, cv2.COLOR_BGR2RGB)
            self.col1.update_camera(rgb)
        if cam.current_frame_stage2 is not None:
            rgb2 = cv2.cvtColor(cam.current_frame_stage2, cv2.COLOR_BGR2RGB)
            self.col2.update_camera(rgb2)

    # ── 工具 ─────────────────────────────────────────────────────────────────

    def _clear_all_logs(self):
        """清空三列日志窗口。"""
        for col in (self.col1, self.col2, self.col3):
            col.txt_log.clear()

    @pyqtSlot()
    def _reset_background(self):
        """触发背景重置，失败时弹出提示。"""
        if not self.col2.reset_background():
            QMessageBox.warning(self, "Warning", "No camera frame available. Cannot reset background.")
            return
        if self.s2_worker is not None and self.s2_worker.isRunning() and self.col2.background_frame is not None:
            self.s2_worker.update_background(self.col2.background_frame)
            self.col2.log("Background update applied to running reconstruction.")

    # ── Stage-1 ──────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _start_stage1(self):
        """启动 Stage-1 分类线程。"""
        if self.engine is None:
            QMessageBox.warning(self, "Notice", "Model is not fully loaded yet."); return
        if self.s1_worker is not None and self.s1_worker.isRunning():
            QMessageBox.information(self, "Notice", "Classification is in progress. Please wait."); return
        if cam.current_frame_stage1 is None:
            QMessageBox.warning(self, "Warning", "No camera frame captured."); return

        frame = cam.current_frame_stage1.copy()
        self.col1.btn_capture.setEnabled(False)
        self.col1.btn_capture.setText("Classifying...")

        self.s1_worker = Stage1Worker(self.engine, frame)
        self.s1_worker.finished.connect(self._on_stage1_done)
        self.s1_worker.error.connect(self._on_stage1_error)
        self.s1_worker.start()

    @pyqtSlot(str, str, str, float)
    def _on_stage1_done(self, summary: str, cls: str, name: str, conf: float):
        """接收分类结果并自动触发 Stage-2 重建。"""
        self.selected_cls  = cls
        self.selected_name = name
        self.selected_conf = conf

        self.col1.log(summary + "\n")
        self.col1.set_result(name, cls, conf)
        self.col1.btn_capture.setEnabled(True)
        self.col1.btn_capture.setText("▶  Capture and Classify")

        self.col2.log(f"Classification received: {name} ({cls},  {conf:.2f}%)  <- Auto-start realtime reconstruction")

        self._pending_auto_stage2 = True
        if self.s2_worker is not None and self.s2_worker.isRunning():
            self.col2.log("Stage-2 is running. Stopping current session and restarting with latest class...")
            self.s2_worker.requestInterruption()
            self.col2.set_stopping()
        else:
            QTimer.singleShot(0, self._start_stage2)

        set_badge_kind(self.topbar.lbl_stage, "Stage-1 ✓", "green")

    @pyqtSlot(str)
    def _on_stage1_error(self, msg: str):
        """处理 Stage-1 失败状态并恢复按钮。"""
        self.col1.log(f"[Error] {msg}")
        QMessageBox.critical(self, "Classification Failed", msg)
        self.col1.btn_capture.setEnabled(True)
        self.col1.btn_capture.setText("▶  Capture and Classify")

    # ── Stage-2 ──────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _start_stage2(self):
        """启动 Stage-2 实时重建线程。"""
        if self.selected_cls is None:
            QMessageBox.warning(self, "Notice", "Please complete classification in Column 1 first."); return
        if self.s2_worker is not None and self.s2_worker.isRunning():
            self.col2.log("Stop requested: waiting for realtime worker to exit...")
            self.s2_worker.requestInterruption()
            self.col2.set_stopping()
            return
        if cam.current_frame_stage2 is None:
            QMessageBox.warning(self, "Warning", "No camera frame captured."); return
        if self.col2.background_frame is None:
            QMessageBox.warning(self, "Warning", "Background frame not initialized. Please try again later."); return

        bg = self.col2.background_frame.copy()
        self._last_pts = 0
        self._last_pose_t = -1.0
        self.col3.m_reg_fps.setText("-")
        self.col3.set_rotation_matrix(None)

        self.col2.set_processing(True)
        self.col3.log(f"Starting realtime reconstruction and registration...  class={self.selected_name}")
        set_badge_kind(self.topbar.lbl_stage, "Stage-2 Realtime", "blue")

        self.s2_worker = Stage2Worker(
            self.engine, bg, self.selected_cls, self.selected_name)
        self.s2_worker.log_line.connect(self.col2.log)
        self.s2_worker.log_line.connect(self.col3.log)
        self.s2_worker.render_pcd.connect(self._render_registration)
        self.s2_worker.empty_plot.connect(self._show_empty_plot)
        self.s2_worker.pose_time.connect(self._update_pose_time)
        self.s2_worker.pts_count.connect(self._update_pts)
        self.s2_worker.fps_update.connect(self._update_fps)
        self.s2_worker.finished.connect(self._on_stage2_done)
        self.s2_worker.error.connect(self._on_stage2_error)
        self.s2_worker.start()

    @pyqtSlot(int)
    def _update_pts(self, pts: int):
        """更新实时点数到第二列。"""
        self._last_pts = pts
        self.col2.set_live_pts(pts)

    @pyqtSlot(float)
    def _update_pose_time(self, sec: float):
        """更新配准耗时到第三列。"""
        self._last_pose_t = sec
        self.col3.m_time.setText(f"{sec:.4f} s" if sec >= 0 else "-")

    @pyqtSlot(float, float)
    def _update_fps(self, recon_fps: float, reg_fps: float):
        """更新重建与配准 FPS 指标。"""
        self.col2.set_live_fps(recon_fps)
        if reg_fps > 0:
            self.col3.m_reg_fps.setText(f"{reg_fps:.2f} Hz")

    @pyqtSlot(object, object, object)
    def _render_registration(self, src_tf: o3d.geometry.PointCloud,
                              tgt: o3d.geometry.PointCloud,
                              trans: np.ndarray):
        """使用 Open3D 离屏渲染配准结果并显示到第三列。"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="BUFFER-X", width=960, height=720, visible=False)

        src_vis = copy.deepcopy(src_tf)
        tgt_vis = copy.deepcopy(tgt)
        src_vis.paint_uniform_color([1.0, 0.0, 0.0])
        tgt_vis.paint_uniform_color([0.0, 0.0, 1.0])

        plane = o3d.geometry.TriangleMesh.create_box(width=20.0, height=15.0, depth=0.01)
        plane.paint_uniform_color([0.4, 0.4, 0.4])
        plane.translate([0.0, 0.0, 5.5])

        vis.add_geometry(src_vis)
        vis.add_geometry(tgt_vis)
        vis.add_geometry(plane)

        # 设置固定视角，保证每帧显示风格一致。
        ctr = vis.get_view_control()
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_lookat([10.0, 7.5, 0.0])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_zoom(0.2)

        # 捕获离屏渲染结果并转为 Qt 可显示图像。
        vis.poll_events()
        vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True), dtype=np.float32)
        vis.destroy_window()

        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        self.col3.show_result(ndarray_to_qpixmap(img))

        # 更新总点数与配准指标。
        n_src = len(np.asarray(src_tf.points))
        n_tgt = len(np.asarray(tgt.points))
        self.col3.set_metrics(
            self.selected_name, self.selected_cls,
            n_src + n_tgt, self._last_pose_t)
        self.col3.set_rotation_matrix(trans)

    @pyqtSlot(str)
    def _show_empty_plot(self, text: str):
        """显示“跳过配准”的占位图。"""
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center",
                fontsize=11, color="#888780")
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        plt.close(fig)
        self.col3.show_result(ndarray_to_qpixmap(buf[:, :, :3]))
        self.col3.set_skipped()
        self.col3.set_rotation_matrix(None)

    @pyqtSlot(str)
    def _on_stage2_done(self, summary: str):
        """处理 Stage-2 正常结束，并在需要时自动重启。"""
        self.col3.log("\n" + summary + "\n")
        if self._last_pose_t >= 0:
            self.col3.set_done()
        else:
            self.col3.set_skipped()
        self.col2.set_done(self._last_pts)
        self.col2.set_processing(False)
        set_badge_kind(self.topbar.lbl_stage, "Stage-2 Realtime ✓", "green")

        if self._pending_auto_stage2:
            self._pending_auto_stage2 = False
            QTimer.singleShot(0, self._start_stage2)

    @pyqtSlot(str)
    def _on_stage2_error(self, msg: str):
        """处理 Stage-2 异常并清理自动重启标记。"""
        self.col2.log(f"[Error] {msg}")
        self.col3.log(f"[Error] {msg}")
        QMessageBox.critical(self, "Reconstruction Failed", msg)
        self.col2.set_processing(False)
        self._pending_auto_stage2 = False

    # ── 关闭 ─────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        """窗口关闭时停止定时器与相机线程，确保安全退出。"""
        self.timer.stop()
        cam.stop_cameras()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(_STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())