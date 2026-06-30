import threading
import time

import cv2

from runtime_config import CAMERA_ID_STAGE1, CAMERA_ID_STAGE2


cap_stage1 = None
cap_stage2 = None
is_camera_running = False
current_frame_stage1 = None
current_frame_stage2 = None


def _open_camera_by_id(camera_id: int):
    """Open a specific camera ID with backend fallback."""
    for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
        try:
            cap = cv2.VideoCapture(camera_id, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    print(f"Camera opened: id={camera_id}, backend={backend}")
                    return cap
            cap.release()
        except Exception as exc:
            print(f"Open camera failed id={camera_id}: {exc}")
    return None


def camera_loop_stage1() -> None:
    """Continuously read stage-1 camera frames."""
    global cap_stage1, is_camera_running, current_frame_stage1
    cap_stage1 = _open_camera_by_id(CAMERA_ID_STAGE1)
    if cap_stage1 is None:
        print(f"No available stage-1 camera (id={CAMERA_ID_STAGE1}).")
        return

    while is_camera_running:
        if not cap_stage1.isOpened():
            break
        ret, frame = cap_stage1.read()
        if ret:
            current_frame_stage1 = frame.copy()
        time.sleep(0.03)

    if cap_stage1 is not None:
        cap_stage1.release()
        cap_stage1 = None


def camera_loop_stage2() -> None:
    """Continuously read stage-2 camera frames."""
    global cap_stage2, is_camera_running, current_frame_stage2
    cap_stage2 = _open_camera_by_id(CAMERA_ID_STAGE2)
    if cap_stage2 is None:
        print(f"No available stage-2 camera (id={CAMERA_ID_STAGE2}).")
        return

    while is_camera_running:
        if not cap_stage2.isOpened():
            break
        ret, frame = cap_stage2.read()
        if ret:
            current_frame_stage2 = frame.copy()
        time.sleep(0.03)

    if cap_stage2 is not None:
        cap_stage2.release()
        cap_stage2 = None


def start_camera_threads() -> None:
    global is_camera_running
    is_camera_running = True
    threading.Thread(target=camera_loop_stage1, daemon=True).start()
    threading.Thread(target=camera_loop_stage2, daemon=True).start()


def stop_cameras() -> None:
    global is_camera_running
    is_camera_running = False
    time.sleep(0.1)
