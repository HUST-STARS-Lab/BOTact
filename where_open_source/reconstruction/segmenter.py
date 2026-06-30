import cv2
import numpy as np


class Segmenter:
    """Contact region segmentation via weighted background subtraction."""

    def __init__(self, img_width: int = 640, img_height: int = 480):
        self.img_width = img_width
        self.img_height = img_height

    def segment(self, fg_bgr: np.ndarray, bg_bgr: np.ndarray) -> np.ndarray:
        fg = cv2.resize(fg_bgr, (self.img_width, self.img_height))
        bg = cv2.resize(bg_bgr, (self.img_width, self.img_height))

        bg_f = bg.astype(np.float32)
        fg_f = fg.astype(np.float32)

        diff_b = np.abs(bg_f[:, :, 0] - fg_f[:, :, 0])
        diff_g = np.abs(bg_f[:, :, 1] - fg_f[:, :, 1])
        diff_r = np.abs(bg_f[:, :, 2] - fg_f[:, :, 2])

        w_diff = 0.15 * diff_b + 0.50 * diff_g + 0.35 * diff_r
        w_diff = np.clip(w_diff, 0, 255).astype(np.uint8)

        denoised = cv2.medianBlur(w_diff, 5)
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)

        otsu_thresh, _ = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        new_thresh = otsu_thresh * 1.0
        _, mask = cv2.threshold(denoised, new_thresh, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask)
        if contours:
            max_cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(filled_mask, [max_cnt], -1, 255, thickness=-1)

        final_result = np.ones_like(denoised) * 255
        inverted_signal = 255 - denoised
        final_result[filled_mask > 0] = inverted_signal[filled_mask > 0]
        return final_result
