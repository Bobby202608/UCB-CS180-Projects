# =============== Logging ===============

import math
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Iterator, TextIO


class _Tee:
    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()


@contextmanager
def print_logs(log_path: str | Path) -> Iterator[None]:
    """
        Copy standard output to a .log file while preserving terminal output.
    """

    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        with redirect_stdout(_Tee(sys.stdout, log_file)):
            yield


# =============== Image Alignment and Hybrid Images ===============

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import skimage.transform as sktr


def get_points(img_1: np.ndarray, img_2: np.ndarray) -> tuple:
    backend = plt.get_backend().lower()
    if backend == "agg" or "inline" in backend:
        raise RuntimeError(
            "Image alignment requires an interactive Matplotlib backend. "
            "Run this section from a GUI session and select the points manually."
        )

    print("\nLeft-click two corresponding points on each image, in the same order.")
    print("The image switches automatically after two points. Close the window to cancel.")
    fig, ax = plt.subplots()
    points = []
    try:
        for index, img in enumerate((img_1, img_2), start=1):
            ax.clear()
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(f"Image {index}/2: left-click two matching points in the same order")
            fig.canvas.draw()
            while True:
                selected = fig.ginput(2, timeout=0)
                if not plt.fignum_exists(fig.number):
                    raise RuntimeError(f"Error: Alignment cancelled while selecting points in image {index}.")
                if len(selected) != 2:
                    print(f"Image {index}: received {len(selected)} point(s); please select both points again.")
                    continue
                if selected[0] == selected[1]:
                    print(f"Image {index}: the two points must be distinct; please select both again.")
                    continue
                points.extend(selected)
                break
    finally:
        plt.close(fig)
    return tuple(points)


def recenter(img: np.ndarray, r: float, c: float) -> np.ndarray:
    rows, cols = img.shape[:2]
    rpad = int(np.abs(2 * r + 1 - rows))
    cpad = int(np.abs(2 * c + 1 - cols))
    pad_width = [
        (0 if r > (rows - 1) / 2 else rpad, 0 if r < (rows - 1) / 2 else rpad),
        (0 if c > (cols - 1) / 2 else cpad, 0 if c < (cols - 1) / 2 else cpad),
    ]
    if img.ndim == 3:
        pad_width.append((0, 0))
    return np.pad(img, pad_width, "constant")


def find_centers(p1: tuple, p2: tuple) -> tuple:
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy


def align_image_centers(img_1: np.ndarray, img_2: np.ndarray, pts: tuple) -> tuple:
    p1, p2, p3, p4 = pts
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)
    return recenter(img_1, cy1, cx1), recenter(img_2, cy2, cx2)


def rescale_images(img_1: np.ndarray, img_2: np.ndarray, pts: tuple) -> tuple:
    p1, p2, p3, p4 = pts
    len1 = np.hypot(p2[1] - p1[1], p2[0] - p1[0])
    len2 = np.hypot(p4[1] - p3[1], p4[0] - p3[0])
    if len1 == 0 or len2 == 0:
        raise ValueError("Error: The two alignment points in each image must be distinct.")
    dscale = len2 / len1
    channel_axis = -1 if img_1.ndim == 3 else None
    if dscale < 1:
        img_1 = sktr.rescale(img_1, dscale, channel_axis=channel_axis)
    else:
        img_2 = sktr.rescale(img_2, 1.0 / dscale, channel_axis=channel_axis)
    return img_1, img_2


def rotate_im1(img_1: np.ndarray, pts: tuple) -> tuple:
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), p2[0] - p1[0])
    theta2 = math.atan2(-(p4[1] - p3[1]), p4[0] - p3[0])
    dtheta = theta2 - theta1
    return sktr.rotate(img_1, dtheta * 180 / np.pi), dtheta


def match_img_size(img_1: np.ndarray, img_2: np.ndarray) -> tuple:
    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]
    if h1 < h2:
        img_2 = img_2[int(np.floor((h2 - h1) / 2.0)):-int(np.ceil((h2 - h1) / 2.0)), :]
    elif h1 > h2:
        img_1 = img_1[int(np.floor((h1 - h2) / 2.0)):-int(np.ceil((h1 - h2) / 2.0)), :]
    if w1 < w2:
        img_2 = img_2[:, int(np.floor((w2 - w1) / 2.0)):-int(np.ceil((w2 - w1) / 2.0))]
    elif w1 > w2:
        img_1 = img_1[:, int(np.floor((w1 - w2) / 2.0)):-int(np.ceil((w1 - w2) / 2.0))]
    assert img_1.shape == img_2.shape
    return img_1, img_2


def align_images(img_1: np.ndarray, img_2: np.ndarray, points: tuple | None = None) -> tuple:
    pts = get_points(img_1, img_2) if points is None else points
    img_1, img_2 = align_image_centers(img_1, img_2, pts)
    img_1, img_2 = rescale_images(img_1, img_2, pts)
    img_1, _ = rotate_im1(img_1, pts)
    return match_img_size(img_1, img_2)


def hybrid_image(
    img_1: np.ndarray,
    img_2: np.ndarray,
    sigma_high: float,
    sigma_low: float,
    points: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Align inputs in [0, 1], then combine image 1's high and image 2's low frequencies.

        Returns img_1_aligned, img_2_aligned, img_1_high, img_2_low, hybrid.
        img_1_high is signed; only the final hybrid is clipped to [0, 1].
    """

    if sigma_high <= 0 or sigma_low <= 0:
        raise ValueError("Error: Gaussian cutoff frequencies must be positive.")

    # 1. Align the images
    img_1_aligned, img_2_aligned = align_images(img_1, img_2, points)

    # 2a. Extract low-frequency component of image 2
    freq_low = (sigma_low, sigma_low, 0) if img_2_aligned.ndim == 3 else sigma_low
    img_2_low = scipy.ndimage.gaussian_filter(img_2_aligned, freq_low)

    # 2b. Extract high-frequency component of image 1
    freq_high = (sigma_high, sigma_high, 0) if img_1_aligned.ndim == 3 else sigma_high
    img_1_high = img_1_aligned - scipy.ndimage.gaussian_filter(img_1_aligned, freq_high)

    # 3. Add the two images
    hybrid = np.clip(img_2_low + img_1_high, 0.0, 1.0)
    
    return img_1_aligned, img_2_aligned, img_1_high, img_2_low, hybrid
