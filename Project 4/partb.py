import argparse
import cv2
import datetime
import math
import numpy as np
import os
import sys
import time

from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from skimage.feature import corner_harris, peak_local_max
from typing import Iterator, TextIO


# ==================== Configuration Center (EDITABLE) ====================

# 1. Input image paths
TARGET_1_PATH = "input/target_1.jpeg"
TARGET_2_PATH = "input/target_2.jpeg"

GATEWAY_1_PATH = "input/gateway_1.jpeg"
GATEWAY_2_PATH = "input/gateway_2.jpeg"

# 2. Output directories
OUTPUT_DIR_1 = "output/partb_1/"
OUTPUT_DIR_2 = "output/partb_2/"
OUTPUT_DIR_3 = "output/partb_3/"
OUTPUT_DIR_4 = "output/partb_4/"

# ==================== Configuration Center Ends ====================


# ==================== Help Function for Logging ====================

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


# ==================== Section B.1: Harris Corner Detection ====================

def get_harris_corners(im, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 20 pixels), and finds all Harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # Find Harris corners
    # H: Harris matrix, a 2x2 matrix of Gaussian-smoothed image gradient outer products.
    # h: response function, h = 2 * det(H) / (tr(H) + ε)
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=1)

    # Discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

    ndata, dimx = x.shape  # Shape: (M, N)
    ncenters, dimc = c.shape  # Shape: (L, N)
    assert dimx == dimc, 'Data dimension does not match dimension of centers'

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)  # Shape: (M, L)


def ANMS(h, coords, n_pts=500, c_robust=0.9):
    """
    Perform Adaptive Non-Maximal Suppression (ANMS) to select strong, spatially well-
    distributed corners by retaining those farthest from significantly stronger corners.
    """

    if n_pts <= 0:
        raise ValueError("Error: n_pts must be a positive integer.")
    if not 0 < c_robust < 1:
        raise ValueError("Error: c_robust must be between 0 and 1.")

    n = coords.shape[1]  # Shape of coords: (2, n)
    if n == 0:
        raise ValueError("Error: No Harris corners found in the image.")

    responses = h[coords[0], coords[1]]
    points = coords.T.astype(np.float64)  # (n, 2)

    # Suppression radii: the minimum distance to a "stronger" corner
    suppress_r = np.full(n, np.inf)

    for i in range(n):
        # 1. Find "stronger" corners
        stronger = responses[i] < c_robust * responses
        stronger[i] = False

        # 2. Find the minimum distance
        if np.any(stronger):
            distances = dist2(points[i:i+1], points[stronger])
            suppress_r[i] = np.maximum(distances, 0).min()

    # Sort the Harris corners by suppression radius (then response function) in descending order
    idx_order = np.lexsort((-responses, -suppress_r))
    idx_chosen = idx_order[:min(n_pts, n)]

    return coords[:, idx_chosen]


def draw_corners(im, coords):
    """
        Draws corners on an image.
    """

    result = im.copy()

    for y, x in coords.T:
        cv2.circle(
            result,
            (int(x), int(y)),
            radius=3,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA
        )

    return result



















def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        required=True,
        choices=["1", "2", "3", "4"],
        help="Which section to run"
    )
    args = parser.parse_args()

    if args.section in ["1", "2", "3", "4"]:
        if args.section == "1":
            print("Running Section B.1: Harris Corner Detection")
            output_dir = OUTPUT_DIR_1

        elif args.section == "2":
            print("Running Section B.2: Feature Descriptor Extraction")
            output_dir = OUTPUT_DIR_2

        elif args.section == "3":
            print("Running Section B.3: Feature Matching")
            output_dir = OUTPUT_DIR_3

        elif args.section == "4":
            print("Running Section B.4: RANSAC for Robust Homography")
            output_dir = OUTPUT_DIR_4

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with print_logs(os.path.join(output_dir, f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")):
            for path_1, path_2 in [
                (TARGET_1_PATH, TARGET_2_PATH),
                (GATEWAY_1_PATH, GATEWAY_2_PATH)
            ]:
                if not os.path.exists(path_1):
                    raise FileNotFoundError(f"Error: {path_1} not found.")
                if not os.path.exists(path_2):
                    raise FileNotFoundError(f"Error: {path_2} not found.")

                img_1 = cv2.imread(path_1)
                img_2 = cv2.imread(path_2)

                if img_1 is None:
                    raise FileNotFoundError(f"Error: Failed to read {path_1}.")
                if img_2 is None:
                    raise FileNotFoundError(f"Error: Failed to read {path_2}.")

                name_1 = os.path.basename(path_1).split(".")[0]
                name_2 = os.path.basename(path_2).split(".")[0]

                print(f"\nProcessing images: {name_1} and {name_2}")

                # Convert the images to grayscale, and normalize them to [0, 1]
                img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                img_1_gray = img_1_gray.astype(np.float64) / 255.0
                img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
                img_2_gray = img_2_gray.astype(np.float64) / 255.0

                # a. Harris corners without ANMS
                print("\na. Finding Harris corners without ANMS...")

                start_time = time.time()
                h_1, coords_1 = get_harris_corners(img_1_gray)
                h_2, coords_2 = get_harris_corners(img_2_gray)
                time_elapsed = time.time() - start_time

                print(f"Time elapsed: {time_elapsed:.4f} s")
                print(f"Number of corners found in {name_1}: {coords_1.shape[1]}")
                print(f"Number of corners found in {name_2}: {coords_2.shape[1]}")

                img_1_corners = draw_corners(img_1, coords_1)
                img_2_corners = draw_corners(img_2, coords_2)

                # b. Harris corners with ANMS
                print("\nb. Finding Harris corners with ANMS...")

                start_time = time.time()
                coords_1_anms = ANMS(h_1, coords_1)
                coords_2_anms = ANMS(h_2, coords_2)
                time_elapsed = time.time() - start_time

                print(f"Time elapsed: {time_elapsed:.4f} s")
                print(f"Number of corners found in {name_1}: {coords_1_anms.shape[1]}")
                print(f"Number of corners found in {name_2}: {coords_2_anms.shape[1]}")

                img_1_corners_anms = draw_corners(img_1, coords_1_anms)
                img_2_corners_anms = draw_corners(img_2, coords_2_anms)

                for name, harris, anms in [
                    (name_1, img_1_corners, img_1_corners_anms),
                    (name_2, img_2_corners, img_2_corners_anms)
                ]:
                    cv2.imwrite(os.path.join(output_dir, f"{name}_harris.png"), harris)
                    cv2.imwrite(os.path.join(output_dir, f"{name}_anms.png"), anms)

            print(f"\nSaved Harris corners to: {output_dir}")

    else:
        raise argparse.ArgumentError(None, "Error: Invalid section number. Please choose from 1, 2, 3 or 4.")


if __name__ == "__main__":
    main()