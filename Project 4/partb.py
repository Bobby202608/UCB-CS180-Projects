import argparse
import cv2
import datetime
import json
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
DOWNTOWN_1_PATH = "input/downtown_1.jpeg"
DOWNTOWN_2_PATH = "input/downtown_2.jpeg"

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

    if n_pts <= 0 or not isinstance(n_pts, int):
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


# ==================== Section B.2: Feature Descriptor Extraction ====================

def extract_descriptors(im, coords, window_size=40, patch_size=8):
    """
        Extract normalized, axis-aligned patch descriptors.
    """

    if im.ndim != 2:
        raise ValueError("Error: Input image must be 2D grayscale.")

    if coords.ndim != 2 or coords.shape[0] != 2:
        raise ValueError("Error: Coordinates must have shape (2, n).")

    for size in (window_size, patch_size):
        if not isinstance(size, (int, np.integer)) or size <= 0:
            raise ValueError(
                "Error: window_size and patch_size must be positive integers."
            )

    if patch_size > window_size:
        raise ValueError("Error: patch_size must not exceed window_size.")

    # 1. Gaussian blur the image
    # ksize=(0, 0): automatically determined kernel size
    im_blurred = cv2.GaussianBlur(im, ksize=(0, 0), sigmaX=2.0, sigmaY=2.0)

    h, w = im.shape
    half_size = window_size // 2

    valid_coords = []
    descriptors = []

    for y, x in coords.T:
        y, x = int(y), int(x)

        # 2. Extract a 40*40 window around the corner point
        y_min = y - half_size
        y_max = y_min + window_size
        x_min = x - half_size
        x_max = x_min + window_size

        # Skip out-of-bound windows
        if y_min < 0 or y_max > h or x_min < 0 or x_max > w:
            continue

        window = im_blurred[y_min:y_max, x_min:x_max]

        # 3. Downsample the window to a 8*8 patch
        patch = cv2.resize(window, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

        # 4. Normalize the patch
        mean = patch.mean()
        std = patch.std()
        patch = (patch - mean) / (std + 1e-8)

        valid_coords.append((y, x))  # Shape: (n, 2)
        descriptors.append(patch.reshape(-1))  # Shape: (n, patch_size ** 2)

    valid_coords = np.asarray(valid_coords, dtype=np.int64).reshape(-1, 2).T  # Shape: (2, n)
    descriptors = np.asarray(descriptors, dtype=np.float64).reshape(-1, patch_size ** 2)  # Shape: (n, patch_size ** 2)

    return valid_coords, descriptors


def draw_descriptors(descriptors, patch_size=8, n_show=16, scale=20):
    """
    Build a grid image showing several extracted descriptors.

    Inputs:
        descriptors: Array of shape (m, patch_size ** 2).
        patch_size: Side length used during descriptor extraction.
        n_show: Maximum number of descriptors to display.
        scale: Enlargement factor for each displayed patch.

    Returns (once implemented):
        A uint8 grayscale grid image that can be saved with cv2.imwrite.
    """

    # TODO 1: Select up to n_show descriptors and reshape them into square patches.
    # TODO 2: Map values to [0, 255] for display without changing the descriptors.
    # TODO 3: Enlarge the patches with nearest-neighbor interpolation and arrange
    #         them in a grid; handle an empty descriptor array explicitly.
    raise NotImplementedError("Feature descriptor visualization is not implemented yet.")










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
                (DOWNTOWN_1_PATH, DOWNTOWN_2_PATH),
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

                img_1 = img_1.astype(np.float64)
                img_2 = img_2.astype(np.float64)

                name_1 = os.path.basename(path_1).split(".")[0]
                name_2 = os.path.basename(path_2).split(".")[0]

                print(f"\nProcessing images: {name_1} and {name_2}")

                # Convert the images to grayscale, and normalize them to [0, 1]
                img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                img_1_gray = img_1_gray.astype(np.float64) / 255.0
                img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
                img_2_gray = img_2_gray.astype(np.float64) / 255.0

                # Section B.1: Harris Corner Detection
                if args.section == "1":
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
                    print(f"Number of corners found in {name_2}: {coords_2_anms.shape[1]}\n")

                    img_1_corners_anms = draw_corners(img_1, coords_1_anms)
                    img_2_corners_anms = draw_corners(img_2, coords_2_anms)

                    for name, harris, anms in [
                        (name_1, img_1_corners, img_1_corners_anms),
                        (name_2, img_2_corners, img_2_corners_anms)
                    ]:
                        harris_path = os.path.join(output_dir, f"{name}_harris.png")
                        anms_path = os.path.join(output_dir, f"{name}_anms.png")

                        cv2.imwrite(harris_path, harris)
                        cv2.imwrite(anms_path, anms)

                        print(f"Harris corners without ANMS, marked on {name}, saved to: {harris_path}")
                        print(f"Harris corners with ANMS, marked on {name}, saved to: {anms_path}")

                    # Dump the ANMS corner coordinates to a JSON file
                    for name, image_path, coords in [
                        (name_1, path_1, coords_1_anms),
                        (name_2, path_2, coords_2_anms)
                    ]:
                        json_path = os.path.join(output_dir, f"{name}_anms.json")

                        with open(json_path, "w", encoding="utf-8") as output_file:
                            json.dump(
                                {
                                    "image_path": image_path,
                                    "coordinate_order": ["y", "x"],
                                    "coords": coords.tolist()  # Shape: (2, n)
                                },
                                output_file,
                                indent=2,
                                ensure_ascii=False
                            )

                        print(f"ANMS corner coordinates for {name} saved to: {json_path}")

                # Section B.2: Feature Descriptor Extraction
                elif args.section == "2":
                    # Load the ANMS corner coordinates from the JSON file
                    anms_coords = []

                    for name in [name_1, name_2]:
                        json_path = os.path.join(OUTPUT_DIR_1, f"{name}_anms.json")

                        if not os.path.isfile(json_path):
                            raise FileNotFoundError(f"Error: {json_path} not found. Run --section 1 first.")

                        with open(json_path, "r", encoding="utf-8") as input_file:
                            corner_data = json.load(input_file)

                        if not isinstance(corner_data, dict) or corner_data.get("coordinate_order") != ["y", "x"]:
                            raise ValueError(f"Error: {json_path} must use coordinate_order ['y', 'x'].")

                        coords = np.array(corner_data["coords"], dtype=np.int64)
                        if coords.ndim != 2 or coords.shape[0] != 2:
                            raise ValueError(f"Error: Coordinates in {json_path} must have shape (2, n).")

                        anms_coords.append(coords)

                        print(f"Loaded {coords.shape[1]} ANMS corner coordinates for {name} from: {json_path}")

                    coords_1, coords_2 = anms_coords  # Shape: (2, n)
                    
                    valid_coords_1, descriptors_1 = extract_descriptors(img_1_gray, coords_1)
                    valid_coords_2, descriptors_2 = extract_descriptors(img_2_gray, coords_2)










    else:
        raise argparse.ArgumentError(None, "Error: Invalid section number. Please choose from 1, 2, 3 or 4.")


if __name__ == "__main__":
    main()