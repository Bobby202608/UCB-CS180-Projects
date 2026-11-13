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

def get_harris_corners(im, edge_discard=20, threshold=10.0):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 20 pixels), and finds all Harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    Only local maxima with a Harris response strictly above threshold
    (an absolute threshold, default 10) are retained.
    """

    assert edge_discard >= 20

    # Find Harris corners
    # H: Harris matrix, a 2x2 matrix of Gaussian-smoothed image gradient outer products.
    # h: response function, h = 2 * det(H) / (tr(H) + ε)
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=1)  # Shape: (n, 2)

    # Keep only corners whose response exceeds the threshold
    coords = coords[h[coords[:, 0], coords[:, 1]] > threshold]

    # Discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T  # Shape: (2, n)
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

    if h.shape[0] == 0 or h.shape[1] == 0:  # Shape of h: (h, w)
        raise ValueError("Error: Harris matrix is empty.")

    cnt = coords.shape[1]  # Shape of coords: (2, n)
    if cnt == 0:
        raise ValueError("Error: No Harris corners found in the image.")

    if n_pts <= 0 or not isinstance(n_pts, int):
        raise ValueError("Error: n_pts must be a positive integer.")

    if not 0 < c_robust < 1:
        raise ValueError("Error: c_robust must be between 0 and 1.")

    responses = h[coords[0], coords[1]]
    points = coords.T.astype(np.float64)  # (n, 2)

    # Suppression radii: the minimum distance to a "stronger" corner
    suppress_r = np.full(cnt, np.inf)

    for i in range(cnt):
        # 1. Find "stronger" corners
        stronger = responses[i] < c_robust * responses
        stronger[i] = False

        # 2. Find the minimum distance
        if np.any(stronger):
            distances = dist2(points[i:i+1], points[stronger])
            suppress_r[i] = np.maximum(distances, 0).min()

    # Sort the Harris corners by suppression radius (then response function) in descending order
    idx_order = np.lexsort((-responses, -suppress_r))
    idx_chosen = idx_order[:min(n_pts, cnt)]

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

    if im.shape[0] == 0 or im.shape[1] == 0:
        raise ValueError("Error: Input image is empty.")

    if coords.shape[1] == 0:
        raise ValueError("Error: No coordinates provided for descriptor extraction.")

    for size in (window_size, patch_size):
        if not isinstance(size, (int, np.integer)) or size <= 0:
            raise ValueError("Error: window_size and patch_size must be positive integers.")

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


def draw_descriptors(descriptors, patch_size=8, n_show=10, scale=20):
    """
    Build a grid image showing several extracted descriptors.
    """

    if descriptors.shape[0] == 0:
        raise ValueError("Error: No descriptors to draw.")

    if not isinstance(patch_size, (int, np.integer)) or patch_size <= 0:
        raise ValueError("Error: patch_size must be a positive integer.")

    if not isinstance(n_show, (int, np.integer)) or n_show < 0:
        raise ValueError("Error: n_show must be a non-negative integer.")

    if not isinstance(scale, (int, np.integer)) or scale <= 0:
        raise ValueError("Error: scale must be a positive integer.")

    cnt = min(n_show, descriptors.shape[0])
    cols = int(np.ceil(np.sqrt(cnt)))
    rows = int(np.ceil(cnt / cols))
    tile_size = patch_size * scale  # 8*20 = 160 by default

    # Add 1 pixel of white padding between each row/column of tiles
    result = np.zeros((rows*tile_size + rows - 1, cols*tile_size + cols - 1), dtype=np.uint8)
    result.fill(255)

    for i in range(cnt):
        patch = descriptors[i].reshape(patch_size, patch_size)
        patch = np.clip(patch * 255, 0, 255).astype(np.uint8)
        patch = cv2.resize(patch, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)

        row = i // cols
        col = i % cols

        result[row*tile_size + row : row*tile_size + row + tile_size,
               col*tile_size + col : col*tile_size + col + tile_size] = patch

    return result


# ==================== Section B.3: Feature Matching ====================

def match_descriptors(desc_1, desc_2, ratio_thresh=0.7):
    """
    Match descriptors with a nearest-neighbor distance ratio test.
    """

    desc_1 = np.asarray(desc_1, dtype=np.float64)  # Shape: (n1, patch_size**2)
    desc_2 = np.asarray(desc_2, dtype=np.float64)  # Shape: (n2, patch_size**2)

    if desc_1.shape[0] == 0:
        raise ValueError("Error: No descriptors in desc_1 to match.")
    if desc_2.shape[0] == 0:
        raise ValueError("Error: No descriptors in desc_2 to match.")

    if desc_2.shape[0] < 2:
        raise ValueError("Error: At least 2 descriptors are required in desc_2 for the ratio test.")

    if desc_1.shape[1] != desc_2.shape[1]:
        raise ValueError("Error: The descriptors' shapes (patch_size**2) must match.")

    if not 0 <= ratio_thresh <= 1:
        raise ValueError("Error: ratio_thresh must be between 0 and 1.")

    matches = []

    for i, descriptor in enumerate(desc_1):
        distances = np.linalg.norm(desc_2 - descriptor, axis=1)

        # best_1: 1st nearest neighbor, best_2: 2nd nearest neighbor
        best_1, best_2 = np.argsort(distances)[:2]

        if distances[best_1] < ratio_thresh * distances[best_2]:
            matches.append((i, best_1))

    matches = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
    return matches


def draw_matches(img_1, img_2, coords_1, coords_2, matches):
    """
    Draw matched points and connecting lines on two side-by-side BGR images.
    """

    if img_1.ndim != 3 or img_2.ndim != 3:
        raise ValueError("Error: Both images must be 3D BGR.")

    if img_1.shape[0] == 0 or img_1.shape[1] == 0:
        raise ValueError("Error: Input img_1 is empty.")
    if img_2.shape[0] == 0 or img_2.shape[1] == 0:
        raise ValueError("Error: Input img_2 is empty.")

    coords_1 = np.asarray(coords_1)
    coords_2 = np.asarray(coords_2)
    matches = np.asarray(matches)

    h_1, w_1 = img_1.shape[:2]
    h_2, w_2 = img_2.shape[:2]

    # Put the two images on a shared canvas
    # Shared canvas' shape: (max(h_1, h_2), w_1 + w_2), channels: 3
    result = np.zeros((max(h_1, h_2), w_1 + w_2, 3), dtype=np.uint8)
    result[:h_1, :w_1] = img_1
    result[:h_2, w_1:] = img_2

    # Generator for random colors
    rng = np.random.default_rng(2026)

    for i, j in matches:
        y_1, x_1 = coords_1[:, i]
        y_2, x_2 = coords_2[:, j]

        point_1 = (int(x_1), int(y_1))
        point_2 = (int(x_2) + w_1, int(y_2))

        color = tuple(int(value) for value in rng.integers(64, 256, size=3))

        cv2.line(result, point_1, point_2, color, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(result, point_1, radius=4, color=color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(result, point_2, radius=4, color=color, thickness=2, lineType=cv2.LINE_AA)

    return result


# ==================== Section B.4: RANSAC for Robust Homography ====================
















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

                name_1 = os.path.basename(path_1).split(".")[0]
                name_2 = os.path.basename(path_2).split(".")[0]

                print(f"\nProcessing images: {name_1} and {name_2}")

                # Convert the images to grayscale, and normalize them to [0, 1]
                img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                img_1_gray = img_1_gray.astype(np.float64) / 255.0
                img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
                img_2_gray = img_2_gray.astype(np.float64) / 255.0

                # ==================== Section B.1: Harris Corner Detection ====================

                if args.section == "1":
                    # a. Harris corners without ANMS
                    print("\na. Finding Harris corners without ANMS...")

                    start_time = time.time()
                    h_1, coords_1 = get_harris_corners(img_1_gray, threshold=0.5)
                    h_2, coords_2 = get_harris_corners(img_2_gray, threshold=0.5)
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

                    # Save the ANMS corner coordinates to a JSON file
                    for name, img_path, coords in [
                        (name_1, path_1, coords_1_anms),
                        (name_2, path_2, coords_2_anms)
                    ]:
                        json_path = os.path.join(output_dir, f"{name}_anms.json")

                        with open(json_path, "w", encoding="utf-8") as output_file:
                            json.dump(
                                {
                                    "image_path": img_path,
                                    "coordinate_order": ["y", "x"],
                                    "coords": coords.tolist()  # Shape: (2, n)
                                },
                                output_file,
                                indent=2,
                                ensure_ascii=False
                            )

                        print(f"ANMS corner coordinates for {name} saved to: {json_path}")


                # ==================== Section B.2: Feature Descriptor Extraction ====================

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

                        print(f"\nLoaded {coords.shape[1]} ANMS corner coordinates for {name} from: {json_path}")

                    coords_1, coords_2 = anms_coords  # Shape: (2, n)

                    # Extract feature descriptors
                    valid_coords_1, desc_1 = extract_descriptors(img_1_gray, coords_1)
                    valid_coords_2, desc_2 = extract_descriptors(img_2_gray, coords_2)
                    print(f"\nExtracted {desc_1.shape[0]} feature descriptors for {name_1}")
                    print(f"Extracted {desc_2.shape[0]} feature descriptors for {name_2}")

                    # Draw feature descriptors
                    desc_1_drawn = draw_descriptors(desc_1, n_show=100)
                    desc_2_drawn = draw_descriptors(desc_2, n_show=100)

                    for name, img_path, valid_coords, descriptors, drawn in [
                        (name_1, path_1, valid_coords_1, desc_1, desc_1_drawn),
                        (name_2, path_2, valid_coords_2, desc_2, desc_2_drawn)
                    ]:
                        # a. Save the valid coordinates to a JSON file
                        coords_path = os.path.join(output_dir, f"{name}_descriptor_coords.json")
                        with open(coords_path, "w", encoding="utf-8") as output_file:
                            json.dump(
                                {
                                    "image_path": img_path,
                                    "coordinate_order": ["y", "x"],
                                    "coords": valid_coords.tolist()
                                },
                                output_file,
                                indent=2,
                                ensure_ascii=False
                            )
                        print(f"\nValid descriptor coordinates for {name} saved to: {coords_path}")

                        # b. Save the descriptors to a .npy file
                        desc_path = os.path.join(output_dir, f"{name}_descriptors.npy")
                        np.save(desc_path, descriptors)
                        print(f"Descriptors for {name} saved to: {desc_path}")

                        # c. Save the visualized descriptors
                        drawn_path = os.path.join(output_dir, f"{name}_descriptors.png")
                        cv2.imwrite(drawn_path, drawn)
                        print(f"Descriptor visualization for {name} saved to: {drawn_path}")


                # ==================== Section B.3: Feature Matching ====================

                elif args.section == "3":
                    coordinates = []
                    descriptors = []

                    for name in [name_1, name_2]:
                        coords_path = os.path.join(OUTPUT_DIR_2, f"{name}_descriptor_coords.json")
                        desc_path = os.path.join(OUTPUT_DIR_2, f"{name}_descriptors.npy")

                        for input_path in [coords_path, desc_path]:
                            if not os.path.isfile(input_path):
                                raise FileNotFoundError(f"Error: {input_path} not found. Run --section 2 first.")

                        # Load the valid coordinates
                        with open(coords_path, "r", encoding="utf-8") as input_file:
                            corner_data = json.load(input_file)

                        if not isinstance(corner_data, dict) or corner_data.get("coordinate_order") != ["y", "x"]:
                            raise ValueError(f"Error: {coords_path} must use coordinate_order ['y', 'x'].")

                        coords = np.array(corner_data["coords"], dtype=np.int64)
                        if coords.ndim != 2 or coords.shape[0] != 2:
                            raise ValueError(f"Error: Coordinates in {coords_path} must have shape (2, n).")

                        # Load the descriptors
                        descs = np.load(desc_path, allow_pickle=False)

                        if descs.ndim != 2:
                            raise ValueError(f"Error: Descriptors in {desc_path} must have shape (n, patch_size**2).")

                        coordinates.append(coords)
                        descriptors.append(descs)

                        print(f"\nLoaded {coords.shape[1]} valid descriptor coordinates for {name} from: {coords_path}")
                        print(f"Loaded {descs.shape[0]} feature descriptors for {name} from: {desc_path}")

                    coords_1, coords_2 = coordinates  # Shape: (2, n)
                    desc_1, desc_2 = descriptors  # Shape: (n, patch_size**2)

                    # Match the descriptors
                    matches = match_descriptors(desc_1, desc_2)
                    print(f"\nMatched {len(matches)} descriptors")

                    # Draw the matches
                    matches_drawn = draw_matches(img_1, img_2, coords_1, coords_2, matches)

                    # a. Save the matches to a JSON file
                    matches_path = os.path.join(output_dir, f"{name_1}_{name_2}_matches.json")
                    with open(matches_path, "w", encoding="utf-8") as output_file:
                        json.dump(
                            {
                                "image_1_path": path_1,
                                "image_2_path": path_2,
                                "coordinate_order": ["y", "x"],
                                "matches": matches.tolist()
                            },
                            output_file,
                            indent=2,
                            ensure_ascii=False
                        )
                    print(f"\nMatched descriptor indices for {name_1} and {name_2} saved to: {matches_path}")

                    # b. Save the visualized matches
                    drawn_path = os.path.join(output_dir, f"{name_1}_{name_2}_matches.png")
                    cv2.imwrite(drawn_path, matches_drawn)
                    print(f"Matched descriptor visualization for {name_1} and {name_2} saved to: {drawn_path}")


                # ==================== Section B.4: RANSAC for Robust Homography ====================

                elif args.section == "4":
                    matches = []











    else:
        raise argparse.ArgumentError(None, "Error: Invalid section number. Please choose from 1, 2, 3 or 4.")


if __name__ == "__main__":
    main()