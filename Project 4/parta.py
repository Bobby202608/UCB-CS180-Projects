import argparse
import cv2
import json
import numpy as np
import os
import time


# ==================== Configuration Center (EDITABLE) ====================

# 1. Input image paths
TARGET_1_PATH = "input/target_1.jpeg"
TARGET_2_PATH = "input/target_2.jpeg"

GATEWAY_1_PATH = "input/gateway_1.jpeg"
GATEWAY_2_PATH = "input/gateway_2.jpeg"

# 2. Correspondence points for each image pair
TARGET_1_PTS = [
    [955, 1011], [1143, 1006], [992, 1100], [1163, 1093],
    [1316, 1000], [1368, 1003], [1471, 1028], [1552, 1025],
    [1619, 1017], [1634, 1062], [1697, 1024], [1741, 1010],
    [1812, 999], [1904, 950], [2011, 1010], [1132, 920],
    [1365, 889], [1622, 866], [1620, 674], [1694, 733],
    [1841, 762], [1897, 791], [1935, 763], [1987, 786]
]
TARGET_2_PTS = [
    [178, 1050], [418, 1043], [218, 1152], [433, 1134],
    [617, 1033], [670, 1034], [765, 1057], [841, 1052],
    [901, 1042], [910, 1085], [968, 1047], [1003, 1035],
    [1059, 1024], [1131, 983], [1205, 1035], [416, 949],
    [678, 920], [916, 904], [926, 728], [984, 787],
    [1096, 821], [1135, 848], [1163, 825], [1198, 849]
]

GATEWAY_1_PTS = [
    [1233, 156], [1541, 253], [1574, 139], [1804, 204],
    [1274, 473], [1316, 467], [1468, 385], [1454, 465],
    [1626, 347], [1708, 333], [1826, 373], [1258, 653],
    [1466, 593], [1487, 629], [1679, 587], [1720, 563],
    [1893, 709], [1337, 770], [1618, 748], [1351, 968],
    [1695, 956], [1960, 940], [1501, 1068], [1727, 1167]
]
GATEWAY_2_PTS = [
    [75, 32], [422, 279], [471, 183], [663, 327],
    [69, 408], [128, 418], [323, 384], [295, 463],
    [489, 399], [566, 411], [660, 480], [11, 615],
    [288, 601], [305, 642], [505, 640], [547, 626],
    [672, 784], [99, 771], [424, 787], [76, 1004],
    [466, 998], [701, 989], [241, 1114], [463, 1209]
]

# 3. Output directories
OUTPUT_DIR_2 = "output/parta_2/"
OUTPUT_DIR_3 = "output/parta_3/"
OUTPUT_DIR_4 = "output/parta_4/"

# ==================== Configuration Center Ends ====================


# ==================== Section A.2: Recover Homographies ====================

def compute_H(pts_1: np.ndarray, pts_2: np.ndarray) -> np.ndarray:
    """
        Compute the homography matrix H such that pts_2 = H * pts_1.
    """

    H = np.zeros((3, 3))

    if pts_1.ndim != 2 or pts_2.ndim != 2:
        raise ValueError("Error: The correspondence points for each image must be 2-dimensional.")
    if pts_1.shape[1] != 2 or pts_2.shape[1] != 2:
        raise ValueError("Error: Each correspondence point must have 2 coordinates (x, y).")
    if pts_1.shape[0] < 4 or pts_2.shape[0] < 4:
        raise ValueError("Error: At least 4 correspondence points are required for each image.")
    if pts_1.shape[0] != pts_2.shape[0]:
        raise ValueError("Error: The number of correspondence points must be the same for both images.")

    """
        H = [[h11, h12, h13], [h21, h22, h23], [h31, h32, 1]]
        p = [x1, y1, 1]^T
        p' = [x2, y2, 1]^T

        Hp = λp' (λ: scaling factor, will be rescaled to 1)

    =>  h11 * x1 + h12 * y1 + h13 = x2 * λ = x2 * (h31 * x1 + h32 * y1 + 1)
        h21 * x1 + h22 * y1 + h23 = y2 * λ = y2 * (h31 * x1 + h32 * y1 + 1)
        h31 * x1 + h32 * y1 + 1 = λ

    =>  [x1] * h11 + [y1] * h12 + [1] * h13 - [x1 * x2] * h31 - [y1 * x2] * h32 = x2
        [x1] * h21 + [y1] * h22 + [1] * h23 - [x1 * y2] * h31 - [y1 * y2] * h32 = y2

    =>  reformulate into Ah = b
    """

    A, b = [], []

    for (x1, y1), (x2, y2) in zip(pts_1, pts_2):
        A.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2])
        A.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2])
        b.extend([x2, y2])

    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    H = np.append(h, 1).reshape(3, 3)
    return H


# ==================== Section A.3: Warp the Images ====================

def inverseWarp(img: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray, np.ndarray]:
    """
        Return valid source coordinates, an empty output image, and its 2D validity mask.
        Coordinates follow the row-major order of True entries in the mask.
    """

    h, w = img.shape[:2]

    # 1. Find the warped image's extents using four corner points
    src_corners = np.array([
        [0, 0, 1],     # upper-left corner
        [w-1, 0, 1],   # upper-right corner
        [0, h-1, 1],   # lower-left corner
        [w-1, h-1, 1]  # lower-right corner
    ])

    dst_corners = H @ src_corners.T

    # 2. Check the corners' 3rd coordinates
    # If any of the 3rd coordinates is 0, it would cause division-by-zero error in normalization to fail the warping
    dst_z = dst_corners[2]
    if np.any(dst_z == 0):
        raise ValueError("Error: The warped image would have invalid coordinates.")

    # Normalize
    dst_corners = dst_corners[:2] / dst_z

    dst_x_min = int(np.floor(np.min(dst_corners[0])))
    dst_x_max = int(np.ceil(np.max(dst_corners[0])))
    dst_y_min = int(np.floor(np.min(dst_corners[1])))
    dst_y_max = int(np.ceil(np.max(dst_corners[1])))

    dst_offset = (dst_x_min, dst_y_min)

    # 3. Create coordinates for the warped image using the derived extents
    # n: output height, m: output width

    """
        dst_x = [[x1, x2, ..., xm],
                 [x1, x2, ..., xm],
                 ...,
                 [x1, x2, ..., xm]] (shape: (n, m))

        dst_y = [[y1, y1, ..., y1],
                 [y2, y2, ..., y2],
                 ...,
                 [yn, yn, ..., yn]] (shape: (n, m))
    """

    dst_x, dst_y = np.meshgrid(
        np.arange(dst_x_min, dst_x_max + 1, dtype=np.float64),
        np.arange(dst_y_min, dst_y_max + 1, dtype=np.float64)
    )

    # 4. Conduct inverse warping to get coordinates of the source image

    """
        dst = [[x1, x2, ..., xm, x1, x2, ..., xm, ..., x1, x2, ..., xm],
               [y1, y1, ..., y1, y2, y2, ..., y2, ..., yn, yn, ..., yn],
               [ 1,  1, ...,  1,  1,  1, ...,  1, ...,  1,  1, ...,  1]] (shape: (3, nm))
    """

    dst = np.stack([
        dst_x.ravel(),
        dst_y.ravel(),
        np.ones_like(dst_x.ravel())
    ])

    src = np.linalg.inv(H) @ dst

    # Normalize
    with np.errstate(divide="ignore", invalid="ignore"):
        src_x = src[0] / src[2]  # shape: (nm, )
        src_y = src[1] / src[2]  # shape: (nm, )

    # 5. Use a mask to keep valid source coordinates (finite and in the extents) only
    valid = (
        np.isfinite(src_x) & np.isfinite(src_y) &
        (src_x >= 0) & (src_x <= w-1) &
        (src_y >= 0) & (src_y <= h-1)
    )  # shape: (nm, )

    src_x = src_x[valid]
    src_y = src_y[valid]

    # 6. Prepare the ready-to-fill output image
    output = np.zeros(
        (dst_y_max+1 - dst_y_min, dst_x_max+1 - dst_x_min) + img.shape[2:],
        dtype=img.dtype
    )  # shape: (n, m, c) for color, (n, m) for grayscale

    valid = valid.reshape(output.shape[:2])
    return src_x, src_y, dst_offset, output, valid


def warpImageNearestNeighbor(img: np.ndarray, H: np.ndarray) -> tuple[tuple[int, int], np.ndarray, np.ndarray, float]:
    """
        Use nearest neighbor interpolation to warp the image.
    """

    if H.shape != (3, 3):
        raise ValueError("Error: The homography matrix H must be 3x3.")

    start_time = time.perf_counter()
    src_x, src_y, dst_offset, output, valid = inverseWarp(img, H)

    # 1. Find the nearest neighbor pixel with integer coordinates
    x_nearest = np.floor(src_x + 0.5).astype(np.intp)  # shape: (nm, )
    y_nearest = np.floor(src_y + 0.5).astype(np.intp)  # shape: (nm, )

    # 2. Conduct warping
    output[valid] = img[y_nearest, x_nearest]

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return dst_offset, output, valid, elapsed_time


def warpImageBilinear(img: np.ndarray, H: np.ndarray) -> tuple[tuple[int, int], np.ndarray, np.ndarray, float]:
    """
        Use bilinear interpolation to warp the image.
    """

    if H.shape != (3, 3):
        raise ValueError("Error: The homography matrix H must be 3x3.")

    start_time = time.perf_counter()
    src_x, src_y, dst_offset, output, valid = inverseWarp(img, H)

    h, w = img.shape[:2]

    # 1. Find the four surrounding pixels with integer coordinates

    """
        (x1, y1): upper-left pixel
        (x2, y1): upper-right pixel
        (x1, y2): lower-left pixel
        (x2, y2): lower-right pixel
    """

    x1 = np.floor(src_x).astype(np.intp)
    y1 = np.floor(src_y).astype(np.intp)
    x2 = np.minimum(x1+1, w-1)
    y2 = np.minimum(y1+1, h-1)
    # shape: (nm, )

    upper_left = img[y1, x1].astype(np.float64)
    upper_right = img[y1, x2].astype(np.float64)
    lower_left = img[y2, x1].astype(np.float64)
    lower_right = img[y2, x2].astype(np.float64)
    # shape: (nm, c) for color, (nm, ) for grayscale

    # 2. Compute the weight for each surrounding pixel based on distance
    dx = src_x - x1
    dy = src_y - y1

    dx = dx.reshape((-1, ) + (1, ) * (img.ndim - 2))
    dy = dy.reshape((-1, ) + (1, ) * (img.ndim - 2))
    # shape: (nm, 1) for color, (nm, ) for grayscale

    # 3. Conduct warping
    average = (
        (1 - dx) * (1 - dy) * upper_left +
        dx * (1 - dy) * upper_right +
        (1 - dx) * dy * lower_left +
        dx * dy * lower_right
    )
    output[valid] = average
    # shape: (nm, c) for color, (nm, ) for grayscale

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return dst_offset, output, valid, elapsed_time


# ==================== Section A.4: Blend the Images into a Mosaic ====================

def blend(
        warped_1: np.ndarray,
        img_2: np.ndarray,
        dst_offset: tuple[int, int],
        valid_1: np.ndarray,
        valid_2: np.ndarray
    ) -> np.ndarray:
    """
        Blend the two images: (1) warped img_1 and (2) img_2, into a mosaic, using simple feathering.
    """

    h1, w1 = warped_1.shape[:2]
    h2, w2 = img_2.shape[:2]
    dst_x_min, dst_y_min = dst_offset

    # 1. Find the mosaic canvas' shape
    # min/max x/y coordinates of the two images
    x_min = min(dst_x_min, 0)
    y_min = min(dst_y_min, 0)
    x_max = max(dst_x_min + w1, w2)
    y_max = max(dst_y_min + h1, h2)

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    canvas_shape = (canvas_h, canvas_w) + img_2.shape[2:]

    # 2. Shift both images to guarantee non-negative coordinates on the canvas

    """
        if dst_x_min or dst_y_min < 0, x_min or y_min < 0, then x1 or y1 will be 0, x2 or y2 will be positive (shifted)
        if dst_x_min or dst_y_min >= 0, x_min or y_min = 0, then x1 or y1 will be >= 0, x2 or y2 will be 0 (not shifted)
    """

    x1 = dst_x_min - x_min
    y1 = dst_y_min - y_min

    x2 = -x_min
    y2 = -y_min

    # 3. Create the mosaic canvas
    canvas_1 = np.zeros(canvas_shape, dtype=warped_1.dtype)
    canvas_2 = np.zeros(canvas_shape, dtype=img_2.dtype)

    canvas_1[y1:y1+h1, x1:x1+w1] = warped_1
    canvas_2[y2:y2+h2, x2:x2+w2] = img_2

    # 4. Create masks for the canvas
    mask_1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask_2 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask_1[y1:y1+h1, x1:x1+w1] = valid_1
    mask_2[y2:y2+h2, x2:x2+w2] = valid_2

    # 5. Compute each pixel's distance to the nearest invalid pixel (i.e. 0 in the mask)
    def compute_distance(mask):
        # Pad the mask with a circle of 0s to treat out-of-bound pixels as invalid
        padded = np.pad(mask, 1, mode="constant", constant_values=0)

        return cv2.distanceTransform(
            padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )[1:-1, 1:-1].copy()

    d1 = compute_distance(mask_1)
    d2 = compute_distance(mask_2)

    # 6. Compute each image's weight based on the distances
    w1 = d1 / (d1 + d2 + 1e-8)
    w2 = d2 / (d1 + d2 + 1e-8)

    if canvas_1.ndim == 3:
        w1 = w1[..., None]
        w2 = w2[..., None]

    # 7. Weight-average the two images to get the mosaic
    mosaic = w1 * canvas_1 + w2 * canvas_2
    mosaic = np.clip(np.rint(mosaic), 0, 255).astype(img_2.dtype)

    return mosaic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        required=True,
        choices=["2", "3", "4"],
        help="Which section to run"
    )
    args = parser.parse_args()

    if args.section in ["2", "3", "4"]:
        if args.section == "2":
            print("Running Section A.2: Recover Homographies")
            output_dir = OUTPUT_DIR_2
        elif args.section == "3":
            print("Running Section A.3: Warp the Images")
            output_dir = OUTPUT_DIR_3
        else:
            print("Running Section A.4: Blend the Images into a Mosaic")
            output_dir = OUTPUT_DIR_4

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        homographies = {}

        for path_1, path_2, pts_1, pts_2 in [
            (TARGET_1_PATH, TARGET_2_PATH, TARGET_1_PTS, TARGET_2_PTS),
            (GATEWAY_1_PATH, GATEWAY_2_PATH, GATEWAY_1_PTS, GATEWAY_2_PTS)
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

            pts_1 = np.array(pts_1, dtype=np.float64)
            pts_2 = np.array(pts_2, dtype=np.float64)
            H = compute_H(pts_1, pts_2)

            # Section A.2: Recover Homographies
            if args.section == "2":
                # Write the homography matrix to a JSON file
                pair_name = f"{name_1}_{name_2}"
                homographies[pair_name] = {
                    "source": name_1,
                    "target": name_2,
                    "H": H.tolist()
                }

                output_path = os.path.join(output_dir, "homographies.json")
                with open(output_path, "w", encoding="utf-8") as output_file:
                    json.dump(homographies, output_file, indent=2, ensure_ascii=False)
                    output_file.write("\n")

            # Section A.3: Warp the Images
            elif args.section == "3":
                print(f"\nWarping {name_1} using nearest neighbor interpolation...")
                _, warped_nn, _, time_nn = warpImageNearestNeighbor(img_1, H)
                print(f"Warping finished, time elapsed: {time_nn:.4f}s")

                print(f"\nWarping {name_1} using bilinear interpolation...")
                _, warped_bil, _, time_bil = warpImageBilinear(img_1, H)
                print(f"Warping finished, time elapsed: {time_bil:.4f}s\n")

                for result, suffix in [
                    (warped_nn, "nn"),
                    (warped_bil, "bil")
                ]:
                    output_path = os.path.join(output_dir, f"{name_1}_{suffix}.png")
                    cv2.imwrite(output_path, result)
                    print(f"Saved warped image to: {output_path}")

            # Section A.4: Blend the Images into a Mosaic
            else:
                print(f"\nWarping {name_1} using bilinear interpolation...")
                dst_offset, warped_1, valid_1, time = warpImageBilinear(img_1, H)
                print(f"Warping finished, time elapsed: {time:.4f}s")

                valid_2 = np.ones(img_2.shape[:2], dtype=np.bool)

                mosaic = blend(warped_1, img_2, dst_offset, valid_1, valid_2)

                output_path = os.path.join(output_dir, f"{name_1}_{name_2}.png")
                cv2.imwrite(output_path, mosaic)
                print(f"Saved mosaic to: {output_path}")

        if args.section == "2":
            print(f"The homography matrices are written to: {output_path}")

    else:
        raise argparse.ArgumentError(None, "Error: Invalid section number. Please choose from 2, 3 or 4.")


if __name__ == "__main__":
    main()