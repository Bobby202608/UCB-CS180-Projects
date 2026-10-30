import argparse
import cv2
import json
import numpy as np
import os
import time


# ==================== Configuration Center (EDITABLE) ====================

# 1. Image paths
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
    [3439, 436], [4299, 707], [4391, 389], [5032, 569],
    [3552, 1319], [3671, 1303], [4095, 1075], [4055, 1298],
    [4534, 969], [4763, 930], [5094, 1040], [3508, 1822],
    [4088, 1655], [4148, 1753], [4682, 1637], [4797, 1570],
    [5279, 1977], [3730, 2148], [4514, 2087], [3767, 2700],
    [4728, 2665], [5467, 2621], [4185, 2978], [4816, 3254]
]
GATEWAY_2_PTS = [
    [208, 88], [1176, 777], [1314, 511], [1848, 911],
    [193, 1137], [357, 1165], [902, 1072], [823, 1291],
    [1365, 1113], [1580, 1147], [1842, 1340], [31, 1716],
    [803, 1677], [852, 1791], [1409, 1785], [1527, 1747],
    [1875, 2186], [275, 2149], [1182, 2196], [212, 2799],
    [1300, 2784], [1955, 2759], [671, 3108], [1290, 3372]
]

# 3. Homography matrices
TARGET_H = [
    [2.0608300803583686, -0.22612547769545147, -1483.2522715836299],
    [0.5562167615633093, 1.5718379197923689, -619.8083856139616],
    [0.0005604107691495885, -0.00010501105323745819, 1.0]
]

GATEWAY_H = [
    [4.310889578761602, -0.6369067342707085, -13884.903488109532],
    [1.821812238593239, 3.410059308772187, -7462.470190408865],
    [0.0006095464285668095, -7.788290717242186e-05, 1.0]
]

# 4. Output paths
OUTPUT_PATH_2 = "output/parta_2/homographies.json"
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
    if pts_1.shape[0] <= 4 or pts_2.shape[0] <= 4:
        raise ValueError("Error: At least 5 correspondence points are required for each image.")
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

def inverseWarp(img: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Inverse warp the image.
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

    # 5. Keep the source image's coordinates within the extents
    src_x = np.clip(src_x, 0, w-1)
    src_y = np.clip(src_y, 0, h-1)

    # 6. Prepare the ready-to-fill output image
    output = np.zeros(
        (dst_y_max - dst_y_min + 1, dst_x_max - dst_x_min + 1) + img.shape[2:],
        dtype=img.dtype
    )  # shape: (n, m, c) for color, (n, m) for grayscale

    return src_x, src_y, output


def warpImageNearestNeighbor(img: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, float]:
    """
        Use nearest neighbor interpolation to warp the image.
    """

    if H.shape != (3, 3):
        raise ValueError("Error: The homography matrix H must be 3x3.")

    start_time = time.perf_counter()
    src_x, src_y, output = inverseWarp(img, H)

    h, w = img.shape[:2]

    # 1. Find the nearest neighbor with integer coordinates
    x_nearest = np.floor(src_x + 0.5).astype(np.intp)  # shape: (nm, )
    y_nearest = np.floor(src_y + 0.5).astype(np.intp)  # shape: (nm, )

    # 2. Conduct warping
    output[...] = img[y_nearest, x_nearest].reshape(output.shape)  # shape: (n, m, c) for color, (n, m) for grayscale

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return output, elapsed_time


def warpImageBilinear(img: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, float]:
    """
        Use bilinear interpolation to warp the image.
    """

    if H.shape != (3, 3):
        raise ValueError("Error: The homography matrix H must be 3x3.")

    start_time = time.perf_counter()
    src_x, src_y, output = inverseWarp(img, H)

    h, w = img.shape[:2]










    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return output, elapsed_time


# ==================== Section A.4: Blend the Images into a Mosaic ====================










def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        required=True,
        choices=["2", "3", "4"],
        help="Which section to run"
    )
    args = parser.parse_args()

    # Section A.2: Recover Homographies
    if args.section == "2":
        print("Running Section A.2: Recover Homographies")

        output_path = OUTPUT_PATH_2
        output_dir = os.path.dirname(output_path)
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
            if img_1.shape != img_2.shape:
                raise ValueError("Error: The two images must have the same shape.")

            name_1 = os.path.basename(path_1).split(".")[0]
            name_2 = os.path.basename(path_2).split(".")[0]

            # Compute the image pair's homography matrix
            print(f"\nComputing homography matrix for {name_1} and {name_2}...")

            pts_1 = np.array(pts_1, dtype=np.float64)
            pts_2 = np.array(pts_2, dtype=np.float64)
            H = compute_H(pts_1, pts_2)

            # Write the homography matrix to a JSON file
            pair_name = f"{name_1}_{name_2}"
            homographies[pair_name] = {
                "source": name_1,
                "target": name_2,
                "H": H.tolist()
            }

            with open(output_path, "w", encoding="utf-8") as output_file:
                json.dump(homographies, output_file, indent=2, ensure_ascii=False)
                output_file.write("\n")

            print(f"The homography matrix for {name_1} and {name_2} is written to {output_path}")

    # Section A.3: Warp the Images
    elif args.section == "3":
        print("Running Section A.3: Warp the Images")

        output_dir = OUTPUT_DIR_3
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for path_1, H in [
            (TARGET_1_PATH, TARGET_H),
            (GATEWAY_1_PATH, GATEWAY_H)
        ]:
            if not os.path.exists(path_1):
                raise FileNotFoundError(f"Error: {path_1} not found.")

            img_1 = cv2.imread(path_1)

            if img_1 is None:
                raise FileNotFoundError(f"Error: Failed to read {path_1}.")

            name = os.path.basename(path_1).split(".")[0]
            H = np.array(H, dtype=np.float64)

            print(f"\nWarping {name} using nearest neighbor interpolation...")
            warped_nn, time_nn = warpImageNearestNeighbor(img_1, H)
            print(f"Warping finished, time elapsed: {time_nn:.4f}s")

            print(f"\nWarping {name} using bilinear interpolation...")
            warped_bil, time_bil = warpImageBilinear(img_1, H)
            print(f"Warping finished, time elapsed: {time_bil:.4f}s")

            for result, suffix in [
                (warped_nn, "nn"),
                (warped_bil, "bil")
            ]:
                output_path = os.path.join(output_dir, f"{name}_{suffix}.png")
                cv2.imwrite(output_path, result)
                print(f"Saved warped image: {name}_{suffix}.png")

    # Section A.4: Blend the Images into a Mosaic
    elif args.section == "4":
        print("Running Section A.4: Blend the Images into a Mosaic")










    else:
        raise argparse.ArgumentError(None, "Error: Invalid section number. Please choose from 2, 3 or 4.")


if __name__ == "__main__":
    main()