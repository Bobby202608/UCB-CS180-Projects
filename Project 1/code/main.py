import csv
import cv2
import datetime
import numpy as np
import os
import time


"""
    ========== Configuration center (Editable) ==========
"""

# 1. Input images (directory and list of files)
IMAGES_DIR = "CS180_fa2026_proj1_data"
IMAGES_LIST = {
    "small": [
        "cathedral.jpg",
        "monastery.jpg",
        "tobolsk.jpg"
    ],

    "large": [
        "church.tif",
        "emir.tif",
        "harvesters.tif",
        "icon.tif",
        "ilemselga.tif",
        "melons.tif",
        "religous_painting.tif",
        "self_portrait.tif",
        "siren.tif",
        "three_generations.tif",
        "wharf.tif"
    ],

    "extra": [
        "00154.tif",
        "00203.tif",
        "00216.tif",
        "00245.tif",
        "00276.tif"
    ]
}

# 2. Whether to use the images for experiment
USE_IMAGES = {
    "small": True,
    "large": True
}

# 3. Displacements to align the channels
DISPLACEMENT_CONFIG = {
    "border_proportion": 0.05,
    "dx_min": -15,
    "dx_max": 15,
    "dy_min": -15,
    "dy_max": 15
}

# 4. Pyramid for multi-scale alignment
PYRAMID_CONFIG = {
    "levels": 5,
    "scale": 2,
    "d_refinement": 2
}

# 5. Output directories (images and CSV file)
OUTPUT_DIR = {
    "images": "output/images",
    "csv": "output/csv"
}

"""
    ========== End of configuration center ==========
"""

# 6. CSV fields (NOT EDITABLE)
CSV_FIELDS = [
    "image_name",
    "image_type",
    "input_path",
    "dx_g",
    "dy_g",
    "dx_r",
    "dy_r",
    "output_path",
    "time_elapsed(s)",
    "timestamp"
]


"""
    Functions
"""

def normalize(img: np.ndarray):
    """
        Normalize the image pixels to float values in [0, 1].
    """

    img = img.astype(np.float32)

    max_value = img.max()
    if max_value == 0:
        return img
    else:
        return img / max_value


def read_and_split(img_path: str):
    """
        Read the image and split it into three channels (B, G, R).
    """

    # 1. Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Error: Failed to read image: {img_path}")
    
    dtype = img.dtype
    print(f"Image dtype: {dtype}")

    # 2. Normalize the image pixels to float values in [0, 1]
    img = normalize(img)

    # 3. Crop the image's height to a multiple of 3
    h_raw, w_raw = img.shape[:2]
    print(f"Original image size: {w_raw} x {h_raw}")
    
    h = h_raw - h_raw % 3
    w = w_raw

    if h_raw % 3 == 2:
        img = img[1 : h_raw-1, : ]
    elif h_raw % 3 == 1:
        img = img[0 : h_raw-1, : ]

    print(f"Cropped image size: {w} x {h}")

    # 4. Split the image's three channels (from top to bottom: B, G, R)
    img_b = img[0 : h//3, : ]
    img_g = img[h//3 : h//3*2, : ]
    img_r = img[h//3*2 : h, : ]

    return img_b, img_g, img_r


def calculate_score(img_1: np.ndarray, img_2: np.ndarray):
    """
        Calculate the Euclidean Distance (L2 score) of two images.
    """

    score = np.sqrt(
        np.sum(
            (img_1 - img_2) ** 2
        )
    )

    # Normalize the score by the number of pixels
    return score / np.sqrt(img_1.shape[0] * img_1.shape[1])


def single_scale_align(
    img_ref: np.ndarray,
    img_mov: np.ndarray,
    d_config: dict
):
    """
        Exhaustively search the best displacement (dx, dy) for alignment.

        Inputs:
            - img_ref: Reference image (remains fixed)
            - img_mov: Moving image
            - Moving image will be aligned to the reference image

        Outputs:
            - img_out: Aligned moving image
            - dx_best: The best displacement along x-axis
            - dy_best: The best displacement along y-axis
    """

    if img_ref.shape != img_mov.shape:
        raise ValueError("Error: The two images to be aligned must have the same shape.")

    h, w = img_ref.shape[:2]

    # 1. Ignore a proportion of border pixels to calculate the score
    # Because the overlapping of border pixels would exaggerate the score and mislead the alignment
    border_proportion = d_config["border_proportion"]

    if not 0.0 <= border_proportion < 0.5:
        raise ValueError("Error: border_proportion must be in the range [0, 0.5).")

    border_h = int(h * border_proportion)
    border_w = int(w * border_proportion)

    ref_valid = img_ref[border_h : h - border_h, border_w : w - border_w]
    mov_valid = img_mov[border_h : h - border_h, border_w : w - border_w]

    h_valid, w_valid = ref_valid.shape[:2]

    dx_min = d_config["dx_min"]
    dx_max = d_config["dx_max"]
    dy_min = d_config["dy_min"]
    dy_max = d_config["dy_max"]

    score_best = float("inf")
    dx_best = 0
    dy_best = 0

    # 2. Exhaustively search around the initial displacement
    for dx in range(dx_min, dx_max + 1):
        for dy in range(dy_min, dy_max + 1):
            # a. Skip: the displacement is so large that the two images cannot overlap at all
            if abs(dx) >= w_valid or abs(dy) >= h_valid:
                continue

            # b. Tentatively shift the moving image
            mov_new = np.roll(mov_valid, shift = (dy, dx), axis = (0, 1))

            # c. Only consider the overlapping area of the two images
            x_overlap = slice(max(dx, 0), min(w_valid + dx, w_valid))
            y_overlap = slice(max(dy, 0), min(h_valid + dy, h_valid))

            ref_overlap = ref_valid[y_overlap, x_overlap]
            mov_overlap = mov_new[y_overlap, x_overlap]

            # d. Calculate the L2 score of the two overlapping areas
            score_new = calculate_score(ref_overlap, mov_overlap)

            if score_new < score_best:
                score_best = score_new
                dx_best = dx
                dy_best = dy

    if not np.isfinite(score_best):
        raise ValueError("Error: The displacement search found no overlapping pixels.")

    # 3. Shift the moving image with the best displacement
    img_out = np.roll(img_mov, shift = (dy_best, dx_best), axis = (0, 1))

    # 4. Ignore the non-overlapping pixels by setting them to 0
    if dx_best > 0:
        img_out[ : , : dx_best, ...] = 0
    elif dx_best < 0:
        img_out[ : , w + dx_best : , ...] = 0

    if dy_best > 0:
        img_out[ : dy_best, : , ...] = 0
    elif dy_best < 0:
        img_out[h + dy_best : , : , ...] = 0

    return img_out, dx_best, dy_best


def pyramid_align(
    img_ref: np.ndarray,
    img_mov: np.ndarray,
    d_config: dict,
    p_config: dict,
    level: int
):
    """
        Use a Gaussian pyramid for multi-scale alignment.

        Recursively align the images from the lowest-resolution level,
        then scale and refine the estimated displacement at each higher-resolution level,
        until reaching back to the original resolution.

        In a word: Multi-scale alignment = Several single-scale alignments.

        Inputs:
            - img_ref: Reference image (remains fixed)
            - img_mov: Moving image
            - Moving image will be aligned to the reference image

        Outputs (at current pyramid level):
            - img_out: Aligned moving image
            - dx_best: The best displacement along x-axis
            - dy_best: The best displacement along y-axis
    """

    if img_ref.shape != img_mov.shape:
        raise ValueError("Error: The two images to be aligned must have the same shape.")

    h, w = img_ref.shape[:2]

    # 1. The image at current level is too small, so use single-scale alignment
    if h <= 1 or w <= 1:
        return single_scale_align(img_ref, img_mov, d_config)

    # 2. Reached the top of pyramid, with the lowest image resolution
    if level == p_config["levels"]:
        return single_scale_align(img_ref, img_mov, d_config)

    # 3. Downsample the images for the next level
    scale = p_config["scale"]
    
    h_small = max(1, int(h // scale))
    w_small = max(1, int(w // scale))

    img_ref_small = cv2.resize(img_ref, (w_small, h_small), interpolation = cv2.INTER_AREA)
    img_mov_small = cv2.resize(img_mov, (w_small, h_small), interpolation = cv2.INTER_AREA)

    # 4. Find the best displacement (dx_small, dy_small) at the next level
    img_out_small, dx_small, dy_small = pyramid_align(img_ref_small, img_mov_small, d_config, p_config, level + 1)

    # 5. Scale (dx_small, dy_small) back to the current level: (dx, dy)
    dx = int(round(dx_small * w / w_small))
    dy = int(round(dy_small * h / h_small))

    # 6. Refine (dx, dy) at the current level
    d_refinement = p_config["d_refinement"]
    d_config_refined = d_config.copy()

    d_config_refined["dx_min"] = dx - d_refinement
    d_config_refined["dx_max"] = dx + d_refinement
    d_config_refined["dy_min"] = dy - d_refinement
    d_config_refined["dy_max"] = dy + d_refinement

    return single_scale_align(img_ref, img_mov, d_config_refined)


def align_and_merge(
    img_type: str,
    img_b: np.ndarray,
    img_g: np.ndarray,
    img_r: np.ndarray,
    d_config: dict,
    p_config: dict
):
    """
        Select an alignment method (single-scale or multi-scale) and merge the three channels.
    """

    # For large (TIF) images
    if img_type == "large":
        img_g_new, dx_g, dy_g = pyramid_align(img_b, img_g, d_config, p_config, 1)
        img_r_new, dx_r, dy_r = pyramid_align(img_b, img_r, d_config, p_config, 1)

    # For small (JPG/JPEG/PNG) images
    else:
        img_g_new, dx_g, dy_g = single_scale_align(img_b, img_g, d_config)
        img_r_new, dx_r, dy_r = single_scale_align(img_b, img_r, d_config)

    img_out = cv2.merge((img_b, img_g_new, img_r_new))

    return img_out, dx_g, dy_g, dx_r, dy_r


def process_images(img_paths: list, output_csv_path: str):
    """
        Entrance to image processing.

        Note (with an example):
        - img_path: /path/to/cathedral.jpg
        - img_file: cathedral.jpg
        - img_name: cathedral
    """

    for img_path in img_paths:
        img_file = os.path.basename(img_path)
        print(f"\nProcessing image: {img_file}")

        start_time = time.perf_counter()

        # 1. Read the image and split it into three channels (B, G, R)
        img_b, img_g, img_r = read_and_split(img_path)

        # 2. Align and merge the three channels
        img_type = "large" if img_path.lower().endswith(".tif") else "small"
        img_out, dx_g, dy_g, dx_r, dy_r = align_and_merge(
            img_type,
            img_b,
            img_g,
            img_r,
            DISPLACEMENT_CONFIG,
            PYRAMID_CONFIG
        )

        # 3. Save the output image with timestamp
        img_name = os.path.splitext(img_file)[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        output_img_path = os.path.join(OUTPUT_DIR["images"], f"{img_name}_{timestamp}.jpg")
        img_to_save = np.clip(img_out * 255, 0, 255).astype(np.uint8)

        if not cv2.imwrite(output_img_path, img_to_save):
            raise IOError(f"Error: Failed to save image: {output_img_path}")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # 4. Append a record to the CSV file
        record = {
            "image_name": img_name,
            "image_type": img_type,
            "input_path": img_path,
            "dx_g": dx_g,
            "dy_g": dy_g,
            "dx_r": dx_r,
            "dy_r": dy_r,
            "output_path": output_img_path,
            "time_elapsed(s)": f"{elapsed_time:.4f}",
            "timestamp": timestamp
        }

        append_csv_record(output_csv_path, record)

        print("Processing finished!")
        print(f"Output image saved to: {output_img_path}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")


def append_csv_record(csv_path: str, record: dict):
    """
        Append a record to the CSV file.
    """

    # The CSV file does not exist or is empty, write the header first
    needs_header = (not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0)

    with open(csv_path, "a", newline = "", encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        if record is not None:
            writer.writerow(record)


def main():
    # 1. Categorize all the images into two types: small_images (JPG/JPEG/PNG) and large_images (TIF)
    small_images = []
    large_images = []

    if IMAGES_LIST["small"]:
        for img_file in IMAGES_LIST["small"]:
            small_images.append(os.path.join(IMAGES_DIR, img_file))

    if IMAGES_LIST["large"]:
        for img_file in IMAGES_LIST["large"]:
            large_images.append(os.path.join(IMAGES_DIR, img_file))

    if IMAGES_LIST["extra"]:
        for img_file in IMAGES_LIST["extra"]:
            if img_file.lower().endswith(".tif"):
                large_images.append(os.path.join(IMAGES_DIR, img_file))
            elif img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                small_images.append(os.path.join(IMAGES_DIR, img_file))

    print(f"Total number of images: {len(small_images) + len(large_images)}")

    print(f"\nSmall images (count: {len(small_images)}):")
    for small_image in small_images:
        print(small_image)

    print(f"\nLarge images (count: {len(large_images)}):")
    for large_image in large_images:
        print(large_image)

    # 2. Create directories for output files if they do not exist
    if not os.path.exists(OUTPUT_DIR["images"]):
        os.makedirs(OUTPUT_DIR["images"])

    if not os.path.exists(OUTPUT_DIR["csv"]):
        os.makedirs(OUTPUT_DIR["csv"])

    # 3. Input/output image path, best displacements, time elapsed etc. are saved to a CSV file after processing each image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_path = os.path.join(OUTPUT_DIR["csv"], f"results_{timestamp}.csv")

    # 4. Start processing the images
    print("\nStarting image processing...")

    if USE_IMAGES["small"]:
        process_images(small_images, output_csv_path)

    if USE_IMAGES["large"]:
        process_images(large_images, output_csv_path)

    print("\nAll images have been processed!")
    print(f"Results saved to: {output_csv_path}")


if __name__ == "__main__":
    main()
