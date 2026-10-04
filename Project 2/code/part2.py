import argparse
import cv2
import datetime
import numpy as np
import os
import scipy
import sys
import time

from utils import print_logs


OUTPUT_DIR_1 = "output/part2_1"
OUTPUT_DIR_2 = "output/part2_2"
OUTPUT_DIR_3 = "output/part2_3"
OUTPUT_DIR_4 = "output/part2_4"

# =============== Part 2.1: Image Sharpening ===============

def section_1(args) -> None:
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"\nError: Failed to read the image: {args.input}")

    img_name = os.path.splitext(os.path.basename(args.input))[0]
    output_dir = os.path.join(OUTPUT_DIR_1, img_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # a. Use Gaussian filter to blur the image
    gaussian = cv2.getGaussianKernel(5, 1) @ cv2.getGaussianKernel(5, 1).T
    img_blur = scipy.signal.convolve2d(image, gaussian, mode="same")
    img_blur_path = os.path.join(output_dir, f"{img_name}_blurred.png")

    cv2.imwrite(img_blur_path, img_blur)
    print(f"\nBlurred image saved to: {img_blur_path}")

    # b. Get the high-frequency component
    high_freq = image - img_blur
    high_freq_path = os.path.join(output_dir, f"{img_name}_high_freq.png")

    # Normalize the pixels to [0, 255]
    max_pixel = np.max(high_freq)
    min_pixel = np.min(high_freq)

    if max_pixel > min_pixel:
        high_freq_norm = (high_freq - min_pixel) / (max_pixel - min_pixel) * 255.0
    else:
        high_freq_norm = high_freq

    high_freq_norm = np.clip(high_freq_norm, 0, 255).astype(np.uint8)
    cv2.imwrite(high_freq_path, high_freq_norm)
    print(f"High-frequency component saved to: {high_freq_path}")

    # c. Use unsharp mask filter to sharpen the image
    # unsharp kernel = (1 + α) I - α G
    alpha = [0.5, 1.0, 2.0, 5.0, 10.0]

    for a in alpha:
        identity = np.zeros_like(gaussian)
        identity[gaussian.shape[0] // 2, gaussian.shape[1] // 2] = 1.0
        unsharp_kernel = (1.0 + a) * identity - a * gaussian

        img_sharp = scipy.signal.convolve2d(image.astype(np.float64), unsharp_kernel, mode="same")
        img_sharp = np.clip(img_sharp, 0, 255).astype(np.uint8)
        img_sharp_path = os.path.join(output_dir, f"{img_name}_sharpened_{a}.png")

        cv2.imwrite(img_sharp_path, img_sharp)
        print(f"Sharpened image (α = {a}) saved to: {img_sharp_path}")


# =============== Part 2.2: Hybrid Images ===============

def section_2(args) -> None:
    from utils import hybrid_image

    # img_1: high-frequency component to be extracted
    img_1 = cv2.imread(args.input[0], cv2.IMREAD_COLOR)
    if img_1 is None:
        raise FileNotFoundError(f"\nError: Failed to read the image: {args.input[0]}")

    # img_2: low-frequency component to be extracted
    img_2 = cv2.imread(args.input[1], cv2.IMREAD_COLOR)
    if img_2 is None:
        raise FileNotFoundError(f"\nError: Failed to read the image: {args.input[1]}")

    img_1_name = os.path.splitext(os.path.basename(args.input[0]))[0]
    img_2_name = os.path.splitext(os.path.basename(args.input[1]))[0]

    output_dir = os.path.join(OUTPUT_DIR_2, f"{img_1_name}_{img_2_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Normalize the pixels to [0, 1]
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    # Cutoff frequencies
    sigma_high = 12.0
    sigma_low = 8.0

    img_1_aligned, img_2_aligned, img_1_high, img_2_low, hybrid = hybrid_image(
        img_1, img_2, sigma_high, sigma_low
    )

    def save_rgb(path: str, image: np.ndarray) -> None:
        rgb = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # a1. Aligned image 1
    img_1_aligned_path = os.path.join(output_dir, f"{img_1_name}_aligned.png")
    save_rgb(img_1_aligned_path, img_1_aligned)
    print(f"\nAligned image 1 saved to: {img_1_aligned_path}")

    # a2. Aligned image 2
    img_2_aligned_path = os.path.join(output_dir, f"{img_2_name}_aligned.png")
    save_rgb(img_2_aligned_path, img_2_aligned)
    print(f"Aligned image 2 saved to: {img_2_aligned_path}")

    # b2. Low-frequency component of image 2
    img_2_low_path = os.path.join(output_dir, f"{img_2_name}_low.png")
    save_rgb(img_2_low_path, img_2_low)
    print(f"Low-frequency component of image 2 saved to: {img_2_low_path}")

    # b1. High-frequency component of image 1
    max_pixel = np.max(img_1_high)
    min_pixel = np.min(img_1_high)

    if max_pixel > min_pixel:
        # save_rgb expects floating-point RGB values in [0, 1].
        img_1_high = (img_1_high - min_pixel) / (max_pixel - min_pixel)

    img_1_high_path = os.path.join(output_dir, f"{img_1_name}_high.png")
    save_rgb(img_1_high_path, img_1_high)
    print(f"High-frequency component of image 1 saved to: {img_1_high_path}")

    # c. Hybrid image
    hybrid_path = os.path.join(output_dir, f"{img_1_name}_{img_2_name}_hybrid.png")
    save_rgb(hybrid_path, hybrid)
    print(f"Hybrid image (σ_low = {sigma_low}, σ_high = {sigma_high}) saved to: {hybrid_path}")


# =============== Part 2.3: Gaussian and Laplacian Stacks ===============

def section_3(args) -> None:
    pass










# =============== Part 2.4: Multiresolution Blending ===============

def section_4(args) -> None:
    pass










def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for running the code of Part 2.")

    parser.add_argument(
        "--section",
        type=int,
        choices=(1, 2, 3, 4),
        required=True,
        help="Section to run (2.1/2.2/2.3/2.4)",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="One image for section 1; two images for sections 2, 3, and 4",
    )
    args = parser.parse_args()

    input_expected = 1 if args.section == 1 else 2
    if len(args.input) != input_expected:
        parser.error(f"Error: Part 2.{args.section} requires exactly {input_expected} input image(s).")

    if args.section == 1:
        args.input = args.input[0]
    return args


def main() -> None:
    args = parse_args()

    # Part 2.1: Image Sharpening
    if args.section == 1:
        if not os.path.exists(OUTPUT_DIR_1):
            os.makedirs(OUTPUT_DIR_1)

        with print_logs(os.path.join(OUTPUT_DIR_1, f"part2_1_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")):
            print("Executing Part 2.1: Image Sharpening")
            section_1(args)

    # Part 2.2: Hybrid Images
    elif args.section == 2:
        if not os.path.exists(OUTPUT_DIR_2):
            os.makedirs(OUTPUT_DIR_2)

        with print_logs(os.path.join(OUTPUT_DIR_2, f"part2_2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")):
            print("Executing Part 2.2: Hybrid Images")
            section_2(args)

    # Part 2.3: Gaussian and Laplacian Stacks
    elif args.section == 3:
        if not os.path.exists(OUTPUT_DIR_3):
            os.makedirs(OUTPUT_DIR_3)

        with print_logs(os.path.join(OUTPUT_DIR_3, f"part2_3_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")):
            print("Executing Part 2.3: Gaussian and Laplacian Stacks")
            section_3(args)

    # Part 2.4: Multiresolution Blending
    elif args.section == 4:
        if not os.path.exists(OUTPUT_DIR_4):
            os.makedirs(OUTPUT_DIR_4)

        with print_logs(os.path.join(OUTPUT_DIR_4, f"part2_4_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")):
            print("Executing Part 2.4: Multiresolution Blending")
            section_4(args)

    else:
        raise argparse.ArgumentError(None, "Error: Invalid section number. Please input 1, 2, 3 or 4.")


if __name__ == "__main__":
    main()
