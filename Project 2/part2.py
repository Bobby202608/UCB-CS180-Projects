import argparse
import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import sys
import time

from utils import print_logs


OUTPUT_DIR_1 = "output/part2_1"
OUTPUT_DIR_2 = "output/part2_2"
OUTPUT_DIR_3 = "output/part2_3_4"

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
        # Normalize the pixels to [0, 255]
        min_value = np.min(image)
        max_value = np.max(image)

        if max_value > min_value:
            result = (image - min_value) / (max_value - min_value)
        else:
            result = np.zeros_like(image)

        rgb = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # a1. Aligned image 1
    img_1_aligned_path = os.path.join(output_dir, f"{img_1_name}_aligned.png")
    save_rgb(img_1_aligned_path, img_1_aligned)
    print(f"\nAligned image 1 saved to: {img_1_aligned_path}")

    # a2. Aligned image 2
    img_2_aligned_path = os.path.join(output_dir, f"{img_2_name}_aligned.png")
    save_rgb(img_2_aligned_path, img_2_aligned)
    print(f"Aligned image 2 saved to: {img_2_aligned_path}")

    # b1. High-frequency component of image 1
    img_1_high_path = os.path.join(output_dir, f"{img_1_name}_high.png")
    save_rgb(img_1_high_path, img_1_high)
    print(f"High-frequency component of image 1 saved to: {img_1_high_path}")

    # b2. Low-frequency component of image 2
    img_2_low_path = os.path.join(output_dir, f"{img_2_name}_low.png")
    save_rgb(img_2_low_path, img_2_low)
    print(f"Low-frequency component of image 2 saved to: {img_2_low_path}")

    # c. Hybrid image
    hybrid_path = os.path.join(output_dir, f"{img_1_name}_{img_2_name}_hybrid.png")
    save_rgb(hybrid_path, hybrid)
    print(f"Hybrid image (σ_low = {sigma_low}, σ_high = {sigma_high}) saved to: {hybrid_path}")

    # d. Log magnitude of Fourier Transform
    def save_fft(path: str, image: np.ndarray) -> None:
        # Convert the image to grayscale if it is RGB
        if image.ndim == 3:
            image = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2])

        log_magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image, axes=(0, 1)))))

        # Normalize the log magnitude to [0, 1]
        log_max = np.max(log_magnitude)
        log_min = np.min(log_magnitude)

        if log_max > log_min:
            log_magnitude = (log_magnitude - log_min) / (log_max - log_min)
        else:
            log_magnitude = np.zeros_like(log_magnitude)

        plt.imsave(path, log_magnitude, cmap="gray", vmin=0.0, vmax=1.0)

    save_fft(os.path.join(output_dir, f"{img_1_name}_fft.png"), img_1_aligned)
    save_fft(os.path.join(output_dir, f"{img_2_name}_fft.png"), img_2_aligned)
    save_fft(os.path.join(output_dir, f"{img_1_name}_high_fft.png"), img_1_high)
    save_fft(os.path.join(output_dir, f"{img_2_name}_low_fft.png"), img_2_low)
    save_fft(os.path.join(output_dir, f"{img_1_name}_{img_2_name}_hybrid_fft.png"), hybrid)
    print(f"Log magnitude of Fourier Transform saved to: {output_dir}")


# =============== Part 2.3 & 2.4: Gaussian and Laplacian Stacks, Multiresolution Blending ===============

def section_3_4(args) -> None:
    img_1 = cv2.imread(args.input[0], cv2.IMREAD_COLOR)
    if img_1 is None:
        raise FileNotFoundError(f"\nError: Failed to read the image: {args.input[0]}")

    img_2 = cv2.imread(args.input[1], cv2.IMREAD_COLOR)
    if img_2 is None:
        raise FileNotFoundError(f"\nError: Failed to read the image: {args.input[1]}")

    if img_1.shape != img_2.shape:
        raise ValueError("Error: The two images must have the same shape.")

    img_1_name = os.path.splitext(os.path.basename(args.input[0]))[0]
    img_2_name = os.path.splitext(os.path.basename(args.input[1]))[0]

    output_dir = os.path.join(OUTPUT_DIR_3, f"{img_1_name}_{img_2_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Normalize the pixels to [0, 1]
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    height, width = img_1.shape[:2]

    # Mask: left half is 1, right half is 0
    mask = np.zeros((height, width), dtype=np.float64)
    mask[:, :width // 2] = 1.0

    levels = 6  # Levels from 0 to 5
    sigma = 2.0  # Standard deviation for Gaussian filter

    gaussian_1, gaussian_2, gaussian_mask = [img_1], [img_2], [mask]
    laplacian_1, laplacian_2 = [], []

    print("\nPart 2.3")
    print(f"Creating Gaussian and Laplacian Stacks. Levels: {levels}, Initial σ = {sigma}")

    # Create Gaussian Stack (for image and mask)
    # G_0 = I, G_1 = G_0 * G(σ), G_2 = G_1 * G(2σ), ...
    for level in range(1, levels):
        gaussian_1.append(scipy.ndimage.gaussian_filter(gaussian_1[-1], (sigma, sigma, 0)))
        gaussian_2.append(scipy.ndimage.gaussian_filter(gaussian_2[-1], (sigma, sigma, 0)))
        gaussian_mask.append(scipy.ndimage.gaussian_filter(gaussian_mask[-1], sigma))
        sigma *= 2.0

    # Create Laplacian Stack
    # L_5 = G_5, L_4 = G_4 - G_5, L_3 = G_3 - G_4, ...
    for level in range(levels - 1):
        laplacian_1.append(gaussian_1[level] - gaussian_1[level + 1])
        laplacian_2.append(gaussian_2[level] - gaussian_2[level + 1])
    laplacian_1.append(gaussian_1[-1])
    laplacian_2.append(gaussian_2[-1])

    print("\nPart 2.4")

    # Masked images
    img_1_masked_path = os.path.join(output_dir, f"{img_1_name}_masked.png")
    plt.imsave(img_1_masked_path, img_1 * mask[..., None])
    print(f"Masked {img_1_name} saved to: {img_1_masked_path}")

    img_2_masked_path = os.path.join(output_dir, f"{img_2_name}_masked.png")
    plt.imsave(img_2_masked_path, img_2 * (1.0 - mask[..., None]))
    print(f"Masked {img_2_name} saved to: {img_2_masked_path}")

    # Blend the images from Laplacian Stack by level, using Gaussian masks
    left, right, blended = [], [], []

    for level in range(levels):
        weight = gaussian_mask[level][..., None]

        img_left = weight * laplacian_1[level]
        left.append(img_left)

        img_right = (1.0 - weight) * laplacian_2[level]
        right.append(img_right)

        img_blended = img_left + img_right
        blended.append(img_blended)

        # Normalize the pixels to [0, 1]
        images = [img_left, img_right, img_blended]
        if level < levels - 1:
            pixel_max = max(np.max(img) for img in images)
            pixel_min = min(np.min(img) for img in images)

            if pixel_max > pixel_min:
                images = [(img - pixel_min) / (pixel_max - pixel_min) for img in images]
            else:
                images = [np.zeros_like(img) for img in images]

        images = [np.clip(img, 0, 1) for img in images]

        # Left part
        left_path = os.path.join(output_dir, f"{img_1_name}_{level}.png")
        plt.imsave(left_path, images[0])
        print(f"{img_1_name} for level {level} saved to: {left_path}")

        # Right part
        right_path = os.path.join(output_dir, f"{img_2_name}_{level}.png")
        plt.imsave(right_path, images[1])
        print(f"{img_2_name} for level {level} saved to: {right_path}")

        # Blended image
        blended_path = os.path.join(output_dir, f"blended_{level}.png")
        plt.imsave(blended_path, images[2])
        print(f"Blended image for level {level} saved to: {blended_path}")

    # Sum up all the levels in Laplacian Stack to get the final images
    # I = G_0 = (G_0 - G_1) + (G_1 - G_2) + ... + G_5 = L_0 + L_1 + ... + L_5
    result_1 = np.sum(left, axis=0)
    result_1_path = os.path.join(output_dir, f"{img_1_name}_result.png")
    plt.imsave(result_1_path, np.clip(result_1, 0, 1))
    print(f"Final {img_1_name} image saved to: {result_1_path}")

    result_2 = np.sum(right, axis=0)
    result_2_path = os.path.join(output_dir, f"{img_2_name}_result.png")
    plt.imsave(result_2_path, np.clip(result_2, 0, 1))
    print(f"Final {img_2_name} image saved to: {result_2_path}")

    result = np.sum(blended, axis=0)
    result_path = os.path.join(output_dir, f"blended_result.png")
    plt.imsave(result_path, np.clip(result, 0, 1))
    print(f"Final blended image saved to: {result_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for running the code of Part 2.")

    parser.add_argument(
        "--section",
        type=int,
        choices=(1, 2, 3),
        required=True,
        help="Section to run (2.1/2.2/2.3&2.4)",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="One image for section 1; two images for sections 2 and 3",
    )

    args = parser.parse_args()

    input_expected = 1 if args.section == 1 else 2
    if len(args.input) != input_expected:
        parser.error(f"Error: Section {args.section} requires exactly {input_expected} input image(s).")

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

    # Part 2.3 & 2.4: Gaussian and Laplacian Stacks, Multiresolution Blending
    elif args.section == 3:
        if not os.path.exists(OUTPUT_DIR_3):
            os.makedirs(OUTPUT_DIR_3)

        with print_logs(os.path.join(OUTPUT_DIR_3, f"part2_3_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")):
            print("Executing Part 2.3 & 2.4: Gaussian and Laplacian Stacks, Multiresolution Blending")
            section_3_4(args)

    else:
        raise argparse.ArgumentError(None, "Error: Invalid section number. Please input 1, 2 or 3.")


if __name__ == "__main__":
    main()
