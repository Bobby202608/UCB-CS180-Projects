import argparse
import cv2
import datetime
import numpy as np
import os
import scipy
import sys
import time

from utils import print_logs


OUTPUT_DIR_1 = "output/part1_1"
OUTPUT_DIR_2 = "output/part1_2"
OUTPUT_DIR_3 = "output/part1_3"

# =============== Part 1.1: Convolutions from Scratch ===============

KERNELS = {
    "box_9x9": np.ones((9, 9), dtype=np.float64) / 81.0,  # Dimension: 9 x 9
    "D_x": np.array([[1.0, 0.0, -1.0]]),                  # Dimension: 1 x 3
    "D_y": np.array([[1.0], [0.0], [-1.0]]),              # Dimension: 3 x 1
}


# a. Convolution with 4 for-loops:
def convolve_4_loops(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k_height, k_width = kernel.shape

    # Flip the kernel
    kernel_flip = np.flip(kernel, axis=(0, 1))

    # Change to the original size
    result = np.zeros((img.shape[0] - k_height + 1, img.shape[1] - k_width + 1), dtype=np.float64)

    for row in range(result.shape[0]):
        for col in range(result.shape[1]):
            for i in range(k_height):
                for j in range(k_width):
                    result[row, col] += img[row + i, col + j] * kernel_flip[i, j]

    return result


# b. Convolution with 2 for-loops:
def convolve_2_loops(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k_height, k_width = kernel.shape

    # Flip the kernel
    kernel_flip = np.flip(kernel, axis=(0, 1))

    # Change to the original size
    result = np.zeros((img.shape[0] - k_height + 1, img.shape[1] - k_width + 1), dtype=np.float64)

    for row in range(result.shape[0]):
        for col in range(result.shape[1]):
            window = img[row:row + k_height, col:col + k_width]
            result[row, col] = np.sum(window * kernel_flip)

    return result


def section_1(args: argparse.Namespace) -> None:
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"\nError: Failed to read the image: {args.input}")

    height, width = image.shape
    print(f"\nImage shape: {height} x {width}")

    box_kernel = KERNELS["box_9x9"]
    k_height, k_width = box_kernel.shape
    print(f"Box kernel shape: {k_height} x {k_width}")

    # Use "same" padding
    padding = ((k_height // 2, (k_height - 1) // 2), (k_width // 2, (k_width - 1) // 2))
    image_padded = np.pad(image, padding, mode="constant")
    print(f"Padded image shape: {image_padded.shape[0]} x {image_padded.shape[1]}")

    # a. Convolution with 4 for-loops
    print("\na. Convolution with 4 for-loops:")

    start_time = time.perf_counter()
    result_a = convolve_4_loops(image_padded, box_kernel)
    end_time = time.perf_counter()

    print(f"Execution time: {end_time - start_time:.4f} s")

    result_a_path = os.path.join(OUTPUT_DIR_1, "result_a.png")
    cv2.imwrite(result_a_path, result_a)
    print(f"Result image saved to: {result_a_path}")

    # b. Convolution with 2 for-loops
    print("\nb. Convolution with 2 for-loops:")

    start_time = time.perf_counter()
    result_b = convolve_2_loops(image_padded, box_kernel)
    end_time = time.perf_counter()

    print(f"Execution time: {end_time - start_time:.4f} s")

    result_b_path = os.path.join(OUTPUT_DIR_1, "result_b.png")
    cv2.imwrite(result_b_path, result_b)
    print(f"Result image saved to: {result_b_path}")

    # c. Directly use scipy.signal.convolve2d
    print("\nc. Directly use scipy.signal.convolve2d:")

    start_time = time.perf_counter()
    result_c = scipy.signal.convolve2d(image_padded, box_kernel, mode="valid")
    end_time = time.perf_counter()

    print(f"Execution time: {end_time - start_time:.4f} s")

    result_c_path = os.path.join(OUTPUT_DIR_1, "result_c.png")
    cv2.imwrite(result_c_path, result_c)
    print(f"Result image saved to: {result_c_path}")


# =============== Part 1.2: Finite Difference Operator ===============
def section_2(args) -> None:
    D_x = KERNELS["D_x"]
    D_y = KERNELS["D_y"]
    pass










# =============== Part 1.3: Derivative of Gaussian Filter ===============
def section_3(args) -> None:
    pass










def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for running the code of Part 1.")

    parser.add_argument(
        "--section",
        type=int,
        choices=(1, 2, 3),
        required=True,
        help="Section to run (1.1/1.2/1.3)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input image",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Part 1.1: Convolutions from Scratch
    if args.section == 1:
        if not os.path.exists(OUTPUT_DIR_1):
            os.makedirs(OUTPUT_DIR_1)

        with print_log(os.path.join(OUTPUT_DIR_1, f"part1_1_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")):
            print("Executing Part 1.1: Convolutions from Scratch")
            section_1(args)

    # Part 1.2: Finite Difference Operator
    elif args.section == 2:
        if not os.path.exists(OUTPUT_DIR_2):
            os.makedirs(OUTPUT_DIR_2)

        with print_log(os.path.join(OUTPUT_DIR_2, f"part1_2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")):
            print("Executing Part 1.2: Finite Difference Operator")
            section_2(args)

    # Part 1.3: Derivative of Gaussian Filter
    elif args.section == 3:
        if not os.path.exists(OUTPUT_DIR_3):
            os.makedirs(OUTPUT_DIR_3)

        with print_log(os.path.join(OUTPUT_DIR_3, f"part1_3_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")):
            print("Executing Part 1.3: Derivative of Gaussian Filter")
            section_3(args)

    else:
        raise argparse.ArgumentError(None, "Error: Invalid section number. Please input 1, 2 or 3.")


if __name__ == "__main__":
    main()