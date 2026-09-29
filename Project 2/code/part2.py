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
    pass










# =============== Part 2.2: Hybrid Images ===============

def section_2(args) -> None:
    pass










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
        required=True,
        help="Path to the input image",
    )

    return parser.parse_args()


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