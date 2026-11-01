# CS 180 Project 1

## Images of the Russian Empire: Colorizing the Prokudin-Gorskii Photo Collection

**Name:** Jiankang Chang

**Student ID:** 3043265878

**Date:** September 15, 2026

### File Structure

```
submission/
├── code/
│   ├── main.py
│   ├── requirements.txt
│   └── README.md
└── web/
    └── page.pdf
```

### Operation Guidelines

Please follow the steps below to run the project code:

#### 1. Create and activate an Anaconda environment

```
conda create -n CS180 python=3.12
conda activate CS180
```

#### 2. Install all required dependencies

```
pip install -r requirements.txt
```

#### 3. Prepare the images

Put all images to be tested under **ONE** directory, such as `images/`.

#### 4. Configure the program

There is a "Configuration center" at the beginning of `main.py`. You may make the following configurations to the program:

- **`IMAGES_DIR`**: Directory where input images are stored.

- **`IMAGES_LIST`**: A list of all images to be tested.
  - `small`: Small images in **JPG/JPEG/PNG** format.
  - `large`: Large images in **TIF** format.
  - `extra`: Self-picked images, **regardless of** format. They will be automatically categorized after starting the program.

- **`USE_IMAGES`**: Whether to use the small/large images for experiment.

- **`DISPLACEMENT_CONFIG`**: Configurations to image displacement.
  - `border_proportion`: The proportion of border pixels ignored for scoring, in order not to exaggerate the score and mislead the alignment. **Should be in the range [0, 0.5)**.
  - `dx_min`, `dx_max`: Range of displacement along **x-axis**. Positive for right shift, negative for left shift.
  - `dy_min`, `dy_max`: Range of displacement along **y-axis**. Positive for down shift, negative for up shift.

- **`PYRAMID_CONFIG`**: Configurations to Gaussian Pyramid.
  - `levels`: Total number of the Pyramid levels, including the one for the original full-resolution image. **Should be at least 2**.
  - `scale`: Downsampling factor between adjacent levels. **Should be greater than 1**. For example, if `scale` = 2, both the image's width and height would be halved at the next level.
  - `d_refinement`: Refinement radius (in pixels) used around the best displacement estimated at the coarser Pyramid level. For example, if `d_refinement` = 2, the current level would search the best x displacement from `dx - 2` to `dx + 2` and y displacement from `dy - 2` to `dy + 2`.

- **`OUTPUT_DIR`**: Directory where output files are stored.
  - `images`: Directory of synthesized color images.
  - `csv`: Directory of a CSV file recording relevant information of each processing.

- **NOTE:** **`CSV_FIELDS`** is **NOT** editable!

#### 5. Run the program

```
python code/main.py
```

#### 6. Check the output files

After the program finishes running, you may find the synthesized color images, as well as a CSV file recording relevant information, at the directory you have specified.
