# CS 180 Project 4

## Image Warping and Mosaicing

**Name:** Jiankang Chang

**Student ID:** 3043265878

**Date:** October 31, 2026

### File Structure

```
submission/
├── code/
│   ├── parta.py
│   ├── partb.py  # coming soon
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

#### 2. Install all the required dependencies

```
pip install -r requirements.txt
```

#### 3. Prepare the input images

Input images should be **in pairs**. Each pair of images should have a **projective** relationship with **the same** center of projection.

#### 4. Configure the program

#### Part A

There is a configuration center at the beginning of `parta.py` file. You may configure the program on:

- **(1)** Input image paths.
- **(2)** Coordinates of correspondence points. **Same** number of points for each pair.
- **(3)** Output directories for each section.

#### 5. Run the program

#### Section A.2

```
python code/parta.py --section 2
```

#### Section A.3

```
python code/parta.py --section 3
```

#### Section A.4

```
python code/parta.py --section 4
```

#### Part B

```
coming soon
```

#### 6. Check the output files

After the program finishes running, you may find the output files in your specified output directories.