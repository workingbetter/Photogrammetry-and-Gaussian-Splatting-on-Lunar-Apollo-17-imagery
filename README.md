# Photogrammetry and Gaussian Splatting on Lunar Apollo 17 Imagery

This repository contains the solution to the assignment on photogrammetry and Gaussian splatting using Apollo 17 lunar imagery. The assignment is divided into two parts:

I used **Agisoft Metashape** for photogrammetry and **Python** (with libraries like OpenCV, scikit-image, and Open3D) for image and point cloud analysis. The Python code is designed to run in **Google Colab** for ease of use.

---

## Prerequisites

Before starting, ensure you have the following:
- **Agisoft Metashape** installed (trial version available).
- **Python** with the following libraries:
  - `numpy`
  - `opencv-python`
  - `scikit-image`
  - `open3d`
- **Google Colab** for running the Python scripts.
- **Apollo 17 imagery** (15 images) from the provided [Google Drive link](https://drive.google.com/drive/folders/18t2fq0a8yKQDM4BYSeuSHrAbmJujMYLO?usp=drive_link).

To install the required Python libraries, run:
```bash
pip install numpy opencv-python scikit-image open3d
```




1. Upload the required images and PLY files to Google Colab.
2. Adjust the file paths in the Python scripts to match your Colab environment.
3. Run the Python cells to compute the metrics.

Alternatively, you can run the scripts locally if you have the necessary environment set up.
