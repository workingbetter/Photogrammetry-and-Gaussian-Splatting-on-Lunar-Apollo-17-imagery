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

---

## Part A: Photogrammetry and Gaussian Splatting

### Step 1: Photogrammetry with Agisoft Metashape

1. **Import Images**:
   - Open Metashape and create a new project.
   - Click `Workflow > Add Photos` and select the 15 Apollo 17 images.

2. **Align Photos**:
   - Click `Workflow > Align Photos`.
   - Use **High** accuracy and enable **Generic preselection**.
   - This creates a sparse point cloud and positions the cameras.

3. **Build Point Cloud**:
   - Click `Workflow > Build Point Cloud`.
   - Set **Quality** to **Medium** and enable **Calculate point confidence**.
   - This generates a detailed 3D point cloud.

4. **Build Texture**:
   - Click `Workflow > Build Texture` to add photo textures to the mesh.

5. **Export Dense Point Cloud**:
   - Go to `File > Export > Export Points` and save as `original_dense_cloud.ply`.

---

### Step 2: Simulate Gaussian Splatting and Evaluate Views

Since Gaussian splatting isn't directly available in Metashape, we simulate it by rendering views from the original camera positions and comparing them to the original images.

1. **Render Views in Metashape**:
   - Go to `Files > export > Render Images`.
   - Select each of the 15 original camera positions and render the images.
   - Save them as `image1.jpg` to `image15.jpg`.

2. **Evaluate with PSNR and SSIM**:
   - Use the following Python script to compare the original and rendered images.

```python
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# List of original and rendered image paths (adjust as needed)
original_images = [f'/content/AS17-137-2090{i}HR.png' for i in range(3, 18)]  # 15 images
rendered_images = [f'/content/rendered_view{i}.png' for i in range(1, 16)]

psnr_values = []
ssim_values = []

for orig_path, rend_path in zip(original_images, rendered_images):
    original = cv2.imread(orig_path)
    rendered = cv2.imread(rend_path)
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    rendered_gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)
    
    psnr_val = psnr(original, rendered)
    ssim_val, _ = ssim(original_gray, rendered_gray, full=True)
    
    psnr_values.append(psnr_val)
    ssim_values.append(ssim_val)

avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

print("Part A - Image Comparison (Averaged over 15 views):")
print(f"Average PSNR: {avg_psnr:.2f} dB")
print(f"Average SSIM: {avg_ssim:.3f}")
```

**Explanation**:
- **PSNR** (Peak Signal-to-Noise Ratio): Higher values indicate better image quality.
- **SSIM** (Structural Similarity Index): Values closer to 1 indicate higher similarity.

---

## Part B: Enhancing Photogrammetry with Gaussian Splatting

### Step 1: Generate Novel Views

1. **Define New Camera Poses**:
   - In Metashape, manually add 10 new camera positions (e.g., by adjusting existing cameras or using interpolation).
   - Alternatively, use a tool like COLMAP to estimate new poses.

2. **Render Novel Views**:
   - Use `Tools > Render Photos` to generate images from the new poses.
   - Save them as `novel_view1.png` to `novel_view10.png`.

3. **Assess Sharpness**:
   - Evaluate the sharpness of these novel views using the Laplacian variance.

```python
# Assess sharpness of novel views
novel_image_paths = [f'/content/novel_view{i}.png' for i in range(1, 11)]
sharpness_values = []

for novel_path in novel_image_paths:
    novel = cv2.imread(novel_path, 0)  # Grayscale
    laplacian = cv2.Laplacian(novel, cv2.CV_64F)
    sharpness = np.var(laplacian)
    sharpness_values.append(sharpness)

avg_sharpness = np.mean(sharpness_values)
print("\nPart B - Novel View Sharpness (Averaged over 10 views):")
print(f"Average Sharpness: {avg_sharpness:.2f}")
```

**Explanation**:
- **Sharpness**: Higher variance indicates sharper images, suggesting better quality for photogrammetry.

---

### Step 2: Augmented Photogrammetry

1. **Combine Images**:
   - Create a new Metashape project and import all 25 images (15 original + 10 novel).

2. **Repeat Photogrammetry**:
   - Follow the same steps as in Part A:
     - Align photos.
     - Build dense point cloud.
     - Build mesh and texture (optional).

3. **Export New Point Cloud**:
   - Save the new dense point cloud as `augmented_dense_cloud.ply`.

---

### Step 3: Compare Original and Augmented Models

Use the Hausdorff distance to quantitatively compare the original and augmented point clouds.

```python
import open3d as o3d

# Load point clouds
original_pc = o3d.io.read_point_cloud('/content/original_dense_cloud.ply')
augmented_pc = o3d.io.read_point_cloud('/content/augmented_dense_cloud.ply')

# Compute Hausdorff distance
dist = o3d.geometry.PointCloud.compute_point_cloud_distance(original_pc, augmented_pc)
hausdorff_dist = max(dist)

print("\nPart B - Point Cloud Comparison:")
print(f"Hausdorff Distance between models: {hausdorff_dist:.4f} units")
```

**Explanation**:
- **Hausdorff Distance**: A smaller distance indicates that the augmented model is similar to the original, meaning the novel views did not degrade the model quality.

---

## Conclusion

This assignment demonstrates the integration of photogrammetry and Gaussian splatting for 3D reconstruction and view synthesis. In **Part A**, we created a 3D model from Apollo 17 images and evaluated the quality of reconstructed views using PSNR and SSIM. In **Part B**, we generated novel views, assessed their sharpness, and used them to enhance the photogrammetry model, comparing the results with the original model.

### Key Findings
- **Part A**: The average PSNR and SSIM values indicate how well the rendered views match the original images.
- **Part B**: The sharpness of novel views and the Hausdorff distance between point clouds show the impact of adding synthetic views to the photogrammetry process.

This workflow can be extended to other datasets or refined with true Gaussian splatting implementations for more accurate view synthesis.

---

## Repository Structure

- `README.md`: This file, containing the assignment solution.
- `evaluation.py`: Python script for image and point cloud evaluation (consolidate the code snippets above).
- `original_dense_cloud.ply`: Exported dense point cloud from Part A.
- `augmented_dense_cloud.ply`: Exported dense point cloud from Part B.
- `images/`: Folder containing original, rendered, and novel view images.

---

## Running the Code

1. Upload the required images and PLY files to Google Colab.
2. Adjust the file paths in the Python scripts to match your Colab environment.
3. Run the Python cells to compute the metrics.

Alternatively, you can run the scripts locally if you have the necessary environment set up.
