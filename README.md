# Computer Vision - 3D Image Reconstruction

## Overview

This project implements a pipeline for reconstructing 3D models from multiple 2D images using python with openCV and open3D. It automates feature extraction, image matching, sparse and dense reconstruction, and mesh generation. Additionaly, we implemented a full pipeline using the COLMAP photogrammetry software in order to obtain the dense reconstruction. The workflow is suitable for research and practical applications in computer vision, robotics, AR/VR, and digital heritage.

## Features



## Project Structure

```
.
├── colmap_pipeline.py          # Main pipeline script for COLMAP automation
├── T2.ipynb                    # Jupyter notebook for automatic 3D reconstruction
├── data/                       # Input images for reconstruction
├── output_colmap/              # Output directory for COLMAP results
│   └── dense/
│       ├── run-colmap-geometric.sh   # Script for geometric stereo fusion & meshing
│       └── run-colmap-photometric.sh # Script for photometric stereo fusion & meshing
└── .gitignore                  # Git ignore file
```
In order to install the colmap software follow this steps: [COLMAP Installation guide](https://colmap.github.io/install.html)
## Colmap pipeline
- **Automatic Feature Extraction:** Detects keypoints and descriptors in input images using COLMAP.
- **Exhaustive Feature Matching:** Uses GPU acceleration for matching features across images.
- **Sparse Reconstruction (Structure from Motion):** Recovers camera poses and a sparse point cloud.
- **Image Undistortion:** Corrects image distortions for accurate dense reconstruction.
- **Dense Reconstruction (Multi-View Stereo):** Generates detailed depth maps.
- **Stereo Fusion:** Fuses depth information into a dense point cloud (`fused.ply`).
- 
## Colmap pipeline Workflow

The main pipeline script, `colmap_pipeline.py`, performs the following steps:

1. **Feature Extraction**
    - Extracts visual features from images.
    - Example command:
      ```
      colmap feature_extractor --database_path <db> --image_path <images> --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1 --FeatureExtraction.use_gpu 1
      ```
2. **Feature Matching**
    - Matches features between all image pairs.
      ```
      colmap exhaustive_matcher --database_path <db> --FeatureMatching.use_gpu 1
      ```
3. **Sparse Reconstruction**
    - Recovers camera poses and a sparse point cloud.
      ```
      colmap mapper --database_path <db> --image_path <images> --output_path <sparse>
      ```
4. **Image Undistortion**
    - Prepares images for dense reconstruction.
      ```
      colmap image_undistorter --image_path <images> --input_path <sparse_model> --output_path <dense> --output_type COLMAP
      ```
5. **Dense Reconstruction**
    - Generates detailed depth maps.
      ```
      colmap patch_match_stereo --workspace_path <dense> --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
      ```
6. **Stereo Fusion**
    - Fuses depth maps to create a dense point cloud.
      ```
      colmap stereo_fusion --workspace_path <dense> --workspace_format COLMAP --input_type geometric --output_path <dense>/fused.ply
      ```

### Mesh Generation

After obtaining a dense point cloud, use the provided shell scripts for mesh generation:
- **Photometric Approach:** `run-colmap-photometric.sh`
- **Geometric Approach:** `run-colmap-geometric.sh`

These scripts create Poisson and Delaunay meshes for further analysis or visualization.

## Requirements

- Python 3.x
- COLMAP installed and added to your system PATH
- NumPy
- Input images placed in the `data/` directory

## Usage

1. Place your images in `data/scan6_max` (or update the path in the script).
2. Run the pipeline:
    ```bash
    python colmap_pipeline.py
    ```
3. Outputs are saved in `output_colmap/`.

## Example

```python
# In colmap_pipeline.py
if __name__ == '__main__':
    image_folder = "data/scan6_max"
    output_folder = "output_colmap"
    colmap_path = "colmap"
    run_colmap_pipeline(image_folder, output_folder, colmap_path)
```

## References

- [COLMAP documentation](https://colmap.github.io/)
- [Multi-View 3D Reconstruction](https://en.wikipedia.org/wiki/Multiview_3D_reconstruction)
