import os
import subprocess
import glob
import numpy as np

def run_colmap_pipeline(image_folder, output_folder, colmap_path="colmap"):
    """
    Run the complete COLMAP pipeline from feature extraction to dense reconstruction.
    """

    # Create output directories if they don't exist
    sparse_folder = os.path.join(output_folder, "sparse")
    dense_folder = os.path.join(output_folder, "dense")
    database_path = os.path.join(output_folder, "database.db")
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(sparse_folder, exist_ok=True)
    os.makedirs(dense_folder, exist_ok=True)
    
    # Step 1: Feature extraction
    print("Step 1: Feature extraction")
    feature_cmd = [
        colmap_path, "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_folder,
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "1"
    ]
    print(' '.join(feature_cmd))
    
    try:
        subprocess.run(feature_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Feature extraction failed: {e}")
        return False
    
    # Step 2: Match features
    print("Step 2: Feature matching")
    match_cmd = [
        colmap_path, "exhaustive_matcher",
        "--database_path", database_path,
        "--FeatureMatching.use_gpu", "1"
    ]
    
    try:
        subprocess.run(match_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Feature matching failed: {e}")
        return False
    
    # Step 3: Sparse reconstruction (Structure from Motion)
    print("Step 3: Sparse reconstruction")
    sfm_cmd = [
        colmap_path, "mapper",
        "--database_path", database_path,
        "--image_path", image_folder,
        "--output_path", sparse_folder
    ]
    
    try:
        subprocess.run(sfm_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Sparse reconstruction failed: {e}")
        return False
    
    # Find the largest sparse model
    sparse_models = glob.glob(os.path.join(sparse_folder, "*/"))
    if not sparse_models:
        print("No sparse models found")
        return False
    
    # Sort by model size (using number of images as proxy)
    largest_model = 0
    max_images = 0
    for i, model_dir in enumerate(sparse_models):
        images_txt = os.path.join(model_dir, "images.txt")
        if os.path.exists(images_txt):
            with open(images_txt, 'r') as f:
                num_images = sum(1 for line in f if line.strip() and not line.startswith("#"))
                num_images = num_images // 2  # Each image has 2 lines
                if num_images > max_images:
                    max_images = num_images
                    largest_model = i
    
    selected_model = os.path.join(sparse_folder, str(largest_model))
    print(f"Selected model {largest_model} with {max_images} images")
    
    # Step 4: Image undistortion
    print("Step 4: Image undistortion")
    undistort_cmd = [
        colmap_path, "image_undistorter",
        "--image_path", image_folder,
        "--input_path", selected_model,
        "--output_path", dense_folder,
        "--output_type", "COLMAP"
    ]
    
    try:
        subprocess.run(undistort_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Image undistortion failed: {e}")
        return False
    
    # Step 5: Dense reconstruction (Multi-View Stereo)
    print("Step 5: Dense reconstruction")
    mvs_cmd = [
        colmap_path, "patch_match_stereo",
        "--workspace_path", dense_folder,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true"
    ]
    
    try:
        subprocess.run(mvs_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Dense reconstruction failed: {e}")
        return False
    
    # Step 6: Stereo fusion
    print("Step 6: Stereo fusion")
    fusion_cmd = [
        colmap_path, "stereo_fusion",
        "--workspace_path", dense_folder,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", os.path.join(dense_folder, "fused.ply")
    ]
    
    try:
        subprocess.run(fusion_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Stereo fusion failed: {e}")
        return False
    
    print("Pipeline completed successfully!")
    return True

if __name__ == '__main__':
    # Replace with your image and output folder paths
    image_folder = "data/scan6_max"
    output_folder = "output_colmap"
        
    # Path to COLMAP executable (may be just "colmap" if it's in your PATH)
    colmap_path = "colmap"
        
    run_colmap_pipeline(image_folder, output_folder, colmap_path)