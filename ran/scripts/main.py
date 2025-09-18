import cv2
import numpy as np
from scipy.spatial import cKDTree


from classes.plot import Plot
from classes.imagemisc import  ImageMisc
from classes.featureDetector import FeatureDetector
from classes.superimage import SuperImage
from classes.superimagepair import SuperImagePair
from classes.camera import SimplePinholeCamera
from classes.sfmGlobal import SfmGlobal
from classes.quality_analyzer import analyze_reconstruction_quality


def remove_nearby_points(points, points_color=None, threshold=0.01):
    """
    Remove points that are closer than `threshold` to any other point.
    Also removes corresponding entries in points_color if provided.

    Args:
        points: np.ndarray of shape (N,3)
        points_color: optional np.ndarray of shape (N,3) corresponding RGB colors
        threshold: float, minimum allowed distance between points

    Returns:
        filtered_points: np.ndarray of filtered points (M,3)
        filtered_colors: np.ndarray of corresponding colors (M,3) if points_color is provided,
                         otherwise None
    """
    if len(points) == 0:
        return points, points_color if points_color is not None else None

    tree = cKDTree(points)
    to_keep = np.ones(len(points), dtype=bool)

    for i, point in enumerate(points):
        if not to_keep[i]:
            continue
        # Find all neighbors within threshold (including self)
        neighbors = tree.query_ball_point(point, threshold)
        neighbors.remove(i)  # remove self
        to_keep[neighbors] = False  # remove all neighbors that are too close

    filtered_points = points[to_keep]

    if points_color is not None:
        filtered_colors = points_color[to_keep]
    else:
        filtered_colors = None

    return filtered_points, filtered_colors


def remove_outliers_std(points, points_color=None, n_std=2.0):
    """
    Remove points that are farther than `n_std` standard deviations from the centroid.
    Optionally removes corresponding entries in points_color.

    Args:
        points: np.ndarray of shape (N,3)
        points_color: optional np.ndarray of shape (N,3) corresponding RGB colors
        n_std: float, number of standard deviations to keep

    Returns:
        filtered_points: np.ndarray of filtered points (M,3)
        filtered_colors: np.ndarray of corresponding colors (M,3) if points_color is provided,
                         otherwise None
    """
    if len(points) == 0:
        return points, points_color if points_color is not None else None

    centroid = points.mean(axis=0)
    std_dev = points.std(axis=0)

    # Keep points within n_std in all axes
    lower_bound = centroid - n_std * std_dev
    upper_bound = centroid + n_std * std_dev
    mask = np.all((points >= lower_bound) & (points <= upper_bound), axis=1)

    filtered_points = points[mask]
    filtered_colors = points_color[mask] if points_color is not None else None

    return filtered_points, filtered_colors


def StructedFromMotionPair(imag1Path, imag2Path, verbose):
    # paths = ImageMisc.get_paths(ROOT_DIR_IMAGES, '*max.png')

    paths = [imag1Path, imag2Path]
    imags_color = ImageMisc.load_images(paths)
    # Plot.plot_images_grid(imags_color, 1, 2, (15, 10))

    imags_gray = list(map(ImageMisc.to_grayscale, imags_color))
    # Plot.plot_images_grid(imags_gray, 1, 2, (15, 10))

    FD = FeatureDetector(kind='sift')
    SUPERIMAGES = []
    for imag_color, imag_gray in zip(imags_color, imags_gray):
        keypoints, descriptors = FD.detect(imag_gray)
        si = SuperImage()
        si.set_imag_color(imag_color)
        si.set_imag_gray(imag_gray)
        si.set_keypoints(keypoints)
        si.set_descriptors(descriptors)
        SUPERIMAGES.append(si)

    imags_w_keypoints = [si.imag_w_keypoints() for si in SUPERIMAGES]
    # Plot.plot_images_grid(imags_w_keypoints, 1, 2, (15, 10))

    si1 = SUPERIMAGES[0]
    si2 = SUPERIMAGES[1]
    sip = SuperImagePair(si1, si2)

    sip.set_matcher(
        cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    )
    sip.match()

    imag_w_matches = sip.imag_w_matches(kind='good-matches')
    # Plot.plot_images_grid([imag_w_matches], 1, 1, (15, 10))

    sip.estimate_fundamental_matrix(ransacReprojThreshold=0.25, confidence=0.99)
    sip.evaluate_fundamental_estimation_quality()

    imag1_epipolar, imag2_epipolar = sip.imag_w_epipolar(num_points=200, scale=2.5, point_size=20)
    # Plot.plot_images_grid([imag1_epipolar, imag2_epipolar], 1, 2, (15, 10))

    imag_gray = next(iter(imags_gray))
    imag_H, imag_W = imag_gray.shape
    came_f = max(imag_H, imag_W)
    came_cx = imag_W/2
    came_cy = imag_H/2
    camera = SimplePinholeCamera(f=came_f, cx=came_cx, cy=came_cy)

    sip.set_intrinsic((camera.K()))

    sip.estimate_essential_matrix()
    sip.evaluate_essential_estimation_quality()

    sip.estimate_pose()
    sip.estimate_points3d()

    R1, t1 = sip.get_camera_1_pose()
    R2, t2 = sip.get_camera_2_pose()
    points3d = sip.get_points3d()
    points3d_color = sip.get_points3d_colors()

    if verbose:
        Plot.plot_cameras_frustum([(R1, t1), (R2, t2)], points3d, points3d_size=15)
        Plot.plot_cameras_frustum([(R1, t1), (R2, t2)], points3d, points3d_color, points3d_size=15)


    return sip

def StructedFromMotionSequential(SUPERIMAGEPAIRs, verbose):
    sfm = SfmGlobal(SUPERIMAGEPAIRs)
    camera_poses, points3d , points3d_color = sfm.run()
    if verbose : Plot.plot_cameras_frustum(camera_poses, points3d)

    return camera_poses, points3d, points3d_color



if __name__ == '__main__':
    # ROOT_DIR_IMAGES = '../SampleSet/MVS Data/scan6_2_1'
    # paths = ImageMisc.get_paths(ROOT_DIR_IMAGES, '*max.png')
    # imag1Path, imag2Path = next(iter(zip(paths[:-1], paths[1:])))
    # superimagepair = StructedFromMotionPair(imag1Path, imag2Path, verbose=True)

    ROOT_DIR_IMAGES = '../SampleSet/MVS Data/scan6_7_1'
    paths = ImageMisc.get_paths(ROOT_DIR_IMAGES, '*max.png')
    SUPERIMAGEPAIRs = []
    for imag1Path, imag2Path in zip(paths[:-1], paths[1:]):
        superimagepair = StructedFromMotionPair(imag1Path, imag2Path, verbose=False)
        SUPERIMAGEPAIRs.append(superimagepair)

    camera_poses, points3d, points3d_color = StructedFromMotionSequential(SUPERIMAGEPAIRs, verbose=False)

    print(f"\nOriginal reconstruction: {len(points3d)} 3D points")

    # Optional: Apply filtering to reduce noise
    points3d, points3d_color = remove_nearby_points(points3d, points3d_color, threshold=0.05)
    points3d, points3d_color = remove_outliers_std(points3d, points3d_color, n_std=2.0)
    
    print(f"After filtering: {len(points3d)} 3D points")

    # Conservative densification to prevent error propagation
    density_factor = 1.2  # Very conservative factor to minimize distortion
    
    print(f"Using conservative density factor: {density_factor:.1f}")
    
    # Import densifier for direct testing
    from classes.point_densifier import densify_point_cloud
    
    # Test conservative method
    print("Testing Conservative method (error-aware)...")
    conservative_points, conservative_colors = densify_point_cloud(
        points3d, points3d_color,
        method='conservative',
        density_factor=density_factor,
        max_points=8000
    )
    
    # GPU-accelerated Open3D visualization
    print("\n=== GPU-Accelerated Visualization Sequence ===")
    print("You will see 3 different visualizations:")
    print("1. Original sparse point cloud (no surface)")
    print("2. Original sparse points with Poisson surface") 
    print("3. Conservative densified points with Poisson surface")
    print("\nEach window will open separately. Close each one to proceed to the next.")
    
    input("\nPress Enter to start visualization sequence...")
    
    # Option 1: Show original sparse point cloud (for reference)
    print("\n1. Showing original sparse point cloud...")
    Plot.show_point_cloud_gpu(camera_poses, points3d, points3d_color)
    
    # Option 2: Show original sparse points with Poisson surface
    print("\n2. Showing original sparse Poisson surface...")
    Plot.show_poisson_surface_gpu(
        camera_poses, points3d, points3d_color,
        densify=False,  # Original sparse points
        poisson_depth=5,  # Lower depth for sparse points
        use_gpu=True
    )
    
    # Option 3: Show conservative densification with Poisson surface
    print("\n3. Showing conservative densified Poisson surface...")
    Plot.show_poisson_surface_gpu(
        camera_poses, conservative_points, conservative_colors,
        densify=False,  # Points are already densified
        poisson_depth=6,  # Higher detail for densified points
        use_gpu=True
    )
    
    print("\n=== Visualization Complete ===")
    print(f"Summary:")
    print(f"- Original points: {len(points3d)}")
    print(f"- Conservative densified points: {len(conservative_points)}")
    print(f"- Density increase: {len(conservative_points)/len(points3d):.2f}x")
    
    