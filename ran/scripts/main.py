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


def remove_nearby_points(points, threshold):
    """
    Remove points that are closer than `threshold` to any other point.

    Args:
        points: np.ndarray of shape (N,3)
        threshold: float, minimum allowed distance between points

    Returns:
        np.ndarray of filtered points
    """
    if len(points) == 0:
        return points

    tree = cKDTree(points)
    to_keep = np.ones(len(points), dtype=bool)

    for i, point in enumerate(points):
        if not to_keep[i]:
            continue
        # Find all neighbors within threshold (including self)
        neighbors = tree.query_ball_point(point, threshold)
        neighbors.remove(i)  # remove self
        to_keep[neighbors] = False  # remove all neighbors that are too close

    return points[to_keep]


def remove_outliers_std(points, n_std):
    """
    Remove points that are farther than `n_std` standard deviations from the centroid.

    Args:
        points: np.ndarray of shape (N,3)
        n_std: float, number of standard deviations to keep

    Returns:
        np.ndarray of filtered points
    """
    if len(points) == 0:
        return points

    centroid = points.mean(axis=0)
    std_dev = points.std(axis=0)

    # Keep points within n_std in all axes
    lower_bound = centroid - n_std * std_dev
    upper_bound = centroid + n_std * std_dev

    mask = np.all((points >= lower_bound) & (points <= upper_bound), axis=1)
    filtered_points = points[mask]

    return filtered_points


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
    camera_poses, points3d = sfm.run()

     # Convert points3d to NumPy array if it's a list
    if isinstance(points3d, list):
        if len(points3d) > 0:
            points3d = np.vstack(points3d)  # Combine all point arrays
        else:
            points3d = np.empty((0, 3))  # Empty array with correct shape

    if verbose : Plot.plot_cameras_frustum(camera_poses, points3d)

    return camera_poses, points3d



if __name__ == '__main__':
    ROOT_DIR_IMAGES = '../SampleSet/MVS Data/scan6_2_1'
    paths = ImageMisc.get_paths(ROOT_DIR_IMAGES, '*max.png')
    imag1Path, imag2Path = next(iter(zip(paths[:-1], paths[1:])))
    superimagepair = StructedFromMotionPair(imag1Path, imag2Path, verbose=True)


    # ROOT_DIR_IMAGES = '../SampleSet/MVS Data/scan6_7_1'
    # paths = ImageMisc.get_paths(ROOT_DIR_IMAGES, '*max.png')
    # SUPERIMAGEPAIRs = []
    # for imag1Path, imag2Path in zip(paths[:-1], paths[1:]):
    #     superimagepair = StructedFromMotionPair(imag1Path, imag2Path, verbose=False)
    #     SUPERIMAGEPAIRs.append(superimagepair)
    #
    # camera_poses, points3d = StructedFromMotionSequential(SUPERIMAGEPAIRs, verbose=False)
    #
    # Plot.plot_cameras_frustum(camera_poses, points3d)
    #
    # points3d = remove_nearby_points(points3d, threshold=0.1)
    # Plot.plot_cameras_frustum(camera_poses, points3d)
    #
    # points3d = remove_outliers_std(points3d, n_std=2)
    # Plot.plot_cameras_frustum(camera_poses, points3d)