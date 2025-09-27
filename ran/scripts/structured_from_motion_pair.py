import os
import cv2
import json

from classes.plot import Plot
from classes.imagemisc import  ImageMisc
from classes.featureDetector import FeatureDetector
from classes.superimage import SuperImage
from classes.superimagepair import SuperImagePair
from classes.camera import SimplePinholeCamera
from classes.data import Data


def StructedFromMotionPair(imag1Path, imag2Path, save_plot_dir="output", show_plot=False, show_plot_3d=False,
                           fx=None, fy=None, cx=None, cy=None):
    """
    Run Structure-from-Motion pipeline on a pair of images.

    Parameters:
        imag1Path (str): Path to first image.
        imag2Path (str): Path to second image.
        save_plot_dir (str): Directory where output images will be saved.
        show_plot (bool): If True, plots will be shown in interactive mode.
        fx, fy (float): Optional focal lengths (pixels). If None, estimated from image size.
        cx, cy (float): Optional principal point coordinates (pixels). If None, image center used.
    """
    os.makedirs(save_plot_dir, exist_ok=True)

    # 1. Load images
    paths = [imag1Path, imag2Path]
    imags_color = ImageMisc.load_images(paths)
    Plot.plot_images_grid(imags_color, 1, 2, (15, 10),
                          save_path=os.path.join(save_plot_dir, "01_color_images.png"),
                          show=show_plot)

    # 2. Grayscale
    imags_gray = list(map(ImageMisc.to_grayscale, imags_color))
    Plot.plot_images_grid(imags_gray, 1, 2, (15, 10),
                          save_path=os.path.join(save_plot_dir, "02_gray_images.png"),
                          show=show_plot)

    # 3. Feature detection
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
    Plot.plot_images_grid(imags_w_keypoints, 1, 2, (15, 10),
                          save_path=os.path.join(save_plot_dir, "03_keypoints.png"),
                          show=show_plot)

    # 4. Matching
    si1, si2 = SUPERIMAGES
    sip = SuperImagePair(si1, si2)
    sip.set_matcher(cv2.BFMatcher(cv2.NORM_L2, crossCheck=False))
    sip.match()

    imag_w_matches = sip.imag_w_matches(kind='good-matches')
    Plot.plot_images_grid([imag_w_matches], 1, 1, (15, 10),
                          save_path=os.path.join(save_plot_dir, "04_good_matches.png"),
                          show=show_plot)

    # 5. Fundamental matrix + epipolar geometry
    sip.estimate_fundamental_matrix(ransacReprojThreshold=0.25, confidence=0.99)
    fundamental_results = sip.evaluate_fundamental_estimation_quality()

    imag1_epipolar, imag2_epipolar = sip.imag_w_epipolar(num_points=200, scale=2.5, point_size=20)
    Plot.plot_images_grid([imag1_epipolar, imag2_epipolar], 1, 2, (15, 10),
                          save_path=os.path.join(save_plot_dir, "05_epipolar_lines.png"),
                          show=show_plot)

    # 6. Camera intrinsics
    imag_gray = next(iter(imags_gray))
    imag_H, imag_W = imag_gray.shape

    came_fx = fx if fx is not None else max(imag_H, imag_W)
    came_fy = fy if fy is not None else max(imag_H, imag_W)
    came_cx = cx if cx is not None else imag_W / 2
    came_cy = cy if cy is not None else imag_H / 2

    camera = SimplePinholeCamera(fx=came_fx, fy=came_fy, cx=came_cx, cy=came_cy)
    sip.set_intrinsic(camera.K())

    # 7. Essential matrix + pose + 3D reconstruction
    sip.estimate_essential_matrix()
    essential_results = sip.evaluate_essential_estimation_quality()
    sip.estimate_pose()
    sip.estimate_points3d()

    # Save evaluation results to JSON
    eval_results = {
        "fundamental_matrix": fundamental_results,
        "essential_matrix": essential_results
    }

    json_path = os.path.join(save_plot_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    R1, t1 = sip.get_camera_1_pose()
    R2, t2 = sip.get_camera_2_pose()
    points3d = sip.get_points3d()
    points3d_color = sip.get_points3d_colors()

    Plot.plot_cameras_frustum(
        [(R1, t1), (R2, t2)], points3d, points3d_size=15,
        save_path=os.path.join(save_plot_dir, "06_3d_points.png"),
        show=show_plot
    )

    Plot.plot_cameras_frustum(
        [(R1, t1), (R2, t2)], points3d, points3d_color, points3d_size=15,
        save_path=os.path.join(save_plot_dir, "07_3d_points_color.png"),
        show=show_plot_3d
    )

    return sip


DATA = {
    'casa' : Data.get_casa,
    'chaleira' : Data.get_chaleira,
    'banana' : Data.get_banana,
    'banana2' : Data.get_banana2,
    'banana3': Data.get_banana3,
    'cubo': Data.get_cubo,
    'rosto': Data.get_rosto,
    'cachorro': Data.get_cachorro,
    'celula': Data.get_celula
}

if __name__ == '__main__':
    dataset = 'rosto'
    paths = DATA[dataset](2)
    print(paths)

    imag1Path, imag2Path = next(iter(zip(paths[:-1], paths[1:])))
    superimagepair = StructedFromMotionPair(imag1Path, imag2Path, save_plot_dir=f"output/sfmp_{dataset}", show_plot=False, show_plot_3d=True)


    # fx = 2.59530310e+03
    # fy = 2.57686065e+03
    # cx = 8.15407092e+02
    # cy = 81447017e+02
    # superimagepair = StructedFromMotionPair(imag1Path, imag2Path, save_plot_dir=f"output/sfmp_{dataset}_cal", show_plot=False, show_plot_3d=True,
    #                                         fx=fx, fy=fy, cx=cx, cy=cy)


    Plot.show()