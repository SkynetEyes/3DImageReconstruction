import cv2
import numpy as np
import glob
import os

# --- Parameters ---
chessboard_size = (7, 7)   # number of inner corners per row and column
square_size = 25.0

# --- Prepare object points (0,0,0), (1,0,0), (2,0,0), ... scaled by square_size ---
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # scale to real-world units

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# --- Load your chessboard images ---
images = glob.glob("../SampleSet/Banana2/calib_images/*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        # Refine corner locations
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        # Draw and display the corners (for visualization)
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)

        scale = 0.25
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        cv2.imshow('Chessboard', img_resized)
        cv2.waitKey(0)

cv2.destroyAllWindows()

# --- Calibration ---
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Calibration RMS error:", ret)
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist.ravel())

# --- Save calibration results ---
save_dir = os.path.dirname(images[0]) if images else "."
save_path = os.path.join(save_dir, "calibration_results.txt")

with open(save_path, "w") as f:
    f.write(f"Calibration RMS error: {ret}\n\n")
    f.write("Camera matrix:\n")
    f.write(str(mtx) + "\n\n")
    f.write("Distortion coefficients:\n")
    f.write(str(dist.ravel()) + "\n\n")
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        f.write(f"Image {i}: \n")
        f.write(f"Rotation vector: {rvec.ravel()}\n")
        f.write(f"Translation vector (in mm): {tvec.ravel()}\n\n")

print(f"\nCalibration results saved to {save_path}")