import cv2
import numpy as np


class SuperImagePair():
    def __init__(self, simga1, simga2):
        self.simag1 = simga1
        self.simag2 = simga2

        self.K = None
        self.matcher = None

        self.matches = None
        self.matches_good = None
        self.matches_inlier = None

        self.F = None
        self.E = None
        self.R = None
        self.t = None
        self.points3d = None

    def set_intrinsic(self, K):
        self.K = K


    def set_matcher(self, matcher):
        self.matcher = matcher


    def match(self, ratio_thresh=0.75):
        if self.matcher is None:
            print('Matcher must be set.')
            return

        dess1 = self.simag1.descripts
        dess2 = self.simag2.descripts

        # --- Check if descriptors exist ---
        if dess1 is None or dess2 is None:
            print("Descriptors are missing.")
            return
        if len(dess1) == 0 or len(dess2) == 0:
            print("Empty descriptors.")
            return

        # --- Ensure correct dtype ---
        if dess1.dtype != np.float32 and dess1.dtype != np.uint8:
            dess1 = dess1.astype(np.float32)
        if dess2.dtype != np.float32 and dess2.dtype != np.uint8:
            dess2 = dess2.astype(np.float32)

        # --- Perform matching ---
        matches = self.matcher.knnMatch(dess1, dess2, k=2)
        matches_good = []

        for pair in matches:
            if len(pair) < 2:  # sometimes only 1 match is found
                continue
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                matches_good.append(m)

        self.matches = matches
        self.matches_good = matches_good

        print(f"Found {len(matches_good)} good matches (out of {len(matches)} total).")


    def estimate_fundamental_matrix(self, ransacReprojThreshold, confidence):
        matches_good = self.matches_good
        kpts1 = self.simag1.keypoints
        kpts2 = self.simag2.keypoints

        if matches_good is None:
            print('Must run "match" method first.')
            return

        if len(matches_good) >= 8:  # minimum for 8-point algorithm
            pts1 = np.float32([kpts1[m.queryIdx].pt for m in matches_good])
            pts2 = np.float32([kpts2[m.trainIdx].pt for m in matches_good])

            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold, confidence)

            if F is None:
                print("Fundamental matrix not found.")
                return

            # Enforce rank 2
            U, S, Vt = np.linalg.svd(F)
            S[2] = 0
            F = U @ np.diag(S) @ Vt

            inlier_count = int(mask.sum())
            ratio = inlier_count / len(matches_good)

            print("Good matches: {:06d}, Inlier matches (RANSAC): {:06d}, Ratio: {:.6f}".format(
                len(matches_good), inlier_count, ratio))

            print("Fundamental matrix F:")
            for row in F:
                formatted_row = ["{:13.6e}".format(val) for val in row]
                print("  [" + " ".join(formatted_row) + "]")

            matches_inlier = [gm for gm, m in zip(matches_good, mask.ravel()) if m == 1]

            self.F = F
            self.matches_inlier = matches_inlier  # keep naming consistent

        else:
            print("Not enough good matches to estimate F.")


    def estimate_essential_matrix(self):
        if self.F is None:
            print("Fundamental matrix must be estimated first.")
            return None
        if self.K is None:
            print("Camera intrinsics K not provided.")
            return None

        # Compute E
        self.E = self.K.T @ self.F @ self.K

        # Enforce rank-2 constraint
        U, S, Vt = np.linalg.svd(self.E)
        S[2] = 0
        self.E = U @ np.diag(S) @ Vt

        # Print nicely
        print("Essential matrix E:")
        for row in self.E:
            formatted_row = ["{:13.6e}".format(val) for val in row]
            print("  [" + " ".join(formatted_row) + "]")


    def estimate_pose(self):
        if self.E is None:
            print("Essential matrix must be estimated first.")
            return None, None
        if self.matches_inlier is None or len(self.matches_inlier) < 5:
            print("Not enough inlier matches to recover pose.")
            return None, None

        kpts1 = self.simag1.keypoints
        kpts2 = self.simag2.keypoints

        # Points in pixel coordinates
        pts1 = np.float32([kpts1[m.queryIdx].pt for m in self.matches_inlier])
        pts2 = np.float32([kpts2[m.trainIdx].pt for m in self.matches_inlier])

        # Recover pose
        _, R, t, mask = cv2.recoverPose(self.E, pts1, pts2, self.K)

        # Filter matches according to mask
        mask = mask.ravel()  # flatten to 1D
        matches_pose = [m for m, inlier in zip(self.matches_inlier, mask) if inlier != 0]

        # Store results in the object
        self.R = R # rotation of camera 2 relative to camera 1
        self.t = t # translation vector of camera 2 relative to camera 1, up to an unknown scale
        self.matches_pose = matches_pose  # matches consistent with recovered pose

        print("Recovered Pose:")
        print("Rotation R:\n", R)
        print("Translation t (up to scale):\n", t.ravel())
        print(f"Number of matches used for pose: {len(matches_pose)}")

    def estimate_points3d(self):
        """
        Triangulate 3D points from inlier matches used in pose recovery.
        Stores results in self.points3d as (N,3) array in camera1 coordinates.
        """
        if self.R is None or self.t is None:
            print("Pose not estimated yet. Run estimate_pose() first.")
            return None
        if self.matches_pose is None or len(self.matches_pose) == 0:
            print("No inlier matches for triangulation.")
            return None
        if self.K is None:
            print("Camera intrinsics not set.")
            return None

        kpts1 = self.simag1.keypoints
        kpts2 = self.simag2.keypoints

        # Points in pixel coordinates
        pts1 = np.float32([kpts1[m.queryIdx].pt for m in self.matches_pose])
        pts2 = np.float32([kpts2[m.trainIdx].pt for m in self.matches_pose])

        # Projection matrices
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])  # camera1 at origin
        P2 = self.K @ np.hstack([self.R, self.t])  # camera2 pose relative to camera1

        pts1 = pts1.T  # shape (2, N)
        pts2 = pts2.T

        # Triangulate
        points4d_h = cv2.triangulatePoints(P1, P2, pts1, pts2)  # shape (4, N)
        points3d = (points4d_h[:3] / points4d_h[3]).T  # convert to (N,3)

        self.points3d = points3d
        print(f"Triangulated {points3d.shape[0]} 3D points.")


    def get_camera_1_pose(self):
        """
        Returns the pose of camera 1 in the world frame (camera 1 frame as reference)
        """
        R1 = np.eye(3)
        t1 = np.zeros((3, 1))
        return R1, t1


    def get_camera_2_pose(self):
        """
        Returns the pose of camera 2 in the world frame (camera 1 frame as reference)
        """
        if self.R is None or self.t is None:
            print("Must run estimate_pose() first")
            return None, None

        R2 = self.R.T
        C2 = -self.R.T @ self.t  # camera center in camera 1 frame
        return R2, C2


    def get_points3d(self):
        return self.points3d

    # ---- all your other methods remain unchanged ----
    def imag_w_matches(self, kind, max_matches=0):
        if kind == 'good-matches':
            matches = self.matches_good
        elif kind == 'inlier-matches':
            matches = self.matches_inlier
        else:
            print('match_kind in {good-matches, inlier-matches}')

        imag1 = self.simag1.imag_color
        kpts1 = self.simag1.keypoints

        imag2 = self.simag2.imag_color
        kpts2 = self.simag2.keypoints

        if max_matches:
            matches = matches[:max_matches]

        imag = cv2.drawMatches(
            imag1, kpts1,
            imag2, kpts2,
            matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return imag


    def imag_w_epipolar(self, num_points=40, scale=1.5, point_size=12):
        """
        Return two images:
        - Image 1: with epipolar lines (from points in image2) + keypoints
        - Image 2: with epipolar lines (from points in image1) + keypoints
        """
        if self.F is None:
            print("You must estimate the fundamental matrix first.")
            return None, None

        if self.matches_inlier is None or len(self.matches_inlier) == 0:
            print("No inlier matches found. Run estimate_fundamental_matrix() first.")
            return None, None

        imag1 = self.simag1.imag_color.copy()
        imag2 = self.simag2.imag_color.copy()
        kpts1 = self.simag1.keypoints
        kpts2 = self.simag2.keypoints

        # pick subset of inlier matches
        np.random.seed(0)
        matches_drawn = np.random.choice(
            self.matches_inlier,
            size=min(num_points, len(self.matches_inlier)),
            replace=False
        )

        pts1 = np.float32([kpts1[m.queryIdx].pt for m in matches_drawn])
        pts2 = np.float32([kpts2[m.trainIdx].pt for m in matches_drawn])

        # ---- Assign a color per match and reuse it ----
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(matches_drawn))]

        # ---- Lines in image 2 (from points in image1) ----
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, self.F).reshape(-1, 3)
        for (x1, y1), (a, b, c), color in zip(pts1, lines2, colors):
            cv2.circle(imag1, (int(x1), int(y1)), point_size, color, -1)  # larger point in image1
            x0, y0 = 0, int(-c / b) if abs(b) > 1e-6 else 0
            x1l, y1l = imag2.shape[1], int(-(c + a * imag2.shape[1]) / b) if abs(b) > 1e-6 else imag2.shape[0]
            cv2.line(imag2, (x0, y0), (x1l, y1l), color, 2)  # epipolar line in image2

        # ---- Lines in image 1 (from points in image2) ----
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, self.F).reshape(-1, 3)
        for (x2, y2), (a, b, c), color in zip(pts2, lines1, colors):
            cv2.circle(imag2, (int(x2), int(y2)), point_size, color, -1)  # larger point in image2
            x0, y0 = 0, int(-c / b) if abs(b) > 1e-6 else 0
            x1l, y1l = imag1.shape[1], int(-(c + a * imag1.shape[1]) / b) if abs(b) > 1e-6 else imag1.shape[0]
            cv2.line(imag1, (x0, y0), (x1l, y1l), color, 2)  # epipolar line in image1

        # ---- Resize images to see lines better ----
        if scale != 1.0:
            imag1 = cv2.resize(imag1, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            imag2 = cv2.resize(imag2, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        return imag1, imag2


    def evaluate_fundamental_estimation_quality(self):
        if self.F is None or self.matches_inlier is None:
            print("Fundamental matrix not estimated yet.")
            return None

        if len(self.matches_inlier) == 0:
            print("No inlier matches to evaluate.")
            return None

        kpts1 = self.simag1.keypoints
        kpts2 = self.simag2.keypoints

        pts1 = np.float32([kpts1[m.queryIdx].pt for m in self.matches_inlier])
        pts2 = np.float32([kpts2[m.trainIdx].pt for m in self.matches_inlier])

        # Convert points to homogeneous coordinates
        pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
        pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])

        F = self.F

        algebraic = []
        sampson = []
        sym_epi = []

        for p1, p2 in zip(pts1_h, pts2_h):
            # Algebraic error
            err = abs(p2 @ F @ p1)
            algebraic.append(err)

            # Sampson distance
            Fx1 = F @ p1
            Ftx2 = F.T @ p2
            denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2
            if denom > 1e-12:
                sampson.append((err ** 2) / denom)

            # Symmetric epipolar distance
            l2 = Fx1  # epipolar line in image2
            l1 = Ftx2  # epipolar line in image1
            d2 = abs(p2 @ l2) / np.sqrt(l2[0] ** 2 + l2[1] ** 2 + 1e-12)
            d1 = abs(p1 @ l1) / np.sqrt(l1[0] ** 2 + l1[1] ** 2 + 1e-12)
            sym_epi.append(0.5 * (d1 + d2))

        # Rank and singular values
        _, svals, _ = np.linalg.svd(F)
        rank = int(np.sum(svals > 1e-12))
        if rank != 2:
            print(f"Warning: Fundamental matrix rank is {rank}, should be 2.")

        results = {
            "mean_algebraic": np.mean(algebraic),
            "median_algebraic": np.median(algebraic),
            "mean_sampson": np.mean(sampson),
            "median_sampson": np.median(sampson),
            "mean_symmetric_epipolar": np.mean(sym_epi),
            "median_symmetric_epipolar": np.median(sym_epi),
            "rank": rank,
            "singular_values": [f"{sv:.3e}" for sv in svals],
            "n_inliers": len(algebraic)
        }

        print("Fundamental matrix F estimation quality:")
        for k, v in results.items():
            print(f"  {k.replace('_', ' ')}: {v}")

        return results


    def evaluate_essential_estimation_quality(self):
        if self.E is None or self.matches_inlier is None:
            print("Essential matrix not estimated yet.")
            return None
        if self.K is None:
            print("Camera intrinsics K not provided.")
            return None
        if len(self.matches_inlier) == 0:
            print("No inlier matches to evaluate.")
            return None

        kpts1 = self.simag1.keypoints
        kpts2 = self.simag2.keypoints

        pts1 = np.float32([kpts1[m.queryIdx].pt for m in self.matches_inlier])
        pts2 = np.float32([kpts2[m.trainIdx].pt for m in self.matches_inlier])

        # Convert to homogeneous coordinates
        pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
        pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])

        # Normalize points using camera intrinsics
        K_inv = np.linalg.inv(self.K)
        pts1_n = (K_inv @ pts1_h.T).T  # shape: (N, 3)
        pts2_n = (K_inv @ pts2_h.T).T  # shape: (N, 3)

        E = self.E
        algebraic = []
        sampson = []
        sym_epi = []

        for p1, p2 in zip(pts1_n, pts2_n):
            # Algebraic error
            err = abs(p2 @ E @ p1)
            algebraic.append(err)

            # Sampson distance
            Ex1 = E @ p1
            Etx2 = E.T @ p2
            denom = Ex1[0] ** 2 + Ex1[1] ** 2 + Etx2[0] ** 2 + Etx2[1] ** 2
            if denom > 1e-12:
                sampson.append((err ** 2) / denom)

            # Symmetric epipolar distance
            l2 = Ex1
            l1 = Etx2
            d2 = abs(p2 @ l2) / np.sqrt(l2[0] ** 2 + l2[1] ** 2 + 1e-12)
            d1 = abs(p1 @ l1) / np.sqrt(l1[0] ** 2 + l1[1] ** 2 + 1e-12)
            sym_epi.append(0.5 * (d1 + d2))

        # Rank and singular values
        _, svals, _ = np.linalg.svd(E)
        rank = int(np.sum(svals > 1e-12))

        # Check singular value equality
        if len(svals) >= 2 and abs(svals[0] - svals[1]) / max(svals[0], 1e-12) > 0.1:
            print("Warning: Top two singular values are not equal; E may be noisy.")

        if rank != 2:
            print(f"Warning: Essential matrix rank is {rank}, should be 2.")

        results = {
            "mean_algebraic": np.mean(algebraic),
            "median_algebraic": np.median(algebraic),
            "mean_sampson": np.mean(sampson),
            "median_sampson": np.median(sampson),
            "mean_symmetric_epipolar": np.mean(sym_epi),
            "median_symmetric_epipolar": np.median(sym_epi),
            "rank": rank,
            "singular_values": [f"{sv:.3e}" for sv in svals],
            "n_inliers": len(algebraic)
        }

        print("Essential matrix E estimation quality:")
        for k, v in results.items():
            print(f"  {k.replace('_', ' ')}: {v}")

        return results