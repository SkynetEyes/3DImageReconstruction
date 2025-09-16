import numpy as np

class SfmGlobal:
    def __init__(self, sip_list):
        """
        Global Structure from Motion over a sequence of SuperImagePairs (SIP).

        Parameters
        ----------
        sip_list : list
            List of SuperImagePair objects, already processed with:
                - sip.match()
                - sip.estimate_fundamental_matrix()
                - sip.estimate_essential_matrix()
                - sip.estimate_pose()
        """
        self.sip_list = sip_list
        self.camera_poses = []   # list of (R, C) for each camera in global frame
        self.points3d = []       # accumulated 3D points in world frame

    def run(self):
        # --- Fix the first camera as the world origin ---
        R_global = np.eye(3)
        C_global = np.zeros((3, 1))
        self.camera_poses.append((R_global, C_global))  # Camera 1 pose

        # --- Iterate through SIPs ---
        for i, sip in enumerate(self.sip_list):
            # Pose of second camera wrt first in this SIP (already estimated)
            R2, C2 = sip.get_camera_2_pose()

            # Global pose of previous camera (Ik)
            R_prev, C_prev = self.camera_poses[-1]

            # Compose transformations to get global pose of Ik+1
            R_new = R_prev @ R2
            C_new = R_prev @ C2 + C_prev

            # Store the new camera pose in global coordinates
            self.camera_poses.append((R_new, C_new))

            # Triangulate new 3D points (in SIP local frame)
            sip.estimate_points3d()
            local_pts = sip.get_points3d()

            # Transform points from SIP local frame -> global frame
            world_pts = (R_prev @ local_pts.T).T + C_prev.ravel()

            self.points3d.extend(world_pts)

        # Stack all 3D points into an array
        if len(self.points3d) > 0:
            self.points3d = np.vstack(self.points3d)

        return self.camera_poses, self.points3d