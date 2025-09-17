import matplotlib.pyplot as plt
import numpy as np

def draw_camera_frustum(ax, R, t, scale=0.1, depth_factor=3.0, color='r'):
    """
    Draw a simple camera frustum as a pyramid pointing along Z-axis.
    R: rotation matrix of camera (3x3)
    t: camera center (3,)
    scale: size of the base (width/height)
    depth_factor: how much longer the frustum depth is relative to scale
    """
    origin = t.ravel()

    # Width, height, depth of frustum
    w, h = scale, scale
    d = scale * depth_factor  # make frustum deeper

    # Define frustum in camera local coordinates
    corners = np.array([
        [ w,  h, d],
        [ w, -h, d],
        [-w, -h, d],
        [-w,  h, d],
    ]).T  # shape (3,4)

    # Rotate and translate to world coordinates
    world_corners = R @ corners + origin.reshape(3,1)

    # Draw pyramid edges
    for i in range(4):
        ax.plot([origin[0], world_corners[0,i]],
                [origin[1], world_corners[1,i]],
                [origin[2], world_corners[2,i]], c=color)
    for i in range(4):
        ax.plot([world_corners[0,i], world_corners[0,(i+1)%4]],
                [world_corners[1,i], world_corners[1,(i+1)%4]],
                [world_corners[2,i], world_corners[2,(i+1)%4]], c=color)

    # Draw Z axis (view direction) as arrow
    z_axis = R[:,2] * d
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              color=color, arrow_length_ratio=0.2, linewidth=2)



class Plot:
    @classmethod
    def plot_images_grid(cls, images, nrows=1, ncols=1, figsize=(12, 8)):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        # Make axes iterable, whether it's 1 Axes or an array
        if nrows * ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, img in zip(axes, images):
            if img.ndim == 2:  # grayscale
                ax.imshow(img, cmap="gray")
            else:  # RGB
                ax.imshow(img)
            ax.axis("off")

        for ax in axes[len(images):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_cameras_frustum(cls, camera_poses, points3d=None, points3d_color=None, scale=0.33, points3d_size=2.5):
        """
        Plot multiple camera frustums and optional 3D points.

        camera_poses: list of tuples [(R1, C1), (R2, C2), ...]
            - R: 3x3 rotation matrix (camera->world)
            - C: 3x1 camera center in world coordinates
        points3d: optional Nx3 array of 3D points
        points3d_color: optional Nx3 array of RGB colors for each 3D point (uint8)
        scale: frustum size
        points3d_size: size of the scatter points for 3D points
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.get_cmap('tab10', len(camera_poses))  # automatic colors
        for i, (R, C) in enumerate(camera_poses):
            color = colors(i) if hasattr(colors, '__call__') else 'r'
            draw_camera_frustum(ax, R, C, scale=scale, color=color)
            ax.scatter([], [], [], c=[color], marker='o', label=f'Camera {i + 1}')  # dummy for legend

        # 3D points
        if points3d is not None:
            if points3d_color is not None and len(points3d_color) == len(points3d):
                # Normalize colors to [0,1] for matplotlib
                colors_norm = points3d_color.astype(np.float32) / 255.0
                ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2],
                           c=colors_norm, marker='.', s=points3d_size, label='3D points')
            else:
                ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2],
                           c='g', marker='.', s=points3d_size, label='3D points')

        # Axis labels and aspect
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])

        # Legend
        ax.legend()

        # Optional: set camera-friendly perspective
        ax.view_init(elev=20, azim=-60)

        plt.show()
