import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



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

    @classmethod
    def plot_cameras_surface(cls, camera_poses, points3d=None, points3d_color=None, scale=0.33):
        """
        Plot multiple camera frustums and optional 3D surface created from points.

        Parameters
        ----------
        camera_poses : list of tuples [(R1, C1), (R2, C2), ...]
            - R: 3x3 rotation matrix (camera->world)
            - C: 3x1 camera center in world coordinates
        points3d : Nx3 array of 3D points
        points3d_color : optional Nx3 array of RGB colors (uint8)
        scale : float, frustum size
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot camera frustums
        colors = plt.cm.get_cmap('tab10', len(camera_poses))
        for i, (R, C) in enumerate(camera_poses):
            color = colors(i) if hasattr(colors, '__call__') else 'r'
            draw_camera_frustum(ax, R, C, scale=scale, color=color)
            ax.scatter([], [], [], c=[color], marker='o', label=f'Camera {i + 1}')  # dummy for legend

        # Plot surface from points
        if points3d is not None and len(points3d) >= 3:
            # Triangulate in XY plane
            tri = Delaunay(points3d[:, :2])

            # If color is provided, average per triangle
            if points3d_color is not None and len(points3d_color) == len(points3d):
                colors_norm = points3d_color.astype(np.float32) / 255.0
                face_color = np.mean(colors_norm[tri.simplices], axis=1)
            else:
                face_color = 'lightblue'

            ax.plot_trisurf(points3d[:, 0], points3d[:, 1], points3d[:, 2],
                            triangles=tri.simplices, facecolor=face_color, linewidth=0.2, alpha=0.9)

        # Axis labels and aspect
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])

        # Legend
        ax.legend()
        ax.view_init(elev=20, azim=-60)

        plt.show()


    @classmethod
    def plot_cameras_surface_grid(cls, camera_poses, points3d, points3d_color=None, grid_size=100, alpha=0.9):
        """
        Plot camera frustums and a smooth surface using grid-based interpolation.

        Args:
            camera_poses: list of (R, C) tuples
            points3d: Nx3 array of 3D points
            points3d_color: optional Nx3 array of colors (uint8)
            grid_size: number of points per axis for the interpolation grid
            alpha: surface transparency
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot cameras
        colors = plt.cm.get_cmap('tab10', len(camera_poses))
        for i, (R, C) in enumerate(camera_poses):
            color = colors(i) if hasattr(colors, '__call__') else 'r'
            draw_camera_frustum(ax, R, C, scale=0.33, color=color)
            ax.scatter([], [], [], c=[color], marker='o', label=f'Camera {i+1}')

        # Interpolate points to grid
        x, y, z = points3d[:,0], points3d[:,1], points3d[:,2]
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), grid_size),
                                     np.linspace(y.min(), y.max(), grid_size))
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

        # Plot surface
        ax.plot_surface(grid_x, grid_y, grid_z, alpha=alpha, cmap='viridis')

        # Axis and legend
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,1])
        ax.legend()
        ax.view_init(elev=20, azim=-60)
        plt.show()

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    import numpy as np

    @classmethod
    def show_poisson_surface_plot(cls, camera_poses, points3d, points3d_color=None):
        import open3d as o3d

        # Convert points to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        points3d = np.asarray(points3d)
        pcd.points = o3d.utility.Vector3dVector(points3d)

        if points3d_color is not None:
            colors = np.asarray(points3d_color, dtype=np.float32) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Estimate normals for Poisson reconstruction
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)

        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=4)
        mesh.compute_vertex_normals()

        # Remove low-density vertices
        densities = np.asarray(densities)
        mask = densities >= np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(~mask)

        # Convert to numpy arrays
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # Compute face colors
        if len(mesh.vertex_colors) > 0:
            vertex_colors = np.asarray(mesh.vertex_colors)
            face_colors = vertex_colors[triangles].mean(axis=1)
        else:
            face_colors = np.ones((len(triangles), 3)) * [0.6, 0.8, 1.0]

        # Create Poly3DCollection
        faces = vertices[triangles]
        mesh_collection = Poly3DCollection(faces, facecolors=face_colors, edgecolor='gray', linewidths=0.2)

        # Plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.add_collection3d(mesh_collection)

        # Add cameras
        if camera_poses is not None:
            for R, C in camera_poses:
                ax.scatter(C[0], C[1], C[2], color='red', s=30)

        # Set limits
        max_range = np.ptp(vertices, axis=0).max() / 2.0
        mid = vertices.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        plt.tight_layout()
        plt.show()


