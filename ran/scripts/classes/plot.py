import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .point_densifier import densify_point_cloud

# Check for GPU support
_gpu_available = None

def check_gpu_support():
    """Check if CUDA/GPU support is available for Open3D."""
    global _gpu_available
    if _gpu_available is None:
        try:
            # Try to create a CUDA device
            if hasattr(o3d.core, 'cuda') and o3d.core.cuda.device_count() > 0:
                _gpu_available = True
                print(f"GPU acceleration available: {o3d.core.cuda.device_count()} CUDA devices found")
            else:
                _gpu_available = False
                print("GPU acceleration not available, using CPU")
        except:
            _gpu_available = False
            print("GPU acceleration not available, using CPU")
    return _gpu_available



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
    def show_poisson_surface_plot(cls, camera_poses, points3d, points3d_color=None, 
                                densify=True, densify_method='delaunay', density_factor=2.0,
                                poisson_depth=6, normal_radius=0.05):
        """
        Show Poisson surface reconstruction with optional point densification.
        
        Args:
            camera_poses: List of camera poses (R, t)
            points3d: 3D points array (N, 3)
            points3d_color: Optional colors array (N, 3)
            densify: Whether to apply point densification before Poisson reconstruction
            densify_method: Method for densification ('delaunay', 'knn', 'rbf', 'grid')
            density_factor: Factor to increase point density (2.0 = ~4x more points)
            poisson_depth: Depth parameter for Poisson reconstruction (higher = more detail)
            normal_radius: Radius for normal estimation
        """
        import open3d as o3d

        # Apply point densification if requested
        if densify and len(points3d) >= 4:
            print(f"Original point cloud: {len(points3d)} points")
            try:
                # Use adaptive max_points based on system performance
                adaptive_max_points = min(15000, max(5000, len(points3d) * 3))
                
                points3d, points3d_color = densify_point_cloud(
                    points3d, points3d_color, 
                    method=densify_method, 
                    density_factor=density_factor,
                    max_points=adaptive_max_points
                )
                print(f"After densification: {len(points3d)} points")
            except Exception as e:
                print(f"Densification failed: {e}")
                print("Proceeding with original points...")

        # Convert points to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        points3d = np.asarray(points3d)
        pcd.points = o3d.utility.Vector3dVector(points3d)

        if points3d_color is not None:
            colors = np.asarray(points3d_color, dtype=np.float32) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Estimate normals for Poisson reconstruction
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)

        # Poisson reconstruction with configurable depth
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth)
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
        fig = plt.figure(figsize=(12, 8))
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
        plt.title(f'Poisson Surface Reconstruction ({len(points3d)} points, densified={densify})')
        plt.tight_layout()
        plt.show()

    @classmethod
    def show_poisson_surface_gpu(cls, camera_poses, points3d, points3d_color=None, 
                                densify=True, densify_method='knn', density_factor=1.5,
                                poisson_depth=6, normal_radius=0.05, use_gpu=True):
        """
        GPU-accelerated Poisson surface reconstruction and visualization using Open3D.
        
        This is MUCH faster than matplotlib for complex meshes!
        
        Args:
            camera_poses: List of camera poses (R, t)
            points3d: 3D points array (N, 3)
            points3d_color: Optional colors array (N, 3)
            densify: Whether to apply point densification
            densify_method: Method for densification ('knn' recommended for speed)
            density_factor: Factor to increase point density
            poisson_depth: Depth parameter for Poisson reconstruction
            normal_radius: Radius for normal estimation
            use_gpu: Whether to use GPU acceleration (if available)
        """
        print("=== GPU-Accelerated Poisson Surface Visualization ===")
        
        # Check GPU availability
        gpu_available = check_gpu_support() and use_gpu
        
        # Apply point densification if requested
        if densify and len(points3d) >= 4:
            print(f"Original point cloud: {len(points3d)} points")
            try:
                adaptive_max_points = min(20000, max(5000, len(points3d) * 4))
                points3d, points3d_color = densify_point_cloud(
                    points3d, points3d_color, 
                    method=densify_method, 
                    density_factor=density_factor,
                    max_points=adaptive_max_points
                )
                print(f"After densification: {len(points3d)} points")
            except Exception as e:
                print(f"Densification failed: {e}")
                print("Proceeding with original points...")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        points3d = np.asarray(points3d)
        pcd.points = o3d.utility.Vector3dVector(points3d)

        if points3d_color is not None:
            colors = np.asarray(points3d_color, dtype=np.float32) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Estimate normals (can be GPU accelerated)
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)

        # Poisson reconstruction
        print(f"Performing Poisson reconstruction (depth={poisson_depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth)
        
        # Clean up the mesh
        print("Cleaning mesh...")
        mesh.compute_vertex_normals()
        
        # Remove low-density vertices
        densities = np.asarray(densities)
        if len(densities) > 0:
            mask = densities >= np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(~mask)

        # Create camera visualizations
        camera_geometries = []
        if camera_poses is not None:
            for i, (R, t) in enumerate(camera_poses):
                # Create a small sphere for camera position
                camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                camera_sphere.translate(t.flatten())
                camera_sphere.paint_uniform_color([1, 0, 0])  # Red
                camera_geometries.append(camera_sphere)
                
                # Create camera frustum (simplified)
                frustum = cls._create_camera_frustum_geometry(R, t, scale=0.1)
                camera_geometries.append(frustum)

        # Visualize with Open3D's fast viewer
        print("Opening GPU-accelerated 3D viewer...")
        print("Controls: Mouse to rotate, Scroll to zoom, Right-click drag to pan")
        print("Press 'Q' to close the viewer")
        
        geometries = [mesh] + camera_geometries
        
        # Set up visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Poisson Surface ({len(points3d)} points)", 
                         width=1200, height=800)
        
        # Add geometries
        for geom in geometries:
            vis.add_geometry(geom)
            
        # Configure rendering options for better performance
        render_option = vis.get_render_option()
        render_option.mesh_show_wireframe = False
        render_option.mesh_show_back_face = True
        render_option.point_size = 2.0
        render_option.line_width = 1.0
        
        # Enable GPU rendering if available
        if gpu_available:
            print("Using GPU acceleration for rendering")
            # Open3D will automatically use GPU if available
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()
        
    @classmethod
    def show_point_cloud_gpu(cls, camera_poses, points3d, points3d_color=None):
        """
        Show just the point cloud using GPU-accelerated Open3D visualization.
        No surface reconstruction - just points and cameras.
        """
        print("=== GPU Point Cloud Visualization ===")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        points3d = np.asarray(points3d)
        pcd.points = o3d.utility.Vector3dVector(points3d)

        if points3d_color is not None:
            colors = np.asarray(points3d_color, dtype=np.float32) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create camera visualizations
        camera_geometries = []
        if camera_poses is not None:
            for i, (R, t) in enumerate(camera_poses):
                # Create a small sphere for camera position
                camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                camera_sphere.translate(t.flatten())
                camera_sphere.paint_uniform_color([1, 0, 0])  # Red
                camera_geometries.append(camera_sphere)
                
                # Create camera frustum (simplified)
                frustum = cls._create_camera_frustum_geometry(R, t, scale=0.1)
                camera_geometries.append(frustum)

        # Visualize with Open3D's fast viewer
        print("Opening GPU-accelerated point cloud viewer...")
        print("Controls: Mouse to rotate, Scroll to zoom, Right-click drag to pan")
        print("Press 'Q' to close the viewer")
        
        geometries = [pcd] + camera_geometries
        
        # Set up visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Point Cloud ({len(points3d)} points)", 
                         width=1200, height=800)
        
        # Add geometries
        for geom in geometries:
            vis.add_geometry(geom)
            
        # Configure rendering options
        render_option = vis.get_render_option()
        render_option.point_size = 3.0  # Larger points for better visibility
        render_option.show_coordinate_frame = True
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()
        
    @classmethod
    def _create_camera_frustum_geometry(cls, R, t, scale=0.1):
        """Create a camera frustum geometry for Open3D visualization."""
        # Camera frustum vertices in camera coordinates
        frustum_points = np.array([
            [0, 0, 0],  # Camera center
            [-scale, -scale, scale*2],  # Bottom-left
            [scale, -scale, scale*2],   # Bottom-right
            [scale, scale, scale*2],    # Top-right
            [-scale, scale, scale*2],   # Top-left
        ])
        
        # Transform to world coordinates
        world_points = (R @ frustum_points.T).T + t.flatten()
        
        # Create lines for frustum edges
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # From center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]   # Rectangle edges
        ]
        
        # Create LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(world_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.2]] * len(lines))  # Yellow
        
        return line_set

    @classmethod
    def compare_poisson_gpu_vs_matplotlib(cls, camera_poses, points3d, points3d_color=None,
                                        densify_method='knn', density_factor=1.5):
        """
        Compare GPU vs matplotlib rendering performance and quality.
        Shows matplotlib first, then GPU version.
        """
        print("=== Performance Comparison: Matplotlib vs GPU ===")
        
        # Show matplotlib version (slower)
        print("\n1. Matplotlib rendering (slower but familiar):")
        import time
        start_time = time.time()
        
        cls.show_poisson_surface_plot(
            camera_poses, points3d, points3d_color,
            densify=True, densify_method=densify_method, 
            density_factor=density_factor,
            poisson_depth=6
        )
        
        matplotlib_time = time.time() - start_time
        print(f"Matplotlib rendering time: {matplotlib_time:.2f} seconds")
        
        # Show GPU version (faster)
        print("\n2. GPU-accelerated rendering (much faster):")
        start_time = time.time()
        
        cls.show_poisson_surface_gpu(
            camera_poses, points3d, points3d_color,
            densify=True, densify_method=densify_method,
            density_factor=density_factor,
            poisson_depth=6
        )
        
        gpu_time = time.time() - start_time
        print(f"GPU rendering time: {gpu_time:.2f} seconds")
        
        if gpu_time > 0:
            speedup = matplotlib_time / gpu_time
            print(f"Speedup: {speedup:.1f}x faster with GPU viewer!")

    @classmethod
    def show_point_cloud_gpu(cls, camera_poses, points3d, points3d_color=None):
        """
        Fast GPU-accelerated point cloud visualization without surface reconstruction.
        Useful for quickly inspecting raw reconstruction results.
        """
        print("=== GPU Point Cloud Viewer ===")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points3d)

        if points3d_color is not None:
            colors = np.asarray(points3d_color, dtype=np.float32) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            # Use height-based coloring if no colors provided
            z_coords = points3d[:, 2]
            z_norm = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min())
            colors = plt.cm.viridis(z_norm)[:, :3]  # RGB only
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create camera geometries
        camera_geometries = []
        if camera_poses is not None:
            for R, t in camera_poses:
                camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                camera_sphere.translate(t.flatten())
                camera_sphere.paint_uniform_color([1, 0, 0])
                camera_geometries.append(camera_sphere)

        # Visualize
        geometries = [pcd] + camera_geometries
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Point Cloud ({len(points3d)} points)",
            width=1200, height=800,
            point_show_normal=False
        )

    @classmethod
    def compare_poisson_surface_plots(cls, camera_poses, points3d, points3d_color=None,
                                    densify_method='delaunay', density_factor=2.0):
        """
        Compare Poisson surface reconstruction with and without densification side by side.
        
        Args:
            camera_poses: List of camera poses (R, t)
            points3d: 3D points array (N, 3)
            points3d_color: Optional colors array (N, 3)
            densify_method: Method for densification ('delaunay', 'knn', 'rbf', 'grid')
            density_factor: Factor to increase point density
        """
        import open3d as o3d
        
        fig = plt.figure(figsize=(16, 8))
        
        # Original reconstruction (left)
        ax1 = fig.add_subplot(121, projection='3d')
        cls._plot_poisson_surface_subplot(ax1, camera_poses, points3d, points3d_color, 
                                        densify=False, title="Original Sparse Points")
        
        # Densified reconstruction (right)  
        ax2 = fig.add_subplot(122, projection='3d')
        cls._plot_poisson_surface_subplot(ax2, camera_poses, points3d, points3d_color,
                                        densify=True, densify_method=densify_method, 
                                        density_factor=density_factor,
                                        title=f"Densified ({densify_method})")
        
        plt.tight_layout()
        plt.show()
    
    @classmethod
    def _plot_poisson_surface_subplot(cls, ax, camera_poses, points3d, points3d_color=None,
                                    densify=False, densify_method='delaunay', density_factor=2.0,
                                    title="Poisson Surface"):
        """
        Helper function to create a single Poisson surface plot in a subplot.
        """
        import open3d as o3d
        
        # Apply point densification if requested
        working_points = points3d.copy()
        working_colors = points3d_color.copy() if points3d_color is not None else None
        
        if densify and len(working_points) >= 4:
            try:
                # Use conservative max_points for comparison plots
                adaptive_max_points = min(8000, max(3000, len(working_points) * 2))
                
                working_points, working_colors = densify_point_cloud(
                    working_points, working_colors, 
                    method=densify_method, 
                    density_factor=density_factor,
                    max_points=adaptive_max_points
                )
            except Exception as e:
                print(f"Densification failed for subplot: {e}")

        # Convert points to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        working_points = np.asarray(working_points)
        pcd.points = o3d.utility.Vector3dVector(working_points)

        if working_colors is not None:
            colors = np.asarray(working_colors, dtype=np.float32) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Estimate normals for Poisson reconstruction
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)

        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
        mesh.compute_vertex_normals()

        # Remove low-density vertices
        densities = np.asarray(densities)
        if len(densities) > 0:
            mask = densities >= np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(~mask)

        # Convert to numpy arrays
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        if len(triangles) > 0:
            # Compute face colors
            if len(mesh.vertex_colors) > 0:
                vertex_colors = np.asarray(mesh.vertex_colors)
                face_colors = vertex_colors[triangles].mean(axis=1)
            else:
                face_colors = np.ones((len(triangles), 3)) * [0.6, 0.8, 1.0]

            # Create Poly3DCollection
            faces = vertices[triangles]
            mesh_collection = Poly3DCollection(faces, facecolors=face_colors, edgecolor='gray', linewidths=0.2)
            ax.add_collection3d(mesh_collection)

            # Set limits based on mesh
            max_range = np.ptp(vertices, axis=0).max() / 2.0
            mid = vertices.mean(axis=0)
        else:
            # Fallback to point cloud limits if mesh creation failed
            max_range = np.ptp(working_points, axis=0).max() / 2.0
            mid = working_points.mean(axis=0)
            # Plot points as scatter if mesh failed
            ax.scatter(working_points[:, 0], working_points[:, 1], working_points[:, 2], 
                      c='blue', s=1, alpha=0.6)

        # Add cameras
        if camera_poses is not None:
            for R, C in camera_poses:
                ax.scatter(C[0], C[1], C[2], color='red', s=30)

        # Set limits and labels
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(f'{title}\n({len(working_points)} points)')

