import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d
import os

def draw_camera_frustum(ax, R, t, scale=0.1, depth_factor=3.0, color='r'):
    origin = t.ravel()
    w, h = scale, scale
    d = scale * depth_factor
    corners = np.array([
        [ w,  h, d],
        [ w, -h, d],
        [-w, -h, d],
        [-w,  h, d],
    ]).T
    world_corners = R @ corners + origin.reshape(3,1)
    for i in range(4):
        ax.plot([origin[0], world_corners[0,i]],
                [origin[1], world_corners[1,i]],
                [origin[2], world_corners[2,i]], c=color)
    for i in range(4):
        ax.plot([world_corners[0,i], world_corners[0,(i+1)%4]],
                [world_corners[1,i], world_corners[1,(i+1)%4]],
                [world_corners[2,i], world_corners[2,(i+1)%4]], c=color)
    z_axis = R[:,2] * d
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              color=color, arrow_length_ratio=0.2, linewidth=2)

class Plot:
    _figures_to_show = []

    @classmethod
    def plot_images_grid(cls, images, nrows=1, ncols=1, figsize=(12, 8), save_path=None, show=False):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = [axes] if nrows * ncols == 1 else axes.flatten()
        for ax, img in zip(axes, images):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.axis("off")
        for ax in axes[len(images):]:
            ax.axis("off")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close(fig)  # saved-only figures are closed
        if show:
            cls._figures_to_show.append(fig)
        return fig

    @classmethod
    def plot_cameras_frustum(cls, camera_poses, points3d=None, points3d_color=None,
                             scale=0.33, points3d_size=2.5, save_path=None, show=False):
        angles = [
            (20, -60), (20, 0), (20, 60),
            (40, -60), (40, 0), (40, 60),
            (60, -60), (60, 0), (60, 60)
        ]

        # Save multiple views
        if save_path:
            base_dir, base_file = os.path.split(save_path)
            base_name, ext = os.path.splitext(base_file)
            os.makedirs(base_dir or ".", exist_ok=True)
            for i, (elev, azim) in enumerate(angles, 1):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                colors = plt.cm.get_cmap("tab10", len(camera_poses))
                for j, (R, C) in enumerate(camera_poses):
                    color = colors(j)
                    draw_camera_frustum(ax, R, C, scale=scale, color=color)
                    ax.scatter([], [], [], c=[color], marker="o", label=f"Camera {j + 1}")
                if points3d is not None:
                    if points3d_color is not None:
                        colors_norm = points3d_color.astype(np.float32)/255.0
                        ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2],
                                   c=colors_norm, s=points3d_size)
                    else:
                        ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2],
                                   c="g", s=points3d_size)
                ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
                ax.set_box_aspect([1,1,1])
                ax.view_init(elev=elev, azim=azim)
                plt.savefig(os.path.join(base_dir, f"{base_name}_{i}{ext}"))
                plt.close(fig)  # save-only figures closed

        # Interactive figure
        if show:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            colors = plt.cm.get_cmap("tab10", len(camera_poses))
            for j, (R, C) in enumerate(camera_poses):
                color = colors(j)
                draw_camera_frustum(ax, R, C, scale=scale, color=color)
                ax.scatter([], [], [], c=[color], marker="o", label=f"Camera {j + 1}")
            if points3d is not None:
                if points3d_color is not None:
                    colors_norm = points3d_color.astype(np.float32)/255.0
                    ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2],
                               c=colors_norm, s=points3d_size)
                else:
                    ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2],
                               c="g", s=points3d_size)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.set_box_aspect([1,1,1])
            ax.view_init(elev=20, azim=-60)
            cls._figures_to_show.append(fig)
            return fig

    @classmethod
    def plot_cameras_surface(cls, camera_poses, points3d=None, points3d_color=None,
                             scale=0.33, save_path=None, show=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.get_cmap('tab10', len(camera_poses))
        for i, (R, C) in enumerate(camera_poses):
            color = colors(i)
            draw_camera_frustum(ax, R, C, scale=scale, color=color)
            ax.scatter([], [], [], c=[color], marker='o', label=f'Camera {i+1}')
        if points3d is not None and len(points3d) >= 3:
            tri = Delaunay(points3d[:,:2])
            face_color = 'lightblue'
            if points3d_color is not None and len(points3d_color) == len(points3d):
                colors_norm = points3d_color.astype(np.float32)/255.0
                face_color = np.mean(colors_norm[tri.simplices], axis=1)
            ax.plot_trisurf(points3d[:,0], points3d[:,1], points3d[:,2],
                            triangles=tri.simplices, facecolor=face_color, linewidth=0.2, alpha=0.9)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,1])
        ax.legend()
        ax.view_init(elev=20, azim=-60)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close(fig)
        if show:
            cls._figures_to_show.append(fig)
        return fig

    @classmethod
    def plot_cameras_surface_grid(cls, camera_poses, points3d, points3d_color=None,
                                  grid_size=100, alpha=0.9, save_path=None, show=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.get_cmap('tab10', len(camera_poses))
        for i, (R, C) in enumerate(camera_poses):
            color = colors(i)
            draw_camera_frustum(ax, R, C, scale=0.33, color=color)
            ax.scatter([], [], [], c=[color], marker='o', label=f'Camera {i+1}')
        x, y, z = points3d[:,0], points3d[:,1], points3d[:,2]
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), grid_size),
                                     np.linspace(y.min(), y.max(), grid_size))
        grid_z = griddata((x,y), z, (grid_x, grid_y), method='cubic')
        ax.plot_surface(grid_x, grid_y, grid_z, alpha=alpha, cmap='viridis')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,1])
        ax.legend()
        ax.view_init(elev=20, azim=-60)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close(fig)
        if show:
            cls._figures_to_show.append(fig)
        return fig

    @classmethod
    def show_poisson_surface_plot(cls, camera_poses, points3d, points3d_color=None,
                                  save_path=None, show=False):
        pcd = o3d.geometry.PointCloud()
        points3d = np.asarray(points3d)
        pcd.points = o3d.utility.Vector3dVector(points3d)
        if points3d_color is not None:
            colors = np.asarray(points3d_color, dtype=np.float32)/255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=4)
        mesh.compute_vertex_normals()
        densities = np.asarray(densities)
        mask = densities >= np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(~mask)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        if len(mesh.vertex_colors) > 0:
            vertex_colors = np.asarray(mesh.vertex_colors)
            face_colors = vertex_colors[triangles].mean(axis=1)
        else:
            face_colors = np.ones((len(triangles),3))*[0.6,0.8,1.0]
        mesh_collection = Poly3DCollection(vertices[triangles], facecolors=face_colors,
                                           edgecolor='gray', linewidths=0.2)
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.add_collection3d(mesh_collection)
        if camera_poses is not None:
            for R, C in camera_poses:
                ax.scatter(C[0], C[1], C[2], color='red', s=30)
        max_range = np.ptp(vertices, axis=0).max()/2.0
        mid = vertices.mean(axis=0)
        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,1])
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close(fig)
        if show:
            cls._figures_to_show.append(fig)
        return fig

    @classmethod
    def show(cls):
        """Show only figures passed with show=True. Blocks execution for interactivity."""
        if cls._figures_to_show:
            plt.show()  # This will block and allow interactive zoom/rotate
            cls._figures_to_show.clear()