import numpy as np
from scipy.spatial import cKDTree


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