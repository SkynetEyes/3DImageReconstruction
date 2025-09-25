import numpy as np

def remove_outliers_std(points, points_color=None, n_std=2.0):
    """
    Remove points that are farther than `n_std` standard deviations from the centroid.
    Optionally removes corresponding entries in points_color.

    Args:
        points: np.ndarray of shape (N,3)
        points_color: optional np.ndarray of shape (N,3) corresponding RGB colors
        n_std: float, number of standard deviations to keep

    Returns:
        filtered_points: np.ndarray of filtered points (M,3)
        filtered_colors: np.ndarray of corresponding colors (M,3) if points_color is provided,
                         otherwise None
    """
    if len(points) == 0:
        return points, points_color if points_color is not None else None

    centroid = points.mean(axis=0)
    std_dev = points.std(axis=0)

    # Keep points within n_std in all axes
    lower_bound = centroid - n_std * std_dev
    upper_bound = centroid + n_std * std_dev
    mask = np.all((points >= lower_bound) & (points <= upper_bound), axis=1)

    filtered_points = points[mask]
    filtered_colors = points_color[mask] if points_color is not None else None

    return filtered_points, filtered_colors