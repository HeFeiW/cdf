import numpy as np
import torch
def discriminate_target_obstacle(points):
    """
    Discriminate points as targets(1) or obstacles(0) based on certain criteria.

    Args:
        points (torch.tensor): Array of points to be evaluated.(N, 3)

    Returns:
        torch.tensor: A boolean array indicating whether each point is an obstacle (True) or target (False).
    """
    # default: all points are considered as target
    # return np.ones(points.shape[0], dtype=bool)
    # 把z中间1/3的设为target，其他为obstacle
    z_min = np.min(points[:, 2].detach().cpu().numpy())
    z_max = np.max(points[:, 2].detach().cpu().numpy())
    z_range = z_max - z_min
    lower_bound = z_min + z_range / 3
    upper_bound = z_max - z_range / 3
    is_obstacle = (points[:, 2] < lower_bound) | (points[:, 2] > upper_bound)
    # return as torch tensor
    return is_obstacle

def seperate_target_obstacle(points):
    """
    Separate points into target points and obstacle points.

    Args:
        points (np.ndarray): Array of points to be separated.(N, 3)
    Returns:
        tuple: A tuple containing two arrays - target points and obstacle points.
    """
    is_obstacle = discriminate_target_obstacle(points)
    target_points = points[~is_obstacle]
    obstacle_points = points[is_obstacle]
    return target_points, obstacle_points