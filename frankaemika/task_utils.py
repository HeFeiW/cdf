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
    # return auto_find_contact_points(points, threshold=0.027)

def auto_find_contact_points(points, threshold=0.01):
    ok = False
    while not ok:
        # 1. 找到物体质心
        print(f'shape of points: {points.shape}')
        centroid = torch.mean(points, axis=0)
        centroid = centroid.detach().cpu()
        # 2. 过质心做一条随机直线
        line_start = centroid - torch.rand(3) * threshold
        line_end = centroid + torch.rand(3) * threshold
        # debug
        line_start = torch.tensor([0, -0.075, 0.45])
        line_end = torch.tensor([-0.2, -0.075, 0.45])
        # 3. 找到距离直线小于threshold的点作为接触点
        line_vec = line_end - line_start
        line_vec /= torch.norm(line_vec)
        point_vecs = points.detach().cpu() - line_start
        print(f'line_start: {line_start}, line_end: {line_end}')
        print(f'line_vec: {line_vec}')
        print(f'point_vecs: {point_vecs}')
        c = input(f'Current threshold: {threshold}. Input new threshold or press enter to keep: ')
        if c != '':
            threshold = float(c)
        else:
            threshold = threshold
        # dists = torch.norm(point_vecs - torch.outer(torch.dot(point_vecs, line_vec), line_vec), dim=1)
        # RuntimeError: 1D tensors expected, but got 2D and 1D tensors
        dists = torch.norm(point_vecs - torch.outer(torch.mv(point_vecs, line_vec), line_vec), dim=1)
        mask = dists < threshold
        contact_points = points[mask]
        object_points = points[~mask]
        # debug: plot the points with contact points in red
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            cpu_object_points = object_points.detach().cpu().numpy()
            cpu_contact_points = contact_points.detach().cpu().numpy()
            ax.scatter(cpu_object_points[:, 0], cpu_object_points[:, 1], cpu_object_points[:, 2], c='b', s=1)
            ax.scatter(cpu_contact_points[:, 0], cpu_contact_points[:, 1], cpu_contact_points[:, 2], c='r', s=5)
            ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], [line_start[2], line_end[2]], c='g')
            # display the plot, do not clear after showing
            plt.show(block=False)
            # ask user if the contact points are ok
            ans = input(f"Found {contact_points.shape[0]} contact points. Are they ok? (y/n): ")
            
            if ans.lower() == 'y':
                ok = True
        except ImportError:
            print("cannot visualize contact points.")
            ok = True
    return contact_points, object_points
