# -----------------------------------------------------------------------------
# Plotting utility functions for motion planning visualization
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_PATH, '../2Dexamples'))
import robot_plot2D


def plot_objects(ax, obj_lists):
    """Plot objects (obstacles and targets) in task space"""
    for obj in obj_lists:
        ax.add_patch(obj.create_patch())
    return ax


def plot_cspace_trajectory(ax, cdf, q_trajectory, obj_lists, title='QP Planning in C-Space'):
    """
    Plot configuration space trajectory with CDF background
    
    Args:
        ax: matplotlib axis
        cdf: CDF2D instance
        q_trajectory: (T, dof) trajectory in configuration space
        obj_lists: list of objects for CDF visualization
        title: plot title
    """
    # Separate obstacles and targets
    target_objs = [obj for obj in obj_lists if obj.attract]
    obstacle_objs = [obj for obj in obj_lists if not obj.attract]
    
    # Plot CDF background with targets (contour filled)
    if len(target_objs) > 0:
        cdf.plot_cdf(ax=ax, obj_lists=target_objs)
    
    # Plot 0-level set for targets (yellow line)
    if len(target_objs) > 0:
        d_target = cdf.calculate_cdf(cdf.Q_grid, target_objs, method='online_computation').detach().cpu().numpy()
        q0 = cdf.Q_grid[:, 0].detach().cpu().numpy().reshape(cdf.nbData, cdf.nbData)
        q1 = cdf.Q_grid[:, 1].detach().cpu().numpy().reshape(cdf.nbData, cdf.nbData)
        d_target_grid = d_target.reshape(cdf.nbData, cdf.nbData)
        ax.contour(q0, q1, d_target_grid, levels=[0], linewidths=3, colors='yellow', alpha=1.0, label='Target 0-level')
    
    # Plot 0-level set for obstacles (cyan line)
    if len(obstacle_objs) > 0:
        d_obstacle = cdf.calculate_cdf(cdf.Q_grid, obstacle_objs, method='online_computation').detach().cpu().numpy()
        q0 = cdf.Q_grid[:, 0].detach().cpu().numpy().reshape(cdf.nbData, cdf.nbData)
        q1 = cdf.Q_grid[:, 1].detach().cpu().numpy().reshape(cdf.nbData, cdf.nbData)
        d_obstacle_grid = d_obstacle.reshape(cdf.nbData, cdf.nbData)
        ax.contour(q0, q1, d_obstacle_grid, levels=[0], linewidths=3, colors='cyan', alpha=1.0, label='Obstacle 0-level')
    
    # Plot trajectory
    ax.plot(q_trajectory[:, 0], q_trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    ax.plot(q_trajectory[0, 0], q_trajectory[0, 1], 'go', markersize=10, label='Start')
    ax.plot(q_trajectory[-1, 0], q_trajectory[-1, 1], 'r*', markersize=15, label='End')
    
    ax.set_title(title, size=20)
    ax.legend(fontsize=12)
    
    return ax


def plot_task_space_trajectory(ax, robot, q_trajectory, obj_lists, device, 
                                title='QP Planning in Task Space', 
                                xlim=(-4.0, 4.0), ylim=(-4.0, 4.0)):
    """
    Plot task space trajectory with robot configurations
    
    Args:
        ax: matplotlib axis
        robot: Robot2D instance
        q_trajectory: (T, dof) trajectory in configuration space
        obj_lists: list of objects to plot
        device: torch device
        title: plot title
        xlim, ylim: axis limits
    """
    # Plot objects
    plot_objects(ax, obj_lists)
    
    # Plot robot trajectory
    q_traj_torch = torch.from_numpy(q_trajectory).float().to(device)
    if q_traj_torch.dim() == 2:
        q_traj_torch = q_traj_torch.unsqueeze(1)  # (T, 1, dof)
    
    robot.plot_trajectory(ax=ax, joint_trajectory=q_traj_torch)
    
    ax.set_title(title, size=20)
    ax.set_xlabel('x', size=16)
    ax.set_ylabel('y', size=16)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', 'box')
    ax.tick_params(axis='both', labelsize=16)
    
    return ax


def plot_joint_angles(ax, q_trajectories, title='Joint Angles vs Time'):
    """
    Plot joint angles over time
    
    Args:
        ax: matplotlib axis
        q_trajectories: (n_rounds, T, dof) or (T, dof) trajectories
        title: plot title
    """
    if q_trajectories.ndim == 2:
        q_trajectories = q_trajectories[np.newaxis, :, :]  # (1, T, dof)
    
    n_rounds, T, dof = q_trajectories.shape
    
    for i in range(n_rounds):
        q_trajectory = q_trajectories[i]
        time_steps = np.arange(len(q_trajectory))
        color = plt.cm.viridis(i / max(n_rounds, 2))
        
        for j in range(dof):
            label = f'Round {i+1} Joint {j+1}' if n_rounds > 1 else f'Joint {j+1}'
            ax.plot(time_steps, q_trajectory[:, j], 
                   label=label, color=color, linestyle=['-', '--', '-.', ':'][j % 4])
    
    ax.set_xlabel('Time Step', size=16)
    ax.set_ylabel('Joint Angle [rad]', size=16)
    ax.set_title(title, size=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)
    
    return ax


def plot_planning_results(cdf, robot, q_trajectories, obj_lists, device, 
                          save_path, filename_prefix='qp_planning'):
    """
    Create comprehensive visualization of planning results
    
    Args:
        cdf: CDF2D instance
        robot: Robot2D instance
        q_trajectories: (n_rounds, T, dof) trajectories
        obj_lists: list of objects
        device: torch device
        save_path: directory to save figures
        filename_prefix: prefix for saved files
    """
    if q_trajectories.ndim == 2:
        q_trajectories = q_trajectories[np.newaxis, :, :]  # (1, T, dof)
    
    n_rounds = q_trajectories.shape[0]
    
    # Create main figure with 3 subplots
    fig = plt.figure(figsize=(24, 8))
    
    # Subplot 1: C-space trajectory
    ax1 = plt.subplot(1, 3, 1)
    q_concat = np.concatenate(q_trajectories, axis=0)  # (n_rounds*T, dof)
    plot_cspace_trajectory(ax1, cdf, q_concat, obj_lists)
    
    # Subplot 2: Task space trajectory
    ax2 = plt.subplot(1, 3, 2)
    plot_task_space_trajectory(ax2, robot, q_concat, obj_lists, device)
    
    # Subplot 3: Joint angles
    ax3 = plt.subplot(1, 3, 3)
    plot_joint_angles(ax3, q_trajectories)
    
    fig.tight_layout()
    
    # Save figure
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f'{filename_prefix}_full.png')
    fig.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved planning results to {save_file}")
    
    return fig


def plot_cvae_goals(ax, q_goals, q_trajectory, title='CVAE Sampled Goals'):
    """
    Plot CVAE-sampled goals along the trajectory
    
    Args:
        ax: matplotlib axis
        q_goals: (T, dof) sampled goal configurations
        q_trajectory: (T, dof) actual trajectory
        title: plot title
    """
    if q_goals is not None and len(q_goals) > 0:
        q_goals = np.array(q_goals)
        ax.scatter(q_goals[:, 0], q_goals[:, 1], 
                  c='blue', marker='x', s=50, alpha=0.5, 
                  label='CVAE Goals', zorder=5)
        
        # Draw arrows from trajectory to goals
        for i in range(0, len(q_trajectory), max(1, len(q_trajectory)//10)):
            if i < len(q_goals):
                ax.arrow(q_trajectory[i, 0], q_trajectory[i, 1],
                        q_goals[i, 0] - q_trajectory[i, 0],
                        q_goals[i, 1] - q_trajectory[i, 1],
                        head_width=0.1, head_length=0.05, 
                        fc='purple', ec='purple', alpha=0.3)
    
    ax.legend(fontsize=12)
    
    return ax


def plot_distance_over_time(ax, distances, labels=None, title='Distance to Target/Obstacle'):
    """
    Plot distance metrics over time
    
    Args:
        ax: matplotlib axis
        distances: dict of {name: distance_array} or list of distance arrays
        labels: list of labels for each distance array
        title: plot title
    """
    if isinstance(distances, dict):
        for name, dist in distances.items():
            ax.plot(dist, label=name, linewidth=2)
    elif isinstance(distances, list):
        for i, dist in enumerate(distances):
            label = labels[i] if labels and i < len(labels) else f'Distance {i+1}'
            ax.plot(dist, label=label, linewidth=2)
    else:
        ax.plot(distances, label='Distance', linewidth=2)
    
    ax.set_xlabel('Time Step', size=16)
    ax.set_ylabel('Distance', size=16)
    ax.set_title(title, size=20)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=16)
    
    return ax


def create_animation_frames(robot, q_trajectory, obj_lists, device, 
                           save_path, frame_skip=5, xlim=(-4.0, 4.0), ylim=(-4.0, 4.0)):
    """
    Create individual frames for animation
    
    Args:
        robot: Robot2D instance
        q_trajectory: (T, dof) trajectory
        obj_lists: list of objects
        device: torch device
        save_path: directory to save frames
        frame_skip: save every N frames
    """
    os.makedirs(save_path, exist_ok=True)
    
    for i in range(0, len(q_trajectory), frame_skip):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot objects
        plot_objects(ax, obj_lists)
        
        # Plot robot at current configuration
        q_current = torch.from_numpy(q_trajectory[i:i+1]).float().to(device)
        robot.plot_trajectory(ax=ax, joint_trajectory=q_current.unsqueeze(1))
        
        # Plot trajectory up to current point
        if i > 0:
            ax.plot(q_trajectory[:i, 0], q_trajectory[:i, 1], 
                   'r--', alpha=0.3, linewidth=1)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', 'box')
        ax.set_title(f'Step {i}/{len(q_trajectory)}', size=16)
        
        frame_file = os.path.join(save_path, f'frame_{i:04d}.png')
        fig.savefig(frame_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Saved {len(range(0, len(q_trajectory), frame_skip))} frames to {save_path}")
