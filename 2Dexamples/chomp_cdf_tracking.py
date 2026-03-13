# -----------------------------------------------------------------------------
# CHOMP + CDF tracking: plan a robot arm trajectory where each waypoint satisfies
#   CDF(q_t, obs_t) = 0  (robot surface touches moving obstacle at each step)
#
# Problem setup
# -------------
#   - 2-DOF planar arm, T waypoints in C-space  (q ∈ R²)
#   - Circular obstacle moves along a parametric trajectory
#   - Waypoint t ↔ obstacle position at phase  t / (T-1)
#   - Cost  =  smoothness prior (CHOMP, fixed endpoints)
#   - Constraint at each free waypoint t:  h_t(q_t) = SDF_task(robot(q_t), obs_t) = 0
#
# Optimizer
# ---------
#   At each iteration (= one animation frame):
#     1. Covariant gradient descent on smoothness cost
#     2. Per-waypoint Newton projection onto CDF = 0
#          q_t  ←  q_t  -  h_t · ∇h_t / ‖∇h_t‖²
#   Fixed endpoints are restored after every update.
#
# Output: GIF  image/chomp_cdf_tracking_<traj>.gif
# -----------------------------------------------------------------------------

import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import sys
import math
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CUR_PATH_EARLY = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CUR_PATH_EARLY)
sys.path.insert(0, os.path.join(CUR_PATH_EARLY, '..', '..', 'RDF'))
from robot2D_torch import Robot2D
from primitives2D_torch import Circle
from Siren import Siren

PI = math.pi
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGE_DIR = os.path.join(CUR_PATH, 'image')
os.makedirs(IMAGE_DIR, exist_ok=True)

DEFAULT_MODEL_PATH = os.path.join(CUR_PATH, 'model_dict', 'siren_lr1e4_eik1_ep5000_model22.pth')


# ============================================================
# Neural CDF helpers
# ============================================================

def load_cdf_net(model_path: str, device):
    """Load a saved Siren CDF model (input dim=4: obs_x, obs_y, q1, q2)."""
    net = torch.load(model_path, map_location=device)
    net.eval()
    print(f'  CDF model loaded from {model_path}')
    return net


def inference_cdf_nn(net, q, obs, device):
    """Neural CDF query.

    net : Siren model  (input 4-D, output 1-D)
    q   : (B, 2) tensor  (robot configs, requires_grad allowed)
    obs : Circle  (provides .center)
    Returns sdf_like : (B,) tensor
    """
    B = q.shape[0]
    obs_center = obs.center.detach()            # (2,)
    obs_rep = obs_center.unsqueeze(0).expand(B, -1)  # (B, 2)
    inputs = torch.cat([obs_rep, q], dim=-1)    # (B, 4)
    out = net.forward(inputs)
    if isinstance(out, tuple):
        out = out[0]
    return out.squeeze(-1)                      # (B,)


# ============================================================
# Robot helpers  (retained for task-space visualisation)
# ============================================================

def make_robot(device):
    link_lengths = torch.tensor([[2.0, 2.0]]).float().to(device)
    link_parent_map = {1: 0, 2: 1}
    robot = Robot2D(
        num_links=2,
        init_states=torch.zeros((1, 2)).to(device),
        link_lengths=link_lengths,
        device=device,
        link_parent_map=link_parent_map,
    )
    return robot


def cdf_value_and_grad(net, q_np, obs, device):
    """Neural CDF scalar value + gradient w.r.t. q for a single waypoint.

    net  : Siren CDF model
    q_np : (D,) numpy array
    obs  : Circle  (provides .center)
    Returns
        h    : float   (CDF value)
        grad : (D,) numpy array
    """
    q_t = torch.tensor(q_np, dtype=torch.float32, device=device, requires_grad=True)
    h = inference_cdf_nn(net, q_t.unsqueeze(0), obs, device)  # (1,)
    h.backward()
    return float(h.detach().cpu()), q_t.grad.detach().cpu().numpy()


# ============================================================
# Obstacle trajectories  (exactly as in moving_obstacle_scene.py)
# ============================================================

def trajectory_circle(t, cx=0.0, cy=0.0, r=2.2):
    return cx + r * math.cos(2*PI*t), cy + r * math.sin(2*PI*t)

def trajectory_figure8(t, scale=2.0):
    return scale * math.cos(2*PI*t), scale * math.sin(2*PI*t) * math.cos(2*PI*t) + 2.0

def trajectory_linear_bounce(t, x0=-2.5, x1=2.5, y=1.5):
    phase = (t * 2) % 2.0
    x = x0 + (x1-x0)*phase if phase <= 1.0 else x1 - (x1-x0)*(phase-1.0)
    return float(x), float(y)


def make_obs(traj_fn, t_norm, obs_radius, device):
    cx, cy = traj_fn(t_norm)
    return Circle(
        center=torch.tensor([cx, cy], dtype=torch.float32).to(device),
        radius=obs_radius,
        attract=False,
        device=device,
    )


# ============================================================
# CHOMP smoothness matrices  (numpy, for free waypoints only)
# ============================================================

def first_diff_matrix(n):
    """D in R^{(n-1) x n}: computes q_{t+1} - q_t."""
    D = np.zeros((n-1, n))
    D[np.arange(n-1), np.arange(n-1)] = -1.0
    D[np.arange(n-1), np.arange(1, n)] = 1.0
    return D

def second_diff_matrix(n):
    """B in R^{(n-2) x n}: computes q_{t+1} - 2q_t + q_{t-1}."""
    if n < 3:
        return np.zeros((0, n))
    B = np.zeros((n-2, n))
    for i in range(n-2):
        B[i, i], B[i, i+1], B[i, i+2] = 1.0, -2.0, 1.0
    return B

def build_smoothness_A(n_free, D, damping=1e-6):
    """
    Smoothness matrix A for n_free free waypoints (both endpoints excluded).
    We treat boundary conditions from the fixed q_start and q_end.
    """
    D1 = first_diff_matrix(n_free)
    D2 = second_diff_matrix(n_free)
    A_scalar = D1.T @ D1 + D2.T @ D2
    A = np.kron(A_scalar, np.eye(D))
    A += damping * np.eye(n_free * D)
    return A


# ============================================================
# Visualisation helpers
# ============================================================

def cspace_grid(net, nbData, obs, device, chunk=512):
    """Evaluate neural CDF on a regular C-space grid."""
    t = torch.linspace(-PI, PI, nbData)
    q0mg, q1mg = torch.meshgrid(t, t, indexing='ij')
    Q = torch.stack([q0mg.reshape(-1), q1mg.reshape(-1)], dim=1).to(device)
    rows = []
    for i in range(0, len(Q), chunk):
        with torch.no_grad():
            rows.append(inference_cdf_nn(net, Q[i:i+chunk], obs, device).cpu().numpy())
    return np.concatenate(rows).reshape(nbData, nbData), q0mg.numpy(), q1mg.numpy()


def draw_robot_arm(ax, robot, q_np, color='steelblue', lw=2.5, alpha=1.0):
    q = torch.tensor(q_np, dtype=torch.float32).unsqueeze(0).to(robot.device)
    lp = robot.forward_kinematics_all_links(q).squeeze(0).cpu().numpy()  # (2, n+1)
    for li in range(robot.num_links):
        par = robot.link_parent_map[li+1]
        ax.plot([lp[0, par], lp[0, li+1]], [lp[1, par], lp[1, li+1]],
                color=color, lw=lw, solid_capstyle='round', alpha=alpha)
    ax.plot(0, 0, 'ko', ms=6)


def render_frame(robot, net, xi, obs_list, traj_pts_task, nbData, device,
                 frame_idx, n_iters, cdf_vals):
    """
    xi       : (T, 2)  current trajectory
    obs_list : list of Circle objects (one per waypoint)
    traj_pts_task : (T, 2) obstacle center positions (task space)
    cdf_vals : (T-2,) CDF values for free waypoints
    """
    T = xi.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # ── Left: C-space ────────────────────────────────────────
    ax = axes[0]
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-PI, PI)
    ax.set_ylim(-PI, PI)
    ax.set_title('C-space   (trajectory + CDF zero-level sets)', size=15, fontweight='bold')
    ax.set_xlabel('q₁ (rad)', size=13)
    ax.set_ylabel('q₂ (rad)', size=13)
    ax.tick_params(labelsize=11)

    # Draw trajectory
    ax.plot(xi[:, 0], xi[:, 1], 'o-', color='royalblue', lw=2,
            ms=5, zorder=4, label='Trajectory')
    ax.plot(xi[0, 0],  xi[0, 1],  's', color='green',  ms=11, zorder=5, label='Start')
    ax.plot(xi[-1, 0], xi[-1, 1], '*', color='orange', ms=14, zorder=5, label='End')

    # Draw CDF zero-level set for a selection of waypoints (colouring by index)
    cmap = plt.get_cmap('plasma')
    step = max(1, T // 6)
    for t_idx in range(0, T, step):
        obs_t = obs_list[t_idx]
        sdf_2d, q0g, q1g = cspace_grid(net, nbData, obs_t, device)
        color = cmap(t_idx / max(T-1, 1))
        ax.contour(q0g, q1g, sdf_2d, levels=[0], colors=[color],
                   linewidths=1.8, alpha=0.7)

    ax.legend(loc='lower right', fontsize=10)

    # CDF residual info
    mean_cdf = float(np.mean(np.abs(cdf_vals)))
    ax.text(0.02, 0.97,
            f'Iter {frame_idx+1:3d}/{n_iters}   |CDF| mean={mean_cdf:.4f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # ── Right: Task space ─────────────────────────────────────
    ax = axes[1]
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_title('Task space   (robot poses + obstacle positions)', size=15, fontweight='bold')
    ax.set_xlabel('x', size=13)
    ax.set_ylabel('y', size=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.2)

    # Draw obstacle trajectory (all)
    ax.plot(traj_pts_task[:, 0], traj_pts_task[:, 1],
            '--', color='salmon', lw=1.5, alpha=0.5, label='Obstacle path')

    # Draw obstacle circles at sampled waypoints
    for t_idx in range(0, T, step):
        obs_t = obs_list[t_idx]
        color = cmap(t_idx / max(T-1, 1))
        c = obs_t.center.cpu().numpy()
        p = mpatches.Circle(c, obs_t.radius,
                            edgecolor=color, facecolor='none',
                            lw=2.0, alpha=0.8)
        ax.add_patch(p)

    # Draw robot arm at each waypoint (faded)
    for t_idx in range(0, T, step):
        color = cmap(t_idx / max(T-1, 1))
        draw_robot_arm(ax, robot, xi[t_idx], color=color, lw=2, alpha=0.6)

    # Draw start / end robot more prominently
    draw_robot_arm(ax, robot, xi[0],  color='green',  lw=3.5)
    draw_robot_arm(ax, robot, xi[-1], color='orange', lw=3.5)

    ax.text(0.02, 0.97, f'Iter {frame_idx+1:3d}/{n_iters}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    ax.legend(loc='lower right', fontsize=10)

    fig.suptitle('CHOMP + CDF=0 Tracking  —  Moving Circular Obstacle', size=14)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    frame_rgb = buf.reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    return frame_rgb


# ============================================================
# Main planning loop
# ============================================================

def plan_and_animate(
    device,
    traj_fn=trajectory_circle,
    obs_radius: float = 0.4,
    T: int = 30,           # number of waypoints (= planning horizon)
    n_iters: int = 80,     # CHOMP iterations (= animation frames)
    step_size: float = 0.05,
    lambda_proj: float = 1.0,  # projection strength (>1 → faster but noisier)
    nbData: int = 40,      # C-space grid resolution for visualisation
    q_start=None,
    q_end=None,
    tag: str = 'circle',
    model_path: str = DEFAULT_MODEL_PATH,
):
    """
    Plan a T-waypoint C-space trajectory satisfying CDF(q_t, obs_t) = 0.
    Returns list of RGB frames (one per CHOMP iteration).
    """
    D = 2

    # ── Default endpoints ─────────────────────────────────────
    if q_start is None:
        q_start = np.array([-PI / 3,  PI / 4], dtype=float)
    if q_end is None:
        q_end   = np.array([ PI / 2, -PI / 3], dtype=float)

    # ── Load neural CDF model ──────────────────────────────────
    net = load_cdf_net(model_path, device)

    # ── Build obstacle list (one per waypoint) ─────────────────
    robot = make_robot(device)
    obs_list = [make_obs(traj_fn, t / (T - 1), obs_radius, device) for t in range(T)]
    traj_pts_task = np.array([obs_list[t].center.cpu().numpy() for t in range(T)])

    # ── Initialise trajectory (linear interp in C-space) ───────
    alphas = np.linspace(0.0, 1.0, T).reshape(-1, 1)
    xi = (1 - alphas) * q_start + alphas * q_end       # (T, D)

    # ── CHOMP smoothness matrix for free waypoints ──────────────
    n_free = T - 2    # waypoints 1 .. T-2
    A = build_smoothness_A(n_free, D)
    A_inv = np.linalg.inv(A)

    # Boundary correction: smoothness gradient receives contribution from
    # fixed endpoints through the finite-difference operator
    # g_smooth = A @ xi_free_flat + b_boundary
    # b from velocity (q1 - q_start) and (q_{T-2} - q_end) terms
    def smoothness_grad(xi_free_flat):
        """Gradient of  1/2 xi^T A xi  for free vars, incl. boundary terms."""
        g = A @ xi_free_flat
        # velocity term at left boundary: first free waypoint interacts with q_start
        g[:D] -= q_start     # -(q_start) contribution from D1^T D1
        # velocity term at right boundary: last free waypoint interacts with q_end
        g[-D:] -= q_end      # -(q_end) contribution from D1^T D1
        return g

    # ── Animate CHOMP iterations ───────────────────────────────
    frames = []
    print(f'  Planning: T={T} waypoints, {n_iters} CHOMP iterations …')

    for iteration in range(n_iters):

        # -------- 1. Smoothness covariant gradient step --------
        xi_free = xi[1:T-1].copy()                # (n_free, D)
        g = smoothness_grad(xi_free.reshape(-1))  # (n_free*D,)
        delta = A_inv @ g                         # (n_free*D,)
        xi_free_new = xi_free - step_size * delta.reshape(n_free, D)

        xi[1:T-1] = xi_free_new
        # Restore fixed endpoints
        xi[0]  = q_start
        xi[-1] = q_end

        # -------- 2. Per-waypoint CDF=0 projection --------
        cdf_vals = np.zeros(n_free)
        for t_idx in range(1, T - 1):
            obs_t = obs_list[t_idx]
            h, grad_h = cdf_value_and_grad(net, xi[t_idx], obs_t, device)
            cdf_vals[t_idx - 1] = h
            denom = float(np.dot(grad_h, grad_h))
            if denom > 1e-10:
                # Newton step: project onto h = 0
                xi[t_idx] = xi[t_idx] - lambda_proj * h * grad_h / denom
                # Clip to joint limits [-π, π]
                xi[t_idx] = np.clip(xi[t_idx], -PI, PI)

        # Restore endpoints again (projection shouldn't touch them, but safety)
        xi[0]  = q_start
        xi[-1] = q_end

        # -------- 3. Render frame --------
        frame = render_frame(robot, net, xi.copy(), obs_list, traj_pts_task,
                             nbData, device, iteration, n_iters, cdf_vals)
        frames.append(frame)

        # Progress log
        mean_cdf = float(np.mean(np.abs(cdf_vals)))
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f'    iter {iteration+1:3d}/{n_iters}  |CDF| mean = {mean_cdf:.5f}')

    return frames, xi


# ============================================================
# GIF saver
# ============================================================

def save_gif(frames, path, fps=10):
    try:
        import imageio
        imageio.mimsave(path, frames, fps=fps, loop=0)
        print(f'  Saved → {path}')
    except ImportError:
        out_dir = path.replace('.gif', '_frames')
        os.makedirs(out_dir, exist_ok=True)
        try:
            from PIL import Image
        except ImportError:
            print('  Neither imageio nor Pillow found. Install one to save output.')
            return
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(out_dir, f'frame_{i:03d}.png'))
        print(f'  Frames saved to {out_dir}/')


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='CHOMP + CDF=0 tracking of a moving circular obstacle')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--traj', default='circle',
                        choices=['circle', 'figure8', 'bounce'])
    parser.add_argument('--T',          type=int,   default=30,
                        help='Number of waypoints')
    parser.add_argument('--iters',      type=int,   default=80,
                        help='CHOMP iterations (= animation frames)')
    parser.add_argument('--step',       type=float, default=0.05,
                        help='CHOMP step size')
    parser.add_argument('--lam',        type=float, default=1.0,
                        help='Projection strength λ')
    parser.add_argument('--radius',     type=float, default=0.4,
                        help='Obstacle radius')
    parser.add_argument('--fps',        type=int,   default=10)
    parser.add_argument('--nbData',     type=int,   default=40,
                        help='C-space grid resolution for vis')
    parser.add_argument('--model_path', type=str,   default=DEFAULT_MODEL_PATH,
                        help='Path to the Siren CDF model (.pth)')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')

    traj_map = {
        'circle':  trajectory_circle,
        'figure8': trajectory_figure8,
        'bounce':  trajectory_linear_bounce,
    }

    frames, final_xi = plan_and_animate(
        device=device,
        traj_fn=traj_map[args.traj],
        obs_radius=args.radius,
        T=args.T,
        n_iters=args.iters,
        step_size=args.step,
        lambda_proj=args.lam,
        nbData=args.nbData,
        tag=args.traj,
        model_path=args.model_path,
    )

    gif_path = os.path.join(IMAGE_DIR, f'chomp_cdf_tracking_{args.traj}.gif')
    save_gif(frames, gif_path, fps=args.fps)

    # Print final CDF residuals
    net_final = load_cdf_net(args.model_path, device)
    obs_list = [make_obs(traj_map[args.traj], t / (args.T - 1), args.radius, device)
                for t in range(args.T)]
    print('\nFinal CDF residuals per waypoint:')
    for t in range(1, args.T - 1):
        h, _ = cdf_value_and_grad(net_final, final_xi[t], obs_list[t], device)
        print(f'  waypoint {t:2d}: h = {h:+.5f}')

    print(f'\nDone.  GIF → {gif_path}')
