# -----------------------------------------------------------------------------
# Moving obstacle scene for CDF visualization
# A circular obstacle moves along a fixed trajectory.
# Generates task-space and C-space (SDF/CDF) visualizations saved as GIFs.
# -----------------------------------------------------------------------------

import matplotlib
matplotlib.use('Agg')   # headless, must be before pyplot import

import numpy as np
import os
import sys
import math
import copy
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# Make sure local modules are importable
_CUR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _CUR)
sys.path.insert(0, os.path.join(_CUR, '..', '..', 'RDF'))
from robot2D_torch import Robot2D
from primitives2D_torch import Circle, Box
import robot_plot2D

# Reuse neural-CDF helpers from chomp_cdf_tracking
from chomp_cdf_tracking import load_cdf_net, inference_cdf_nn, cspace_grid, DEFAULT_MODEL_PATH

PI = math.pi
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGE_DIR = os.path.join(CUR_PATH, 'image')
os.makedirs(IMAGE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Robot setup  (same 2-DOF arm as example.py)
# ---------------------------------------------------------------------------
def make_robot(device):
    link_lengths = torch.tensor([[2.0, 2.0]]).float().to(device)
    link_parent_map = {1: 0, 2: 1}
    init_states = torch.zeros((1, 2)).to(device)
    robot = Robot2D(
        num_links=2,
        init_states=init_states,
        link_lengths=link_lengths,
        device=device,
        link_parent_map=link_parent_map,
    )
    return robot



# ---------------------------------------------------------------------------
# Parametric obstacle trajectories
# ---------------------------------------------------------------------------
def trajectory_circle(t, cx=0.0, cy=0.0, r=2.2):
    """Uniform circular orbit around (cx, cy)."""
    x = cx + r * math.cos(2 * PI * t)
    y = cy + r * math.sin(2 * PI * t)
    return float(x), float(y)


def trajectory_figure8(t, scale=2.0):
    """Lemniscate-of-Gerono figure-eight path."""
    x = scale * math.cos(2 * PI * t)
    y = scale * math.sin(2 * PI * t) * math.cos(2 * PI * t) + 2.0
    return float(x), float(y)


def trajectory_linear_bounce(t, x0=-2.5, x1=2.5, y=1.5):
    """Obstacle bounces back and forth along a horizontal line."""
    phase = (t * 2) % 2.0          # 0→1: left-to-right, 1→2: right-to-left
    if phase <= 1.0:
        x = x0 + (x1 - x0) * phase
    else:
        x = x1 - (x1 - x0) * (phase - 1.0)
    return float(x), float(y)


# ---------------------------------------------------------------------------
# Robot drawing helper (simple line segments)
# ---------------------------------------------------------------------------
def draw_robot(ax, robot, q, color='steelblue', lw=3):
    """Draw robot links as thick lines on ax, at joint angles q (1-D tensor)."""
    q_batch = q.unsqueeze(0) if q.dim() == 1 else q      # (1, dof)
    link_pos = robot.forward_kinematics_all_links(q_batch).squeeze(0)  # (2, n_links+1)
    link_pos_np = link_pos.cpu().numpy()
    for li in range(robot.num_links):
        parent_idx = robot.link_parent_map[li + 1]
        xs = [link_pos_np[0, parent_idx], link_pos_np[0, li + 1]]
        ys = [link_pos_np[1, parent_idx], link_pos_np[1, li + 1]]
        ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle='round')
        ax.plot(xs[1], ys[1], 'o', color=color, markersize=7)
    # base
    ax.plot(0, 0, 'ko', markersize=9)


# ---------------------------------------------------------------------------
# Core animation builder
# ---------------------------------------------------------------------------
def build_frames(device, num_frames=60, trajectory_fn=trajectory_circle,
                 obs_radius=0.4, nbData=50, q_display=None,
                 model_path=DEFAULT_MODEL_PATH):
    """
    Render `num_frames` side-by-side (task space | C-space neural-CDF) images.
    Returns list of (H, W, 3) uint8 numpy arrays.
    """
    robot = make_robot(device)
    net = load_cdf_net(model_path, device)

    if q_display is None:
        q_display = torch.tensor([0.5, 0.9]).float().to(device)

    # Build obstacle trajectory path for drawing
    traj_pts = np.array([trajectory_fn(i / num_frames) for i in range(num_frames + 1)])

    frames = []
    print(f"  Rendering {num_frames} frames …")
    for i_frame in range(num_frames):
        t = i_frame / num_frames

        # ── current obstacle position ──────────────────────────────────────
        cx, cy = trajectory_fn(t)
        obs = Circle(
            center=torch.tensor([cx, cy]).float().to(device),
            radius=obs_radius,
            attract=False,
            device=device,
        )

        # ── Neural CDF on C-space grid (reuses chomp_cdf_tracking.cspace_grid) ──
        sdf_2d, q0_np, q1_np = cspace_grid(net, nbData, obs, device)

        # ── Figure setup ──────────────────────────────────────────────────
        fig, (ax_task, ax_cdf) = plt.subplots(1, 2, figsize=(14, 6.5))

        # ============================================================
        # Left: Task Space
        # ============================================================
        ax_task.set_aspect('equal', 'box')
        ax_task.set_xlim(-4.5, 4.5)
        ax_task.set_ylim(-4.5, 4.5)
        ax_task.set_title('Task Space', size=18, fontweight='bold')
        ax_task.set_xlabel('x', size=14)
        ax_task.set_ylabel('y', size=14)
        ax_task.tick_params(labelsize=12)
        ax_task.grid(True, alpha=0.25)

        # Draw the full planned trajectory (faint dashed)
        ax_task.plot(traj_pts[:, 0], traj_pts[:, 1],
                     '--', color='salmon', linewidth=1.5, alpha=0.55, label='Trajectory path')

        # Draw past portion (darker)
        past = traj_pts[:i_frame + 1]
        ax_task.plot(past[:, 0], past[:, 1],
                     '-', color='tomato', linewidth=2.0, alpha=0.8)

        # Draw obstacle circle
        circle_patch = mpatches.Circle((cx, cy), obs_radius,
                                        linewidth=2.5, edgecolor='red',
                                        facecolor='#FF7F7F', alpha=0.7, zorder=5)
        ax_task.add_patch(circle_patch)
        ax_task.plot(cx, cy, 'r+', markersize=10, markeredgewidth=2, zorder=6)

        # Draw robot at display pose
        draw_robot(ax_task, robot, q_display, color='steelblue', lw=3)

        ax_task.text(0.02, 0.97,
                     f'Frame {i_frame + 1:3d} / {num_frames}',
                     transform=ax_task.transAxes, fontsize=11,
                     verticalalignment='top')

        # ============================================================
        # Right: C-space SDF
        # ============================================================
        ax_cdf.set_aspect('equal', 'box')
        ax_cdf.set_xlim(-PI, PI)
        ax_cdf.set_ylim(-PI, PI)
        ax_cdf.set_title('Configuration Space CDF (neural)', size=18, fontweight='bold')
        ax_cdf.set_xlabel('q₁ (rad)', size=14)
        ax_cdf.set_ylabel('q₂ (rad)', size=14)
        ax_cdf.tick_params(labelsize=12)

        # Colour-mapped filled contour
        vmax = max(np.max(np.abs(sdf_2d)), 1e-6)
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)
        cf = ax_cdf.contourf(q0_np, q1_np, sdf_2d,
                             levels=20, cmap='coolwarm', norm=norm)

        # Zero-level set (collision boundary in C-space)
        ax_cdf.contour(q0_np, q1_np, sdf_2d,
                       levels=[0], linewidths=2.5, colors='black')

        # Mark the current robot configuration
        q_disp_np = q_display.cpu().numpy()
        ax_cdf.plot(q_disp_np[0], q_disp_np[1],
                    '*', color='#00CC00', markersize=14, zorder=10,
                    label=f'Robot pose ({q_disp_np[0]:.2f}, {q_disp_np[1]:.2f})')
        ax_cdf.legend(loc='lower right', fontsize=10)

        # Colour bar
        cbar = fig.colorbar(cf, ax=ax_cdf, fraction=0.046, pad=0.04)
        cbar.set_label('CDF value (neural)', size=11)

        # ── Finalize frame ─────────────────────────────────────────────────
        fig.suptitle(f'Moving Obstacle – t = {t:.2f}', size=16, y=1.01)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        buf = buf.reshape(h, w, 4)[:, :, :3].copy()   # RGBA → RGB
        frames.append(buf)
        plt.close(fig)

        if (i_frame + 1) % 10 == 0 or i_frame == 0:
            print(f'    frame {i_frame + 1:3d}/{num_frames} done')

    return frames


# ---------------------------------------------------------------------------
# GIF saver
# ---------------------------------------------------------------------------
def save_gif(frames, path, fps=12):
    try:
        import imageio
        imageio.mimsave(path, frames, fps=fps, loop=0)
        print(f'  ✓ Saved → {path}')
    except ImportError:
        # Fallback: save individual PNGs
        print('  imageio not found – saving individual PNG frames instead.')
        base = path.replace('.gif', '')
        os.makedirs(base, exist_ok=True)
        for i, frame in enumerate(frames):
            import PIL.Image
            PIL.Image.fromarray(frame).save(os.path.join(base, f'frame_{i:03d}.png'))
        print(f'  Frames saved to {base}/')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Moving obstacle CDF animation')
    parser.add_argument('--device', type=str, default='auto',
                        help='cpu / cuda:0 / auto')
    parser.add_argument('--frames', type=int, default=60,
                        help='Number of animation frames')
    parser.add_argument('--fps', type=int, default=12,
                        help='GIF frames per second')
    parser.add_argument('--traj', type=str, default='circle',
                        choices=['circle', 'figure8', 'bounce'],
                        help='Obstacle trajectory shape')
    parser.add_argument('--radius', type=float, default=0.4,
                        help='Obstacle radius')
    parser.add_argument('--nbData', type=int, default=50,
                        help='C-space grid resolution per axis')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
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
    traj_fn = traj_map[args.traj]
    tag = args.traj

    print(f'\n=== Building animation: {tag} ({args.frames} frames) ===')
    frames = build_frames(
        device,
        num_frames=args.frames,
        trajectory_fn=traj_fn,
        obs_radius=args.radius,
        nbData=args.nbData,
        model_path=args.model_path,
    )

    gif_path = os.path.join(IMAGE_DIR, f'moving_obs_{tag}.gif')
    save_gif(frames, gif_path, fps=args.fps)

    print('\nDone.  Output:')
    print(f'  {gif_path}')
