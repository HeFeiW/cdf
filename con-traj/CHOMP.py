import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class Trajectory:
    def __init__(self, waypoints: np.ndarray, dt: float):
        """
        waypoints: (T, D) trajectory in configuration space
        dt: time step
        """
        self.xi = waypoints
        self.dt = dt
        self.T, self.D = waypoints.shape

class CostFunction(ABC):
    @abstractmethod
    def value(self, traj: Trajectory) -> float:
        pass

    @abstractmethod
    def gradient(self, traj: Trajectory) -> np.ndarray:
        """
        return gradient with shape (T, D)
        """
        pass

class SmoothnessCost(CostFunction):
    def __init__(self, A: np.ndarray):
        """
        A: (TD, TD) smoothness matrix (e.g. finite-diff acceleration)
        """
        self.A = A

    def value(self, traj: Trajectory) -> float:
        xi = traj.xi.reshape(-1)
        return 0.5 * float(xi.T @ (self.A @ xi))

    def gradient(self, traj: Trajectory) -> np.ndarray:
        xi = traj.xi.transpose().reshape(-1)
        grad = self.A @ xi
        grad = grad.reshape(traj.D, traj.T).transpose()
        return grad

class ObstacleCost(CostFunction):
    def __init__(self, environment):
        self.env = environment

    def value(self, traj: Trajectory) -> float:
        T = traj.T
        cost = 0.0
        for t in range(T):
            x = self.env.forward_kinematics(traj.xi[t])
            d = self.env.sdf(x)
            cost += self.env.potential(d)
        return cost * traj.dt

    def gradient(self, traj: Trajectory) -> np.ndarray:
        T, D = traj.T, traj.D
        grad = np.zeros((T, D), dtype=float)
        for t in range(T):
            q = traj.xi[t]
            x = self.env.forward_kinematics(q)
            d = self.env.sdf(x)
            if d < self.env.clearance:
                # ∂c/∂x = -(clearance - d) * ∇d(x)
                grad_d = self.env.sdf_grad(x)
                dc_dx = -(self.env.clearance - d) * grad_d
                # FK Jacobian J = I for point robot in 2D
                grad[t] = dc_dx
        # return grad * traj.dt # TODO: check if dt scaling is needed
        return grad

class Environment(ABC):
    @abstractmethod
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """ q: (D,) -> x: (3,) or (N,3) """
        pass

    @abstractmethod
    def obstacle_cost(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def obstacle_gradient(self, x: np.ndarray) -> np.ndarray:
        """ ∇c(x) """
        pass

class CHOMPOptimizer:
    def __init__(
        self,
        smooth_cost: SmoothnessCost,
        obstacle_cost: ObstacleCost,
        step_size: float,
        A_inv: np.ndarray,
    ):
        self.smooth_cost = smooth_cost
        self.obstacle_cost = obstacle_cost
        self.step_size = step_size
        self.A_inv = A_inv
        # weight for obstacle cost (tuned so that obstacle and smoothness
        # gradients have comparable magnitudes)
        self.lambda_obs = 0.1

    def step(self, traj: Trajectory) -> Trajectory:
        grad_smooth = self.smooth_cost.gradient(traj)
        grad_obs = self.obstacle_cost.gradient(traj)
        # print("Obstacle grad (before scaling) =", grad_obs)
        # print("Smooth grad =", grad_smooth)
        # debug: 
        print("Smooth grad norm:", np.mean(np.linalg.norm(grad_smooth, axis=1)))
        print("Obstacle grad norm:", np.mean(np.linalg.norm(grad_obs, axis=1)))
        grad_total = grad_smooth + grad_obs * self.lambda_obs
        # Covariant gradient descent
        delta = self.A_inv @ grad_total.transpose().reshape(-1)
        delta = delta.reshape(traj.D, traj.T).transpose()

        new_xi = traj.xi - self.step_size * delta
        # Keep endpoints fixed
        new_xi[0] = traj.xi[0]
        new_xi[-1] = traj.xi[-1]
        return Trajectory(new_xi, traj.dt)

    def optimize(self, traj: Trajectory, n_iters: int):
        for k in range(n_iters):
            traj = self.step(traj)
        return traj


# -------------------------- 2D SDF utilities ---------------------------

class Shape2D(ABC):
    @abstractmethod
    def sdf(self, p: np.ndarray) -> float:
        pass

    def grad(self, p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        # Numerical central difference for generality and robustness
        g = np.zeros(2, dtype=float)
        e = np.array([eps, 0.0])
        g[0] = (self.sdf(p + e) - self.sdf(p - e)) / (2 * eps)
        e = np.array([0.0, eps])
        g[1] = (self.sdf(p + e) - self.sdf(p - e)) / (2 * eps)
        n = np.linalg.norm(g)
        if n > 0:
            return g / n
        return g


class Circle2D(Shape2D):
    def __init__(self, center: Tuple[float, float], radius: float):
        self.c = np.asarray(center, dtype=float)
        self.r = float(radius)

    def sdf(self, p: np.ndarray) -> float:
        # Standard signed distance: outside positive, inside negative
        return float(np.linalg.norm(p - self.c) - self.r)

    def grad(self, p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = p - self.c
        n = np.linalg.norm(v)
        if n < eps:
            return np.array([0.0, 0.0])
        return v / n


class Box2D(Shape2D):
    def __init__(self, center: Tuple[float, float], w: float, h: float):
        self.c = np.asarray(center, dtype=float)
        self.hw = float(w) / 2.0
        self.hh = float(h) / 2.0

    def sdf(self, p: np.ndarray) -> float:
        # Signed distance to axis-aligned box: outside positive, inside negative
        # See standard SDF formulation for rectangles.
        local = np.abs(p - self.c) - np.array([self.hw, self.hh])
        # outside distance
        q = np.maximum(local, 0.0)
        dist_out = np.linalg.norm(q)
        # inside distance (negative or zero)
        dist_in = np.minimum(np.maximum(local[0], local[1]), 0.0)
        return float(dist_out + dist_in)
    
    def sample_points(self, n_points: int) -> np.ndarray:
        """Sample n_points on the box boundary (for potential debugging/visualization)."""
        points = []
        for _ in range(n_points):   
            side = np.random.randint(0, 4)
            if side == 0:  # left
                x = self.c[0] - self.hw
                y = np.random.uniform(self.c[1] - self.hh, self.c[1] + self.hh)
            elif side == 1:  # right
                x = self.c[0] + self.hw
                y = np.random.uniform(self.c[1] - self.hh, self.c[1] + self.hh)
            elif side == 2:  # bottom
                x = np.random.uniform(self.c[0] - self.hw, self.c[0] + self.hw)
                y = self.c[1] - self.hh
            else:  # top
                x = np.random.uniform(self.c[0] - self.hw, self.c[0] + self.hw)
                y = self.c[1] + self.hh
            points.append([x, y])
        return np.array(points, dtype=float)
    def grad(self, p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        # Numerical central difference for generality and robustness
        g = np.zeros(2, dtype=float)
        e = np.array([eps, 0.0])
        g[0] = (self.sdf(p + e) - self.sdf(p - e)) / (2 * eps)
        e = np.array([0.0, eps])
        g[1] = (self.sdf(p + e) - self.sdf(p - e)) / (2 * eps)
        n = np.linalg.norm(g)
        if n > 0:
            return g / n
        return g


class Segment2D(Shape2D):
    def __init__(self, a: Tuple[float, float], b: Tuple[float, float], radius: float = 0.0):
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.r = float(radius)
        self.ab = self.b - self.a
        self.ab2 = float(self.ab @ self.ab) if float(self.ab @ self.ab) > 1e-12 else 1e-12

    def sdf(self, p: np.ndarray) -> float:
        # Signed distance to a line segment with optional radius (capsule).
        pa = p - self.a
        t = float(np.clip((pa @ self.ab) / self.ab2, 0.0, 1.0))
        closest = self.a + t * self.ab
        dist = np.linalg.norm(p - closest) - self.r
        return float(dist)
    
    def grad(self, p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        # Numerical central difference for generality and robustness
        g = np.zeros(2, dtype=float)
        e = np.array([eps, 0.0])
        g[0] = (self.sdf(p + e) - self.sdf(p - e)) / (2 * eps)
        e = np.array([0.0, eps])
        g[1] = (self.sdf(p + e) - self.sdf(p - e)) / (2 * eps)
        n = np.linalg.norm(g)
        if n > 0:
            return g / n
        return g
    
    def sample_points(self, n_points: int) -> np.ndarray:
        """Sample n_points on the segment (or capsule) boundary for visualization."""
        points = []
        for _ in range(n_points):
            t = np.random.uniform(0.0, 1.0)
            point_on_line = self.a + t * self.ab
            if self.r > 0:
                angle = np.random.uniform(0.0, 2 * np.pi)
                offset = self.r * np.array([np.cos(angle), np.sin(angle)])
                point_on_boundary = point_on_line + offset
            else:
                point_on_boundary = point_on_line
            points.append(point_on_boundary)
        return np.array(points, dtype=float)

class Environment2D(Environment):
    def __init__(self, shapes: List[Shape2D], clearance: float = 0.05):
        self.shapes = shapes
        self.clearance = float(clearance)

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        # 2D point robot: identity mapping
        return np.asarray(q, dtype=float)

    def obstacle_cost(self, x: np.ndarray) -> float:
        d = self.sdf(x)
        return self.potential(d)

    def obstacle_gradient(self, x: np.ndarray) -> np.ndarray:
        return self.sdf_grad(x)

    # Helpers used by ObstacleCost
    def sdf(self, x: np.ndarray) -> float:
        # Union of obstacles: min signed distance
        ds = [s.sdf(x) for s in self.shapes]
        return float(np.min(ds)) if len(ds) > 0 else float('inf')

    def sdf_grad(self, x: np.ndarray) -> np.ndarray:
        if not self.shapes:
            print("Warning: sdf_grad called with no shapes in environment.")
            return np.zeros(2)
        # Choose the most critical shape (smallest distance)
        idx = int(np.argmin([s.sdf(x) for s in self.shapes]))
        g = self.shapes[idx].grad(x)
        n = np.linalg.norm(g)
        return g / n if n > 0 else g

    def potential(self, d: float) -> float:
        # Quadratic hinge around clearance region
        if d >= self.clearance:
            return 0.0
        elif d < 0.0:
            return 0.5 * self.clearance  - d
        return 0.5 / self.clearance * (self.clearance - d) ** 2


# ----------------------- Smoothness matrix utils -----------------------

def second_difference_matrix(T: int) -> np.ndarray:
    """Build B in R^{(T-2) x T} computing q_{t+1} - 2 q_t + q_{t-1}."""
    if T < 3:
        return np.zeros((0, T))
    B = np.zeros((T - 2, T))
    for i in range(T - 2):
        B[i, i] = 1.0
        B[i, i + 1] = -2.0
        B[i, i + 2] = 1.0
    return B

def first_difference_matrix(T: int) -> np.ndarray:
    """Build D in R^{(T-1) x T} computing q_{t+1} - q_t."""
    if T < 2:
        return np.zeros((0, T))
    D = np.zeros((T - 1, T))
    for i in range(T - 1):
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
    return D

def build_smoothness_A(T: int, D: int, damping: float = 1e-6) -> np.ndarray:
    # B = second_difference_matrix(T)
    B = first_difference_matrix(T)
    C = second_difference_matrix(T)
    w1 = 1.0  # weight for velocity
    w2 = 1.0  # weight for acceleration
    A = B.T @ B * w1 + C.T @ C * w2
    A = np.kron(np.eye(D), A)
    
    A += damping * np.eye(T * D)
    return A


# ----------------------------- Visualization ---------------------------

def plot_environment_and_trajectory(env: Environment2D, trajs: List[Trajectory],
                                    xlim: Tuple[float, float], ylim: Tuple[float, float],
                                    fname: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    # Draw obstacles (approximate outlines)
    for s in env.shapes:
        if isinstance(s, Circle2D):
            circ = plt.Circle(s.c, s.r, color='red', fill=False, linewidth=2)
            ax.add_patch(circ)
        elif isinstance(s, Box2D):
            rect = plt.Rectangle((s.c[0] - s.hw, s.c[1] - s.hh), 2 * s.hw, 2 * s.hh,
                                 color='red', fill=False, linewidth=2)
            ax.add_patch(rect)
        elif isinstance(s, Segment2D):
            ax.plot([s.a[0], s.b[0]], [s.a[1], s.b[1]], 'r-', linewidth=3)
            if s.r > 0:
                # crude capsule visualization by buffering with circles at ends
                ax.add_patch(plt.Circle(s.a, s.r, color='red', fill=False, linewidth=1, alpha=0.6))
                ax.add_patch(plt.Circle(s.b, s.r, color='red', fill=False, linewidth=1, alpha=0.6))

    # Trajectories colored by iteration
    cmap = plt.get_cmap('viridis')
    n_trajs = len(trajs)
    for i, traj in enumerate(trajs):
        P = traj.xi
        # print("Trajectory", i, "points:", P)
        ax.plot(P[:, 0], P[:, 1], '-', color=cmap(i / max(n_trajs - 1, 1)), label=f'Iter {i}')
        
    # plot start and goal respectively in green and blue, . and *
    ax.plot(trajs[0].xi[0, 0], trajs[0].xi[0, 1], 'go', markersize=10, label='Start')
    ax.plot(trajs[0].xi[-1, 0], trajs[0].xi[-1, 1], 'b*', markersize=10, label='Goal')
    ax.legend()
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.set_title('CHOMP 2D Trajectory (color = time)')
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=200)
    else:
        plt.show()
    
if __name__ == "__main__":
    # Problem setup
    T, D = 1000, 2
    # Use a reasonable time step so obstacle gradients are not overly scaled
    dt = 1.0 / (T - 1)
    start = np.array([0.1, 0.1])
    goal = np.array([0.6, 0.9])

    # Linear interpolation initial trajectory
    alphas = np.linspace(0.0, 1.0, T).reshape(-1, 1)
    waypoints = (1 - alphas) * start + alphas * goal
    init_traj = Trajectory(waypoints=waypoints.copy(), dt=dt)

    # Obstacles
    shapes: List[Shape2D] = [
        Circle2D(center=(0.2, 0.4), radius=0.1),
        Circle2D(center=(0.4, 0.6), radius=0.1),
        # Box2D(center=(0.7, 0.3), w=0.2, h=0.2),
        # Segment2D(a=(0.2, 0.8), b=(0.8, 0.6), radius=0.03),
    ]
    env2d = Environment2D(shapes=shapes, clearance=0.04)

    # Smoothness matrix and its inverse (covariant metric)
    A = build_smoothness_A(T, D, damping=1e-4)
    print("Smoothness matrix A shape:", A)
    A_inv = np.linalg.inv(A)
    print("Inverse smoothness matrix A_inv =", A_inv)
    # exit()
    smooth_cost = SmoothnessCost(A)
    obs_cost = ObstacleCost(env2d)

    optimizer = CHOMPOptimizer(
        smooth_cost=smooth_cost,
        obstacle_cost=obs_cost,
        step_size=0.001,
        A_inv=A_inv,
    )

    # Optimize
    n_iters = 100
    trajs_to_plot = [init_traj]
    cur = init_traj
    for k in range(n_iters):
        cur = optimizer.step(cur)
        trajs_to_plot.append(cur)
        # _ = input("Press Enter to continue...")

    # Plot
    plot_environment_and_trajectory(env2d, trajs_to_plot, xlim=(0.0, 1.0), ylim=(0.0, 1.0),
                                    fname='chomp_2d_result.png')
    grad_smooth = optimizer.smooth_cost.gradient(trajs_to_plot[-1])
    grad_obs = optimizer.obstacle_cost.gradient(trajs_to_plot[-1])
    # 画出最后一条轨迹，并在每个点上画出平滑梯度和障碍梯度的箭头
    final_traj = trajs_to_plot[-1]
    grad_obs = grad_obs * 5
    grad_smooth = grad_smooth * 5
    plt.figure(figsize=(8, 8))
    plt.plot(final_traj.xi[:, 0], final_traj.xi[:, 1], 'k-', label='Final Trajectory')
    plt.quiver(final_traj.xi[:, 0], final_traj.xi[:, 1],
               -grad_smooth[:, 0], -grad_smooth[:, 1],
               color='blue', scale=50, width=0.005, label='Smoothness Gradient')
    plt.quiver(final_traj.xi[:, 0], final_traj.xi[:, 1],
               -grad_obs[:, 0], -grad_obs[:, 1],
               color='red', scale=50, width=0.005, label='Obstacle Gradient')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    plt.title('Final Trajectory with Gradients')
    plt.legend()
    plt.show()
    plt.savefig('chomp_2d_final_gradients.png', dpi=200)