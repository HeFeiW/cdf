import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from chomp_config_loader import CHOMPConfig

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
        xi = traj.xi.reshape(-1)
        grad = self.A @ xi
        return grad.reshape(traj.T, traj.D)

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
        lambda_obs: float = 0.1,
    ):
        self.smooth_cost = smooth_cost
        self.obstacle_cost = obstacle_cost
        self.step_size = step_size
        self.A_inv = A_inv
        # weight for obstacle cost (tuned so that obstacle and smoothness
        # gradients have comparable magnitudes)
        self.lambda_obs = lambda_obs

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
        delta = self.A_inv @ grad_total.reshape(-1)
        delta = delta.reshape(traj.T, traj.D)

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
    A = np.kron(A, np.eye(D))
    
    A += damping * np.eye(T * D)
    return A


def build_smoothness_A_free_end(n: int, D: int, damping: float = 1e-6) -> np.ndarray:
    """Build smoothness matrix for fixed-start, free-end boundary conditions.

    The trajectory is parameterised as xi = (q_1, ..., q_n), where q_0 (start)
    is a known constant **excluded** from the optimisation variables.  The end
    point q_n is a free variable that can be moved by the goal-set constraint.

    The finite-difference (velocity) operator is defined as
        K[0, :] = [1, 0, ..., 0]          (encodes q_1 - q_0, q_0 absorbed into e)
        K[i, i-1] = -1, K[i, i] = 1       for i = 1, ..., n-1

    This lower-bidiagonal K gives  A_scalar = K^T K  which is tridiagonal:
        diagonal:     [2, 2, ..., 2, 1]
        off-diagonal: [-1, -1, ..., -1]

    Its inverse satisfies  A_scalar^{-1}[i, j] = min(i, j) + 1  (0-indexed),
    so the (n-1, n-1) entry equals  n  — this is the scalar β used in eq. 14.

    Parameters
    ----------
    n : int
        Number of free waypoints (T_total - 1, excluding the fixed start q_0).
    D : int
        Configuration-space dimension.
    damping : float
        Small regularisation added to the diagonal for numerical stability.

    Returns
    -------
    A : np.ndarray, shape (n*D, n*D)
    """
    # Lower-bidiagonal K (scalar, n×n)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 1.0
        if i > 0:
            K[i, i - 1] = -1.0

    A_scalar = K.T @ K                           # (n, n)
    A = np.kron(A_scalar, np.eye(D))             # (n*D, n*D)
    A += damping * np.eye(n * D)
    return A


# ==================== Goal-Set Constraint interface ====================

class GoalSetConstraint(ABC):
    """Abstract interface for a goal-set constraint applied to the end point.

    The constraint is  h(q_n) = 0  where q_n is the last waypoint.
    Subclasses must implement :meth:`query` which returns both the constraint
    value **and** its Jacobian at queried point.

    This interface is intentionally minimal: it is only a *query* oracle for
    the endpoint; it does not modify the trajectory itself.
    """

    @abstractmethod
    def query(self, q_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the goal-set constraint at the endpoint q_n.

        Parameters
        ----------
        q_n : np.ndarray, shape (D,)
            Current end-point configuration.

        Returns
        -------
        h : np.ndarray, shape (k,)
            Constraint residual.  The constraint is satisfied when h = 0.
        C_tilde : np.ndarray, shape (k, D)
            Jacobian  dh/dq_n  evaluated at q_n.
        """
        pass


# ==================== CHOMP with Goal-Set Constraints ====================

class CHOMPGoalSetOptimizer:
    """CHOMP optimiser extended for goal-set constraints (Dragan et al. 2011).

    The trajectory is represented as xi = (q_0, q_1, ..., q_n) with q_0 fixed.
    The smoothness matrix A is built for the *free* sub-trajectory
    xi_free = (q_1, ..., q_n) using free-end boundary conditions so that
    A^{-1} naturally propagates endpoint corrections to the whole trajectory.

    Two constrained update methods are provided:

    * :meth:`step_general` — the general constrained update (eq. 11 in the
      paper), which builds the full constraint Jacobian
      ``C = [0, ..., 0, C_tilde]`` and solves the projected gradient step
      without additional assumptions on the structure of A^{-1}.

    * :meth:`step_goalset` — the simplified closed-form update (eq. 14 in
      the paper), which exploits the goal-set structure via the last-D-rows
      block ``B_n`` of A^{-1} and the scalar  β = A^{-1}[n-1,n-1] / I_D.
      This is the version recommended for pure goal-set problems.

    Parameters
    ----------
    obstacle_cost : ObstacleCost
        Obstacle cost computed over the full trajectory.
    step_size : float
        Gradient-descent step size η.
    A_free : np.ndarray, shape (n*D, n*D)
        Smoothness matrix for the free variables (build with
        :func:`build_smoothness_A_free_end`).
    q_start : np.ndarray, shape (D,)
        Fixed start configuration q_0.  Used to compute the boundary
        correction term in the smoothness gradient.
    lambda_obs : float
        Weight applied to the obstacle gradient.
    """

    def __init__(
        self,
        obstacle_cost: ObstacleCost,
        step_size: float,
        A_free: np.ndarray,
        q_start: np.ndarray,
        lambda_obs: float = 0.1,
    ):
        self.obstacle_cost = obstacle_cost
        self.step_size = step_size
        self.A_free = A_free
        self.A_free_inv = np.linalg.inv(A_free)
        self.q_start = np.asarray(q_start, dtype=float)
        self.lambda_obs = lambda_obs

        nD = A_free.shape[0]
        self.D = self.q_start.shape[0]
        assert nD % self.D == 0, "A_free shape is inconsistent with D"
        self.n = nD // self.D   # number of free waypoints

        # Boundary correction for the smoothness gradient:
        #   grad_smooth = A_free @ xi_free_flat + b_boundary
        # where b_boundary = K^T @ e_free, e_free[0] = -q_start, rest 0.
        # Since K is lower-bidiagonal with K[0] = [1, 0, ...],
        # K^T @ e_free = [-q_start, 0, ..., 0] (first D entries only).
        self._b_boundary = np.zeros(nD, dtype=float)
        self._b_boundary[: self.D] = -self.q_start

        # Pre-extract B_n for eq. 14: last D rows of A_free_inv
        #   shape (D, n*D)
        self._B_n = self.A_free_inv[(self.n - 1) * self.D :, :]

        # β scalar: A_free_inv[n-1, n-1] (block) should be β * I_D
        B_nn = self._B_n[:, (self.n - 1) * self.D :]   # (D, D)
        self._beta = float(B_nn[0, 0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _free_vars(self, traj: Trajectory) -> np.ndarray:
        """Extract free variables xi_free = traj.xi[1:] flattened."""
        return traj.xi[1:].reshape(-1)   # (n*D,)

    def _smoothness_gradient(self, xi_free_flat: np.ndarray) -> np.ndarray:
        """Gradient of the smoothness prior w.r.t. xi_free (n*D,).

        grad = A_free @ xi_free + b_boundary
        The boundary term encodes the fixed start q_0 via K^T @ e_free.
        """
        return self.A_free @ xi_free_flat + self._b_boundary

    def _obstacle_gradient(self, traj: Trajectory) -> np.ndarray:
        """Obstacle gradient w.r.t. free variables, flattened (n*D,)."""
        full_grad = self.obstacle_cost.gradient(traj)   # (T, D)
        return full_grad[1:].reshape(-1)                 # skip q_0

    def _total_gradient(self, traj: Trajectory) -> np.ndarray:
        """Combined gradient (n*D,)."""
        xi_free_flat = self._free_vars(traj)
        g_smooth = self._smoothness_gradient(xi_free_flat)
        g_obs = self._obstacle_gradient(traj)
        return g_smooth + self.lambda_obs * g_obs

    # ------------------------------------------------------------------
    # Unconstrained step (for comparison / warm-starting)
    # ------------------------------------------------------------------

    def step_unconstrained(self, traj: Trajectory) -> Trajectory:
        """Standard covariant gradient descent step (no constraint).

        The start q_0 is kept fixed; the end q_n is free to move.
        """
        g = self._total_gradient(traj)
        delta = self.A_free_inv @ g                      # (n*D,)
        xi_free_new = self._free_vars(traj) - self.step_size * delta
        new_xi = np.vstack([traj.xi[:1], xi_free_new.reshape(self.n, self.D)])
        return Trajectory(new_xi, traj.dt)

    # ------------------------------------------------------------------
    # General constrained step — equation (11) of Dragan et al. 2011
    # ------------------------------------------------------------------

    def step_general(
        self, traj: Trajectory, goal_constraint: GoalSetConstraint
    ) -> Trajectory:
        """Constrained update using the general formula (eq. 11).

        Builds the full Jacobian  C = [0, ..., 0, C_tilde] ∈ R^{k × n*D}
        and solves the projected gradient step exactly, without relying on
        the B_nn = β·I_D simplification.

        The update decomposes as:

            xi_new = xi_t
                     - (1/η) A^{-1} g                 # unconstrained step
                     + (1/η) P (C A^{-1} g)           # project gradient
                     - P b                             # pull onto constraint

        where  P = A^{-1} C^T (C A^{-1} C^T)^{-1}   (pseudo-inverse in A-metric).
        """
        eta = 1.0 / self.step_size                      # η = 1/step_size
        g = self._total_gradient(traj)                  # (n*D,)
        q_n = traj.xi[-1]                               # (D,)

        h, C_tilde = goal_constraint.query(q_n)         # (k,), (k, D)
        k = h.shape[0]

        # Build full Jacobian C ∈ R^{k × n*D}: non-zero only in last D columns
        C_full = np.zeros((k, self.n * self.D), dtype=float)
        C_full[:, (self.n - 1) * self.D :] = C_tilde

        # P = A_inv C^T (C A_inv C^T)^{-1}   (n*D, k)
        A_inv_CT = self.A_free_inv @ C_full.T            # (n*D, k)
        CAiCT = C_full @ A_inv_CT                        # (k, k)
        CAiCT_inv = np.linalg.inv(CAiCT)
        P = A_inv_CT @ CAiCT_inv                         # (n*D, k)

        # Unconstrained step
        xi_free = self._free_vars(traj)
        xi_unconstrained = xi_free - (1.0 / eta) * (self.A_free_inv @ g)

        # Correction: project gradient onto constraint null-space + feasibility
        proj_term = (1.0 / eta) * P @ (C_full @ (self.A_free_inv @ g))
        feasib_term = P @ h                              # b = h(q_n)

        xi_free_new = xi_unconstrained + proj_term - feasib_term
        new_xi = np.vstack([traj.xi[:1], xi_free_new.reshape(self.n, self.D)])
        return Trajectory(new_xi, traj.dt)

    # ------------------------------------------------------------------
    # Goal-set simplified step — equation (14) of Dragan et al. 2011
    # ------------------------------------------------------------------

    def step_goalset(
        self, traj: Trajectory, goal_constraint: GoalSetConstraint
    ) -> Trajectory:
        """Constrained update using the goal-set simplified formula (eq. 14).

        Exploits the block structure of A^{-1} for goal-set constraints:
          - B_n   = last D rows of A^{-1}             (D, n*D)
          - B_nn  = last D×D block of A^{-1} ≈ β I_D  (scalar β)

        The update reads:

            xi_new = xi_t
                   - (1/η)           A^{-1} g
                   + (1/(η β))       B_n^T C̃^T (C̃ C̃^T)^{-1} C̃ B_n g
                   - (1/β)           B_n^T C̃^T (C̃ C̃^T)^{-1} h

        Physical interpretation:
          * B_n g    — the endpoint component of the unconstrained gradient
          * C̃ B_n g — that component projected through the constraint Jacobian
          * The first correction projects the gradient update into the
            constraint null-space (Euclidean projection in q-space)
          * The second correction moves the endpoint onto the constraint surface
          * B_n^T propagates the endpoint correction back along the trajectory
            as a linear interpolation (due to the structure of A^{-1})
        """
        eta = 1.0 / self.step_size
        g = self._total_gradient(traj)                   # (n*D,)
        q_n = traj.xi[-1]                                # (D,)

        h, C_tilde = goal_constraint.query(q_n)          # (k,), (k, D)

        # B_n g: endpoint sub-vector of the unconstrained update (D,)
        B_n_g = self._B_n @ g                            # (D,)

        # (C̃ C̃^T)^{-1}   (k, k)
        CCT_inv = np.linalg.inv(C_tilde @ C_tilde.T)

        # Gradient projection correction: projects endpoint gradient onto the
        # constraint null-space and propagates back via B_n^T  (n*D,)
        proj_term = (
            (1.0 / (eta * self._beta))
            * self._B_n.T
            @ C_tilde.T
            @ CCT_inv
            @ (C_tilde @ B_n_g)
        )

        # Feasibility correction: pulls endpoint onto the constraint surface (n*D,)
        feasib_term = (
            (1.0 / self._beta)
            * self._B_n.T
            @ C_tilde.T
            @ CCT_inv
            @ h
        )

        xi_free = self._free_vars(traj)
        xi_free_new = (
            xi_free
            - (1.0 / eta) * (self.A_free_inv @ g)
            + proj_term
            - feasib_term
        )
        new_xi = np.vstack([traj.xi[:1], xi_free_new.reshape(self.n, self.D)])
        return Trajectory(new_xi, traj.dt)

    # ------------------------------------------------------------------
    # Convenience: run full optimisation
    # ------------------------------------------------------------------

    def optimize(
        self,
        traj: Trajectory,
        goal_constraint: GoalSetConstraint,
        n_iters: int,
        method: str = "goalset",
    ) -> Trajectory:
        """Run the optimisation loop.

        Parameters
        ----------
        traj : Trajectory
            Initial trajectory.
        goal_constraint : GoalSetConstraint
            End-point constraint oracle.
        n_iters : int
            Number of gradient steps.
        method : str
            ``"goalset"`` (eq. 14, default) or ``"general"`` (eq. 11).
        """
        step_fn = self.step_goalset if method == "goalset" else self.step_general
        for _ in range(n_iters):
            traj = step_fn(traj, goal_constraint)
        return traj


# ==================== Example GoalSetConstraint implementations ====================

class TargetPointConstraint(GoalSetConstraint):
    """Constraint: the endpoint must reach a specific target point q_goal.

    h(q_n) = q_n - q_goal   (shape (D,))
    C_tilde = I_D            (shape (D, D))

    When h = 0 the endpoint coincides with q_goal exactly.
    """

    def __init__(self, q_goal: np.ndarray):
        self.q_goal = np.asarray(q_goal, dtype=float)

    def query(self, q_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = q_n - self.q_goal                           # (D,)
        C_tilde = np.eye(self.q_goal.shape[0])          # (D, D)
        return h, C_tilde


class TargetHyperplaneConstraint(GoalSetConstraint):
    """Constraint: the endpoint must lie on a hyperplane n^T q = d.

    h(q_n) = [n^T q_n - d]   (shape (1,))
    C_tilde = n^T             (shape (1, D))

    This is the typical "goal set" described in the paper: a (D-1)-dimensional
    manifold of valid endpoints.  The optimiser is free to choose *which* point
    on the hyperplane to reach, trading off smoothness and obstacle avoidance.
    """

    def __init__(self, normal: np.ndarray, offset: float):
        """
        Parameters
        ----------
        normal : np.ndarray, shape (D,)
            Unit (or non-unit) normal of the hyperplane.
        offset : float
            Scalar d such that the constraint is n^T q = d.
        """
        self.normal = np.asarray(normal, dtype=float)
        self.offset = float(offset)

    def query(self, q_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.array([float(self.normal @ q_n) - self.offset])   # (1,)
        C_tilde = self.normal[np.newaxis, :]                      # (1, D)
        return h, C_tilde


class TargetRegionConstraint(GoalSetConstraint):
    """Constraint: the endpoint must lie inside a ball of radius r around q_goal.

    Outside the ball:  h(q_n) = ||q_n - q_goal|| - r  (active)
    Inside the ball:   h(q_n) = 0                      (inactive — no force)

    C_tilde = (q_n - q_goal)^T / ||q_n - q_goal||     (shape (1, D))

    This represents a soft goal-region: the optimiser is free once the endpoint
    enters the ball.
    """

    def __init__(self, q_goal: np.ndarray, radius: float):
        self.q_goal = np.asarray(q_goal, dtype=float)
        self.radius = float(radius)

    def query(self, q_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        diff = q_n - self.q_goal
        dist = np.linalg.norm(diff)
        violation = dist - self.radius
        if violation <= 0.0:
            D = q_n.shape[0]
            return np.zeros(1), np.zeros((1, D))
        h = np.array([violation])
        C_tilde = (diff / dist)[np.newaxis, :]                    # (1, D)
        return h, C_tilde


# ----------------------- Configuration-based utilities -----------------------

def create_shapes_from_config(config: CHOMPConfig) -> List[Shape2D]:
    """Create Shape2D objects from configuration.
    
    Args:
        config: CHOMPConfig instance
        
    Returns:
        List of Shape2D objects
    """
    shapes = []
    for shape_cfg in config.obstacle_shapes:
        shape_type = shape_cfg.get('type', '').lower()
        
        if shape_type == 'circle':
            center = shape_cfg.get('center', [0.0, 0.0])
            radius = shape_cfg.get('radius', 0.1)
            shapes.append(Circle2D(center=tuple(center), radius=radius))
        
        elif shape_type == 'box':
            center = shape_cfg.get('center', [0.0, 0.0])
            width = shape_cfg.get('width', 0.1)
            height = shape_cfg.get('height', 0.1)
            shapes.append(Box2D(center=tuple(center), w=width, h=height))
        
        elif shape_type == 'segment':
            point_a = shape_cfg.get('point_a', [0.0, 0.0])
            point_b = shape_cfg.get('point_b', [1.0, 1.0])
            radius = shape_cfg.get('radius', 0.0)
            shapes.append(Segment2D(a=tuple(point_a), b=tuple(point_b), radius=radius))
        
        else:
            print(f"Warning: Unknown obstacle type '{shape_type}', skipping.")
    
    return shapes


def run_chomp_from_config(config_path: str, verbose: bool = True) -> Tuple[Environment2D, Trajectory, CHOMPOptimizer]:
    """Run CHOMP optimizer using configuration from file.
    
    Args:
        config_path: Path to YAML configuration file
        verbose: Whether to print configuration and progress information
        
    Returns:
        Tuple of (environment, final_trajectory, optimizer)
    """
    # Load configuration
    config = CHOMPConfig.from_file(config_path)
    
    if verbose:
        print(config)
    
    # Create initial trajectory using linear interpolation
    alphas = np.linspace(0.0, 1.0, config.T).reshape(-1, 1)
    waypoints = (1 - alphas) * config.start + alphas * config.goal
    init_traj = Trajectory(waypoints=waypoints.copy(), dt=config.dt)
    
    # Create environment from configuration
    shapes = create_shapes_from_config(config)
    env2d = Environment2D(shapes=shapes, clearance=config.clearance)
    
    # Build smoothness matrix
    A = build_smoothness_A(
        config.T, 
        config.D, 
        damping=config.smoothness_damping
    )
    A_inv = np.linalg.inv(A)
    
    # Create cost functions
    smooth_cost = SmoothnessCost(A)
    obs_cost = ObstacleCost(env2d)
    
    # Create optimizer
    optimizer = CHOMPOptimizer(
        smooth_cost=smooth_cost,
        obstacle_cost=obs_cost,
        step_size=config.step_size,
        A_inv=A_inv,
        lambda_obs=config.lambda_obs,
    )
    
    # Run optimization
    if verbose:
        print(f"\nStarting CHOMP optimization ({config.num_iterations} iterations)...")
    
    cur_traj = init_traj
    for k in range(config.num_iterations):
        cur_traj = optimizer.step(cur_traj)
        if verbose and (k + 1) % max(1, config.num_iterations // 10) == 0:
            print(f"  Iteration {k + 1}/{config.num_iterations}")
    
    if verbose:
        print("Optimization complete.")
    
    return env2d, cur_traj, optimizer


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
    import sys
    
    # Default config path
    config_path = 'chomp_config.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Load configuration and run CHOMP
    config = CHOMPConfig.from_file(config_path)
    print(config)
    
    # Create initial trajectory using linear interpolation
    alphas = np.linspace(0.0, 1.0, config.T).reshape(-1, 1)
    waypoints = (1 - alphas) * config.start + alphas * config.goal
    init_traj = Trajectory(waypoints=waypoints.copy(), dt=config.dt)

    # Create environment from configuration
    shapes = create_shapes_from_config(config)
    env2d = Environment2D(shapes=shapes, clearance=config.clearance)

    # Smoothness matrix and its inverse (covariant metric)
    A = build_smoothness_A(config.T, config.D, damping=config.smoothness_damping)
    A_inv = np.linalg.inv(A)
    
    # Create cost functions
    smooth_cost = SmoothnessCost(A)
    obs_cost = ObstacleCost(env2d)

    optimizer = CHOMPOptimizer(
        smooth_cost=smooth_cost,
        obstacle_cost=obs_cost,
        step_size=config.step_size,
        A_inv=A_inv,
        lambda_obs=config.lambda_obs,
    )

    # Optimize
    trajs_to_plot = [init_traj]
    cur = init_traj
    for k in range(config.num_iterations):
        cur = optimizer.step(cur)
        trajs_to_plot.append(cur)

    # Plot trajectory evolution
    plot_environment_and_trajectory(
        env2d, 
        trajs_to_plot, 
        xlim=config.plot_xlim, 
        ylim=config.plot_ylim,
        fname=config.trajectory_plot_name if config.save_trajectory_plot else None
    )
    
    # Plot gradients on final trajectory
    grad_smooth = optimizer.smooth_cost.gradient(trajs_to_plot[-1])
    grad_obs = optimizer.obstacle_cost.gradient(trajs_to_plot[-1])
    final_traj = trajs_to_plot[-1]
    
    grad_obs = grad_obs * config.gradient_scale_factor
    grad_smooth = grad_smooth * config.gradient_scale_factor
    
    plt.figure(figsize=(8, 8))
    plt.plot(final_traj.xi[:, 0], final_traj.xi[:, 1], 'k-', label='Final Trajectory')
    plt.quiver(final_traj.xi[:, 0], final_traj.xi[:, 1],
               -grad_smooth[:, 0], -grad_smooth[:, 1],
               color='blue', scale=50, width=0.005, label='Smoothness Gradient')
    plt.quiver(final_traj.xi[:, 0], final_traj.xi[:, 1],
               -grad_obs[:, 0], -grad_obs[:, 1],
               color='red', scale=50, width=0.005, label='Obstacle Gradient')
    plt.xlim(*config.plot_xlim)
    plt.ylim(*config.plot_ylim)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    plt.title('Final Trajectory with Gradients')
    plt.legend()
    
    if config.save_gradient_plot:
        plt.savefig(config.gradient_plot_name, dpi=200)
    plt.show()

    # ==================================================================
    # Goal-Set CHOMP demo
    # ==================================================================
    print("\n--- Goal-Set CHOMP demo ---")

    # Build free-end smoothness matrix (n = T-1 free variables)
    n_free = config.T - 1
    A_free = build_smoothness_A_free_end(n_free, config.D,
                                         damping=config.smoothness_damping)

    # Re-use the same environment and a fresh linear trajectory
    alphas_gs = np.linspace(0.0, 1.0, config.T).reshape(-1, 1)
    init_xi_gs = (1 - alphas_gs) * config.start + alphas_gs * config.goal
    traj_gs = Trajectory(waypoints=init_xi_gs.copy(), dt=config.dt)

    # Define a goal-set constraint: the endpoint must lie on the horizontal
    # line  y = config.goal[1]  (a hyperplane constraint).
    # This means the optimiser is free to choose the x coordinate of the
    # endpoint while satisfying the y-coordinate constraint.
    goal_normal = np.zeros(config.D)
    goal_normal[1] = 1.0                                  # y-axis direction
    goal_constraint = TargetHyperplaneConstraint(
        normal=goal_normal,
        offset=float(config.goal[1]),
    )

    goal_set_optimizer = CHOMPGoalSetOptimizer(
        obstacle_cost=obs_cost,
        step_size=config.step_size,
        A_free=A_free,
        q_start=config.start,
        lambda_obs=config.lambda_obs,
    )

    print(f"Initial endpoint: {traj_gs.xi[-1]}")
    print(f"Goal hyperplane: y = {config.goal[1]:.4f}")
    print(f"β (A^{{-1}} endpoint diagonal): {goal_set_optimizer._beta:.4f}")

    # --- Run eq. 14 (goal-set simplified) ---
    trajs_gs14 = [traj_gs]
    cur_gs = traj_gs
    for k in range(config.num_iterations):
        cur_gs = goal_set_optimizer.step_goalset(cur_gs, goal_constraint)
        trajs_gs14.append(cur_gs)

    h_final14, _ = goal_constraint.query(cur_gs.xi[-1])
    print(f"[eq.14] Final endpoint: {cur_gs.xi[-1]}  |  constraint residual h = {h_final14}")

    # --- Run eq. 11 (general) ---
    cur_gs11 = Trajectory(waypoints=init_xi_gs.copy(), dt=config.dt)
    trajs_gs11 = [cur_gs11]
    for k in range(config.num_iterations):
        cur_gs11 = goal_set_optimizer.step_general(cur_gs11, goal_constraint)
        trajs_gs11.append(cur_gs11)

    h_final11, _ = goal_constraint.query(cur_gs11.xi[-1])
    print(f"[eq.11] Final endpoint: {cur_gs11.xi[-1]}  |  constraint residual h = {h_final11}")

    # --- Visualise both alongside the unconstrained result ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, trajs_vis, title in zip(
        axes,
        [trajs_to_plot, trajs_gs14, trajs_gs11],
        ["Unconstrained CHOMP", "Goal-Set CHOMP (eq. 14)", "Goal-Set CHOMP (eq. 11)"],
    ):
        cmap = plt.get_cmap("viridis")
        n_t = len(trajs_vis)
        for i, t in enumerate(trajs_vis):
            ax.plot(t.xi[:, 0], t.xi[:, 1], "-",
                    color=cmap(i / max(n_t - 1, 1)), alpha=0.5, linewidth=0.8)
        # Highlight final trajectory
        ax.plot(trajs_vis[-1].xi[:, 0], trajs_vis[-1].xi[:, 1],
                "k-", linewidth=2, label="Final")
        ax.plot(trajs_vis[0].xi[0, 0], trajs_vis[0].xi[0, 1],
                "go", markersize=10, label="Start")
        ax.plot(trajs_vis[-1].xi[-1, 0], trajs_vis[-1].xi[-1, 1],
                "b*", markersize=12, label="End")
        # Draw goal hyperplane (y = goal[1]) for goal-set plots
        if "Goal-Set" in title:
            ax.axhline(config.goal[1], color="purple", linestyle="--",
                       linewidth=1.5, label=f"y = {config.goal[1]:.2f}")
        # Draw obstacles
        for s in env2d.shapes:
            if isinstance(s, Circle2D):
                ax.add_patch(plt.Circle(s.c, s.r, color="red",
                                        fill=False, linewidth=2))
            elif isinstance(s, Box2D):
                ax.add_patch(plt.Rectangle(
                    (s.c[0] - s.hw, s.c[1] - s.hh), 2 * s.hw, 2 * s.hh,
                    color="red", fill=False, linewidth=2))
        ax.set_xlim(*config.plot_xlim)
        ax.set_ylim(*config.plot_ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("chomp_goal_set_comparison.png", dpi=200)
    plt.show()
    print("Saved comparison to chomp_goal_set_comparison.png")