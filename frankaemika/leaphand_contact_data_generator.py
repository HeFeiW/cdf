"""
LeapHand Contact Data Generator
================================

为多指手(LeapHand)生成接触数据集，支持两种方法：

Method 1 (IK-based):  Task-space → Config-space
    离散化task space为 N_t^3 个网格点，对每个点 p 通过优化(L-BFGS)求解
    所有满足 sdf(p, q) ≈ 0 的机器人构型 q（即机器人表面经过 p 的构型）。

Method 2 (FK-based):  Config-space → Task-space
    离散化configuration space为 N_c^dof 个网格，对每个构型 q 通过 FK
    得到机器人表面顶点，遍历task space网格点，记录距离 < 阈值的网格点。
    无需训练好的SDF模型，直接使用网格顶点距离。

LeapHand运动链结构：
    4根手指(finger_0..3)，每根手指通过串联运动链表示：
    palm_lower_left(固定底座) → mcp_joint → pip → dip → fingertip
    每根手指有 4 个旋转关节(DOF=4)，底座(palm)为所有手指共用。

数据保存格式：参见 docs/leaphand_contact_data.md
"""

import os
import sys
import json
import math
import time
import copy
import argparse
from datetime import datetime

import torch
import numpy as np

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_DIR, "../../RDF"))

from panda_layers.parallel_robot_layer import ParallelRobotLayer

# ─── optional imports for Method 1 ────────────────────────────────────────────
try:
    from parallel_bf_sdf import ParallelBPSDF
    _HAS_BPSDF = True
except ImportError:
    _HAS_BPSDF = False

try:
    from torchmin import minimize
    _HAS_TORCHMIN = True
except ImportError:
    _HAS_TORCHMIN = False

try:
    import rdf_utils as utils
    _HAS_RDF_UTILS = True
except ImportError:
    _HAS_RDF_UTILS = False

PI = math.pi


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _make_save_dir(base_dir: str, method: str, robot: str, finger_idx: int) -> str:
    """创建并返回保存目录路径。"""
    path = os.path.join(base_dir, robot, method, f"finger_{finger_idx}")
    os.makedirs(path, exist_ok=True)
    return path


def _save_config(cfg: dict, path: str):
    """将配置信息保存为 JSON 文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)


def _build_task_grid(workspace: np.ndarray, n: int, device: torch.device):
    """
    在 workspace 内创建 n^3 个均匀分布的task space网格点。

    Args:
        workspace: (2, 3) array, [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        n:         每轴离散数
        device:    torch device

    Returns:
        task_pts:  (n^3, 3) float32 tensor on device
    """
    x = torch.linspace(float(workspace[0, 0]), float(workspace[1, 0]), n)
    y = torch.linspace(float(workspace[0, 1]), float(workspace[1, 1]), n)
    z = torch.linspace(float(workspace[0, 2]), float(workspace[1, 2]), n)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    return torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1).to(device)


def _build_config_grid(q_min: torch.Tensor, q_max: torch.Tensor, n_per_dof: int):
    """
    在关节限位内创建 n_per_dof^dof 个均匀分布的configuration space网格点（CPU tensor）。

    Args:
        q_min:      (dof,) 关节下限
        q_max:      (dof,) 关节上限
        n_per_dof:  每个自由度的离散数

    Returns:
        config_grid: (n_per_dof^dof, dof) float32 tensor on CPU
    """
    dof = q_min.shape[0]
    grids_1d = [
        torch.linspace(q_min[d].item(), q_max[d].item(), n_per_dof)
        for d in range(dof)
    ]
    mesh = torch.meshgrid(*grids_1d, indexing="ij")
    return torch.stack([m.flatten() for m in mesh], dim=-1).float()


def _q_to_pose_matrix(q_base: torch.Tensor) -> torch.Tensor:
    """
    将 (B, 6) 的底座配置 [tx, ty, tz, rx, ry, rz] 转换为 (B, 4, 4) SE(3) 矩阵（CPU）。
    与 rdf_utils.q_to_poseMatrix 等价但不依赖 `self`。
    rx/ry/rz 为 ZYX 欧拉角（单位：弧度）。
    """
    import rdf_utils
    B = q_base.shape[0]
    pose = torch.zeros(B, 4, 4, dtype=torch.float32)
    pose[:, :3, 3] = q_base[:, :3].float().cpu()
    pose[:, :3, :3] = torch.from_numpy(
        rdf_utils.batch_euler_to_matrix(
            q_base[:, 3:].detach().cpu().float().numpy()
        )
    ).float()
    pose[:, 3, 3] = 1.0
    return pose


def _build_config_grid_with_base(
    q_min_finger: torch.Tensor,
    q_max_finger: torch.Tensor,
    n_per_dof: int,
    base_min: torch.Tensor,
    base_max: torch.Tensor,
    n_base_random: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    构建带6-DOF底座变换的配置空间采样（CPU tensor）。

    策略：手指关节 × 底座随机采样
      - 手指关节：均匀网格，共 n_per_dof^dof 个配置
      - 底座6-DOF：每个手指配置随机采样 n_base_random 个底座位姿
      - 总配置数：n_per_dof^dof × n_base_random

    格式：[q_joint (dof), tx, ty, tz, rx, ry, rz]，与 Method 1 with_base 格式一致。

    Args:
        q_min_finger:  (dof,) 手指关节下限（soft limit）
        q_max_finger:  (dof,) 手指关节上限（soft limit）
        n_per_dof:     每个手指DOF的离散数
        base_min:      (6,) 底座DOF下限 [workspace_min..., -π,-π,-π]
        base_max:      (6,) 底座DOF上限 [workspace_max..., π, π, π]
        n_base_random: 每个手指配置随机采样的底座位姿数
        seed:          随机种子，保证可重复性

    Returns:
        configs: (n_per_dof^dof × n_base_random, dof+6) float32 tensor on CPU
    """
    torch.manual_seed(seed)
    finger_grid = _build_config_grid(q_min_finger, q_max_finger, n_per_dof)
    N_finger = len(finger_grid)

    base_min_cpu = base_min.cpu().float()
    base_max_cpu = base_max.cpu().float()
    q_base = torch.rand(N_finger * n_base_random, 6)
    q_base = q_base * (base_max_cpu - base_min_cpu) + base_min_cpu

    q_finger = finger_grid.unsqueeze(1).expand(-1, n_base_random, -1)
    q_finger = q_finger.reshape(-1, finger_grid.shape[1])

    return torch.cat([q_finger, q_base], dim=-1)


def _finalize_data(data: dict, dof: int) -> dict:
    """将 data[idx]['q'], data[idx]['idx'] 从 list 转为 numpy array。"""
    for idx in data:
        if len(data[idx]["q"]) > 0:
            data[idx]["q"] = np.array(data[idx]["q"], dtype=np.float32)
            data[idx]["idx"] = np.array(data[idx]["idx"], dtype=np.int32)
        else:
            data[idx]["q"] = np.empty((0, dof), dtype=np.float32)
            data[idx]["idx"] = np.empty((0,), dtype=np.int32)
    return data


def _compute_stats(data: dict) -> dict:
    """计算数据集的统计信息（覆盖率、每点平均构型数等）。"""
    n_total = len(data)
    q_counts = [len(data[i]["q"]) for i in data]
    nonempty = sum(1 for c in q_counts if c > 0)
    return {
        "total_points": n_total,
        "nonempty_points": nonempty,
        "coverage_ratio": nonempty / n_total if n_total > 0 else 0.0,
        "mean_configs_per_point": float(np.mean(q_counts)),
        "max_configs_per_point": int(np.max(q_counts)),
        "total_contact_pairs": int(np.sum(q_counts)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Method 1：IK-based (Task-space → Config-space)
# ══════════════════════════════════════════════════════════════════════════════

class Method1IKDataGenerator:
    """
    Method 1: 离散化task space + IK求解接触集

    对 N_t^3 个task space格点，每点通过 L-BFGS 优化求解满足
    sdf(p, q) ≈ 0 的所有构型 q（机器人表面恰好经过 p 的构型集合）。

    需要：预训练的 BP-SDF 模型（parallel_bf_sdf.ParallelBPSDF）。

    用法示例::

        robot = ParallelRobotLayer(device, paths, robot='leaphand')
        gen = Method1IKDataGenerator(device, robot, paths, serial_idx=0)
        gen.generate(save_dir='./data/leaphand')
    """

    def __init__(
        self,
        device,
        robot: ParallelRobotLayer,
        paths: dict,
        serial_idx: int = 0,
        with_base: bool = False,
        n_task_discrete: int = 20,
        batchsize: int = 20000,
        epsilon: float = 1e-3,
    ):
        """
        Args:
            device:          torch device
            robot:           ParallelRobotLayer 实例
            paths:           路径字典，需含 'model' 键 (BP-SDF 模型路径)
            serial_idx:      手指索引 (0=食指, 1=中指, 2=无名指, 3=大拇指)
            with_base:       是否同时优化底座 6-DOF 位姿
            n_task_discrete: task space 每轴离散数 (默认20 → 20^3=8000点)
            batchsize:       IK 优化时每次随机初始化的构型数
            epsilon:         判定接触的 SDF 阈值
        """
        if not _HAS_BPSDF:
            raise ImportError("需要 parallel_bf_sdf.ParallelBPSDF，请确认 RDF 路径正确。")
        if not _HAS_TORCHMIN:
            raise ImportError("需要 torchmin 库，请执行 pip install torchmin。")

        self.device = device
        self.robot = robot
        self.paths = paths
        self.serial_idx = serial_idx
        self.with_base = with_base
        self.n_task_discrete = n_task_discrete
        self.batchsize = batchsize
        self.epsilon = epsilon

        # BP-SDF 模型
        model_path = paths["model"]
        self.bp_sdf = ParallelBPSDF(8, -1.0, 1.0, robot, model_path, device)
        self.model = torch.load(model_path, map_location=device)

        # 工作空间与关节信息
        self.workspace = robot.space_limits.cpu().numpy()
        serial = robot.serials[serial_idx]
        self.used_links = serial.all_links.copy()

        print(f"[Method1] serial_idx={serial_idx}, dof={serial.dof}, "
              f"with_base={with_base}")
        print(f"[Method1] used_links: {self.used_links}")

    # ── SDF 查询 ────────────────────────────────────────────────────────────

    def _sdf(self, x, q, pose=None, return_index=False):
        """sdf(x, q) → 距离 (Nq,)，可选返回最近link的索引 (Nq,)。"""
        if pose is None:
            Nq = q.shape[0]
            pose = torch.eye(4, device=self.device).unsqueeze(0).expand(Nq, 4, 4)
        if not return_index:
            d, _ = self.bp_sdf.get_serial_sdf_batch(
                x, pose, q, self.model,
                use_derivative=False,
                serial_idx=self.serial_idx,
                used_links=self.used_links,
            )
            return d.min(dim=1)[0]
        else:
            d, _, idx = self.bp_sdf.get_serial_sdf_batch(
                x, pose, q, self.model,
                use_derivative=False,
                serial_idx=self.serial_idx,
                used_links=self.used_links,
                return_index=True,
            )
            d, pts_idx = d.min(dim=1)
            idx = idx[torch.arange(len(idx)), pts_idx]
            return d, idx

    def _sdf_with_pose(self, x, q, return_index=False):
        """带底座6-DOF的SDF查询，q = [joint_angles..., tx, ty, tz, rx, ry, rz]。"""
        if not _HAS_RDF_UTILS:
            raise ImportError("需要 rdf_utils，请确认 RDF 路径正确。")
        q_pose = q[:, -6:]
        q_joint = q[:, :-6]
        pose = utils.q_to_poseMatrix(self, q_pose).to(self.device)
        if not return_index:
            d, _ = self.bp_sdf.get_serial_sdf_batch(
                x, pose, q_joint, self.model,
                use_derivative=False,
                serial_idx=self.serial_idx,
                used_links=self.used_links,
            )
            return d.min(dim=1)[0]
        else:
            d, _, idx = self.bp_sdf.get_serial_sdf_batch(
                x, pose, q_joint, self.model,
                use_derivative=False,
                serial_idx=self.serial_idx,
                used_links=self.used_links,
                return_index=True,
            )
            d, pts_idx = d.min(dim=1)
            idx = idx[torch.arange(len(idx)), pts_idx]
            return d, idx

    # ── 单点 IK 求解 ────────────────────────────────────────────────────────

    def _find_q_for_point(self, p: torch.Tensor):
        """
        对单个task space点 p (1,3)，搜索满足 sdf(p,q)≈0 的所有构型。

        Returns:
            final_q: (M, dof) 满足条件的构型
            idx:     (M,) 最近link索引
        """
        serial = self.robot.serials[self.serial_idx]
        q_min = serial.theta_min_soft
        q_max = serial.theta_max_soft

        if self.with_base:
            base_min = serial.theta_min_base
            base_max = serial.theta_max_base
            full_min = torch.cat([q_min, base_min])
            full_max = torch.cat([q_max, base_max])
            total_dof = serial.dof + 6

            def cost(q):
                return torch.sum(self._sdf_with_pose(p, q) ** 2)

            q0 = torch.rand(self.batchsize, total_dof, device=self.device)
            q0 = q0 * (full_max - full_min) + full_min
            res = minimize(cost, q0, method="l-bfgs",
                           options=dict(line_search="strong-wolfe"),
                           max_iter=50, disp=0)
            d, idx = self._sdf_with_pose(p, res.x, return_index=True)
            bnd_mask = ((res.x > full_min) & (res.x < full_max)).all(dim=1)
        else:
            def cost(q):
                return torch.sum(self._sdf(p, q) ** 2)

            q0 = torch.rand(self.batchsize, serial.dof, device=self.device)
            q0 = q0 * (q_max - q_min) + q_min
            res = minimize(cost, q0, method="l-bfgs",
                           options=dict(line_search="strong-wolfe"),
                           max_iter=50, disp=0)
            d, idx = self._sdf(p, res.x, return_index=True)
            bnd_mask = ((res.x > q_min) & (res.x < q_max)).all(dim=1)

        d, idx = d.squeeze(), idx.squeeze()
        dist_mask = torch.abs(d) < self.epsilon
        final_mask = dist_mask & bnd_mask
        return res.x[final_mask], idx[final_mask]

    # ── 主生成函数 ──────────────────────────────────────────────────────────

    def generate(self, save_dir: str = None):
        """
        生成并保存 Method 1 数据集。

        Args:
            save_dir: 保存根目录（默认 CUR_DIR/data）

        保存文件：
            <save_dir>/leaphand/method1_ik/finger_<i>/data.npy
            <save_dir>/leaphand/method1_ik/finger_<i>/config.json
        """
        if save_dir is None:
            save_dir = os.path.join(CUR_DIR, "data")

        out_dir = _make_save_dir(save_dir, "method1_ik",
                                 self.robot.robot, self.serial_idx)
        serial = self.robot.serials[self.serial_idx]
        dof = serial.dof

        print(f"\n{'='*60}")
        print(f"[Method1] 开始生成 finger_{self.serial_idx} 数据")
        print(f"  workspace  : {self.workspace.tolist()}")
        print(f"  n_discrete : {self.n_task_discrete} → "
              f"{self.n_task_discrete**3} 个task space点")
        print(f"  batchsize  : {self.batchsize}")
        print(f"  epsilon    : {self.epsilon}")
        print(f"{'='*60}")

        t0 = time.time()
        task_pts = _build_task_grid(
            self.workspace, self.n_task_discrete, self.device
        )
        T = len(task_pts)
        data = {i: {"x": task_pts[i].cpu().numpy(), "q": [], "idx": []}
                for i in range(T)}

        for i, p in enumerate(task_pts):
            q_valid, link_idx = self._find_q_for_point(p.unsqueeze(0))
            data[i]["q"] = q_valid.detach().cpu().numpy()
            data[i]["idx"] = link_idx.detach().cpu().numpy()

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = elapsed / (i + 1)
                remaining = rate * (T - i - 1)
                print(f"  [{i+1:5d}/{T}] "
                      f"已耗时 {elapsed:6.1f}s  "
                      f"平均 {rate:.2f}s/pt  "
                      f"预计剩余 {remaining/60:.1f}min  "
                      f"last_n_q={len(data[i]['q'])}")

        total_time = time.time() - t0
        data = _finalize_data(data, dof)
        stats = _compute_stats(data)

        # 保存数据
        data_path = os.path.join(out_dir, "data.npy")
        np.save(data_path, data)

        # 保存配置文件
        cfg = {
            "method": "method1_ik",
            "robot": self.robot.robot,
            "serial_idx": self.serial_idx,
            "finger_links": self.used_links,
            "dof": dof,
            "with_base": self.with_base,
            "workspace": {"min": self.workspace[0].tolist(),
                          "max": self.workspace[1].tolist()},
            "joint_limits_soft": {
                "min": serial.theta_min_soft.cpu().tolist(),
                "max": serial.theta_max_soft.cpu().tolist(),
            },
            "n_task_discrete": self.n_task_discrete,
            "total_task_points": T,
            "batchsize": self.batchsize,
            "epsilon": self.epsilon,
            "model_path": str(self.paths.get("model", "")),
            "device": str(self.device),
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": round(total_time, 2),
            "stats": stats,
        }
        cfg_path = os.path.join(out_dir, "config.json")
        _save_config(cfg, cfg_path)

        print(f"\n[Method1] 完成！耗时 {total_time/60:.1f} min")
        print(f"  覆盖率      : {stats['coverage_ratio']*100:.1f}%")
        print(f"  平均构型数/点: {stats['mean_configs_per_point']:.1f}")
        print(f"  总接触对数  : {stats['total_contact_pairs']}")
        print(f"  数据文件    : {data_path}")
        print(f"  配置文件    : {cfg_path}")

        return data, cfg


# ══════════════════════════════════════════════════════════════════════════════
# Method 2：FK-based (Config-space → Task-space)
# ══════════════════════════════════════════════════════════════════════════════

class Method2FKDataGenerator:
    """
    Method 2: 离散化configuration space + FK得到task space接触集

    对 N_c^dof 个configuration space格点，每个构型 q 通过FK得到机器人
    表面网格顶点位置，然后对 N_t^3 个task space格点，若某点 p 到机器人
    表面的最近顶点距离 < threshold，则将 (p, q) 记为一个接触对。

    优点：无需训练好的SDF模型，直接使用URDF网格顶点。
    缺点：精度取决于网格顶点密度；DOF较多时config space指数爆炸。

    用法示例::

        robot = ParallelRobotLayer(device, paths, robot='leaphand')
        gen = Method2FKDataGenerator(device, robot, paths, serial_idx=0,
                                     n_config_discrete=12, n_task_discrete=20)
        gen.generate(save_dir='./data/leaphand')
    """

    def __init__(
        self,
        device,
        robot: ParallelRobotLayer,
        paths: dict,
        serial_idx: int = 0,
        with_base: bool = False,
        n_task_discrete: int = 20,
        n_config_discrete: int = 10,
        n_base_random: int = 50,
        threshold: float = 5e-3,
        config_batch_size: int = 200,
        task_batch_size: int = 100,
        dist_sub_batch_size: int = 5,
    ):
        """
        Args:
            device:             torch device
            robot:              ParallelRobotLayer 实例
            paths:              路径字典
            serial_idx:         手指索引 (0-3)
            with_base:          是否包含6-DOF底座变换（平移+旋转）。
                                True → 配置向量为 [q_joint (dof), tx, ty, tz, rx, ry, rz]，
                                       总维度 = dof + 6 = 10（leaphand 单指）。
                                       采样策略：手指关节均匀网格 × 底座随机采样。
            n_task_discrete:    task space 每轴离散数 (默认20 → 20^3=8000点)
            n_config_discrete:  config space 每个手指DOF的离散数 (默认10)
            n_base_random:      with_base=True 时每个手指配置随机采样的底座位姿数
                                总配置数 = n_config_discrete^dof × n_base_random
            threshold:          判定接触的距离阈值（米，默认5mm）
            config_batch_size:  FK批处理大小（构型数，影响显存用量较小）
            task_batch_size:    距离计算时task space点的分批大小
            dist_sub_batch_size: 距离计算时config的子批大小。
                torch.cdist 内部矩阵为 (Bsub, T_b, Nv)，需控制峰值显存：
                Bsub * T_b * Nv * 4B < 显存上限。
                对palm (~43k顶点)，默认 Bsub=5, T_b=100 → ~86MB/link。
        """
        self.device = device
        self.robot = robot
        self.paths = paths
        self.serial_idx = serial_idx
        self.with_base = with_base
        self.n_task_discrete = n_task_discrete
        self.n_config_discrete = n_config_discrete
        self.n_base_random = n_base_random
        self.threshold = threshold
        self.config_batch_size = config_batch_size
        self.task_batch_size = task_batch_size
        self.dist_sub_batch_size = dist_sub_batch_size

        self.workspace = robot.space_limits.cpu().numpy()
        serial = robot.serials[serial_idx]
        self.dof = serial.dof

        # 获取此串联链中有网格的link列表（按all_links顺序）
        self.links_with_mesh = [
            lk for lk in serial.all_links
            if lk in serial.Link2Mesh and serial.Link2Mesh[lk] is not None
        ]
        self.link_to_idx = {lk: i for i, lk in enumerate(serial.all_links)}

        total_configs = (
            n_config_discrete ** self.dof * n_base_random
            if with_base else n_config_discrete ** self.dof
        )
        print(f"[Method2] serial_idx={serial_idx}, dof={self.dof}, "
              f"with_base={with_base}")
        print(f"[Method2] config space总点数: "
              f"{n_config_discrete}^{self.dof}"
              + (f" × {n_base_random}(base)" if with_base else "")
              + f" = {total_configs:,}")
        print(f"[Method2] task space总点数 : {n_task_discrete}^3 "
              f"= {n_task_discrete**3:,}")
        print(f"[Method2] 接触阈值: {threshold*1000:.1f} mm")
        print(f"[Method2] 显存控制: config_batch={config_batch_size}, "
              f"dist_sub_batch={dist_sub_batch_size}, task_batch={task_batch_size}")
        peak_mb = dist_sub_batch_size * task_batch_size * 43315 * 4 / 1e6
        print(f"[Method2] 预估峰值显存(per-link per-step): ~{peak_mb:.0f}MB "
              f"(实际取决于最大link顶点数)")
        print(f"[Method2] links_with_mesh: {self.links_with_mesh}")

    # ── FK 距离计算核心 ─────────────────────────────────────────────────────

    def _compute_min_dist(
        self,
        serial,
        batch_q: torch.Tensor,
        task_pts: torch.Tensor,
    ):
        """
        对一批构型 batch_q 和所有 task space 点，计算每对 (q, p) 的
        机器人表面最近顶点距离，以及对应的最近 link 索引。

        显存控制策略（三层分批）：
          外层: config 大批 (config_batch_size)     → FK 高效并行
          中层: config 子批 (dist_sub_batch_size)   → 距离计算显存控制
          内层: task point 子批 (task_batch_size)   → 进一步控制峰值
        torch.cdist 内部矩阵为 (Bsub, T_b, Nv)，峰值显存 ≈
            dist_sub_batch_size × task_batch_size × max(Nv_link) × 4 bytes

        Args:
            serial:    SerialRobotLayer 实例
            batch_q:   (B, dof) 构型批次（GPU）
            task_pts:  (T, 3) task space 点（GPU）

        Returns:
            min_dist:      (B, T) 最近距离（float32）
            min_link_idx:  (B, T) 最近link的索引（在all_links中的位置）
        """
        B = batch_q.shape[0]
        T = task_pts.shape[0]

        if self.with_base:
            # batch_q: (B, dof+6) → split into joint angles and base pose
            q_joint = batch_q[:, :self.dof]   # (B, dof)
            q_base  = batch_q[:, self.dof:]   # (B, 6): [tx,ty,tz,rx,ry,rz]
            pose = _q_to_pose_matrix(q_base).to(self.device)  # (B, 4, 4)
        else:
            q_joint = batch_q
            pose = torch.eye(4, device=self.device, dtype=torch.float32
                             ).unsqueeze(0).expand(B, -1, -1)

        with torch.no_grad():
            verts_dict, _ = serial.forward(pose, q_joint)

        min_dist = torch.full((B, T), float("inf"), device=self.device)
        min_link_idx = torch.zeros((B, T), dtype=torch.long, device=self.device)

        Bs = self.dist_sub_batch_size  # config 子批大小

        for b0 in range(0, B, Bs):
            b1 = min(b0 + Bs, B)
            for link in self.links_with_mesh:
                if link not in verts_dict:
                    continue
                link_verts = verts_dict[link][b0:b1]  # (Bs, Nv, 3)
                link_idx_val = self.link_to_idx[link]
                Bsub = b1 - b0

                for t0 in range(0, T, self.task_batch_size):
                    t1 = min(t0 + self.task_batch_size, T)
                    pts_sub = task_pts[t0:t1]  # (T_b, 3)
                    T_b = t1 - t0

                    # dists: (Bsub, T_b, Nv)
                    # 峰值显存 = Bsub * T_b * Nv * 4 bytes
                    pts_exp = pts_sub.unsqueeze(0).expand(Bsub, T_b, 3)
                    dists = torch.cdist(pts_exp.contiguous(),
                                        link_verts.contiguous())
                    link_min = dists.min(dim=-1)[0]  # (Bsub, T_b)

                    cur = min_dist[b0:b1, t0:t1]
                    better = link_min < cur
                    min_link_idx[b0:b1, t0:t1][better] = link_idx_val
                    min_dist[b0:b1, t0:t1] = torch.min(cur, link_min)

        return min_dist, min_link_idx

    # ── 主生成函数 ──────────────────────────────────────────────────────────

    def generate(self, save_dir: str = None):
        """
        生成并保存 Method 2 数据集。

        Args:
            save_dir: 保存根目录（默认 CUR_DIR/data）

        保存文件：
            <save_dir>/leaphand/method2_fk/finger_<i>/data.npy
            <save_dir>/leaphand/method2_fk/finger_<i>/config.json
        """
        if save_dir is None:
            save_dir = os.path.join(CUR_DIR, "data")

        method_tag = "method2_fk_base" if self.with_base else "method2_fk"
        out_dir = _make_save_dir(save_dir, method_tag,
                                 self.robot.robot, self.serial_idx)
        serial = self.robot.serials[self.serial_idx]
        dof_total = self.dof + 6 if self.with_base else self.dof

        configs_desc = (
            f"{self.n_config_discrete}^{self.dof}×{self.n_base_random}(base)"
            if self.with_base
            else f"{self.n_config_discrete}^{self.dof}"
        )

        print(f"\n{'='*60}")
        print(f"[Method2] 开始生成 finger_{self.serial_idx} 数据"
              f"  [with_base={self.with_base}]")
        print(f"  workspace      : {self.workspace.tolist()}")
        print(f"  n_task_discrete: {self.n_task_discrete} → "
              f"{self.n_task_discrete**3} 个task space点")
        print(f"  配置数         : {configs_desc}")
        print(f"  threshold      : {self.threshold*1000:.1f} mm")
        print(f"{'='*60}")

        t0_wall = time.time()

        # ── 构建网格 ────────────────────────────────────────────────────
        task_pts = _build_task_grid(
            self.workspace, self.n_task_discrete, self.device
        )
        T = len(task_pts)

        if self.with_base:
            config_grid = _build_config_grid_with_base(
                serial.theta_min_soft.cpu(),
                serial.theta_max_soft.cpu(),
                self.n_config_discrete,
                serial.theta_min_base.cpu(),
                serial.theta_max_base.cpu(),
                self.n_base_random,
            )
        else:
            config_grid = _build_config_grid(
                serial.theta_min_soft.cpu(),
                serial.theta_max_soft.cpu(),
                self.n_config_discrete,
            )
        N_total = len(config_grid)

        # ── 初始化数据结构 ───────────────────────────────────────────────
        data = {
            i: {"x": task_pts[i].cpu().numpy(), "q": [], "idx": []}
            for i in range(T)
        }

        # ── 分批处理配置 ─────────────────────────────────────────────────
        n_batches = math.ceil(N_total / self.config_batch_size)
        for batch_no, batch_start in enumerate(
                range(0, N_total, self.config_batch_size)):
            batch_end = min(batch_start + self.config_batch_size, N_total)
            batch_q = config_grid[batch_start:batch_end].to(self.device)
            B = batch_q.shape[0]

            # FK + 距离计算
            min_dist, min_link_idx = self._compute_min_dist(
                serial, batch_q, task_pts
            )

            # 筛选接触对
            contact_mask = min_dist < self.threshold  # (B, T)
            b_arr, t_arr = torch.where(contact_mask)

            if len(b_arr) > 0:
                q_vals = batch_q[b_arr].cpu().numpy()      # (M, dof)
                l_vals = min_link_idx[b_arr, t_arr].cpu().numpy()  # (M,)
                t_vals = t_arr.cpu().numpy()                # (M,)
                for q_v, l_v, t_idx in zip(q_vals, l_vals, t_vals):
                    data[int(t_idx)]["q"].append(q_v)
                    data[int(t_idx)]["idx"].append(int(l_v))

            # 进度日志
            if (batch_no + 1) % max(1, n_batches // 20) == 0 or \
                    batch_no == n_batches - 1:
                elapsed = time.time() - t0_wall
                pct = (batch_no + 1) / n_batches
                remaining = elapsed / pct * (1 - pct)
                n_pairs = sum(len(d["q"]) for d in data.values())
                print(f"  batch [{batch_no+1:4d}/{n_batches}] "
                      f"{pct*100:5.1f}%  "
                      f"已耗时 {elapsed:6.1f}s  "
                      f"预计剩余 {remaining/60:.1f}min  "
                      f"接触对 {n_pairs:,}")

        total_time = time.time() - t0_wall
        data = _finalize_data(data, dof_total)
        stats = _compute_stats(data)

        # 保存数据
        data_path = os.path.join(out_dir, "data.npy")
        np.save(data_path, data)

        # 保存配置文件
        cfg = {
            "method": method_tag,
            "robot": self.robot.robot,
            "serial_idx": self.serial_idx,
            "finger_links": serial.all_links,
            "links_with_mesh": self.links_with_mesh,
            "dof": self.dof,
            "with_base": self.with_base,
            "dof_total": dof_total,
            "workspace": {"min": self.workspace[0].tolist(),
                          "max": self.workspace[1].tolist()},
            "joint_limits_soft": {
                "min": serial.theta_min_soft.cpu().tolist(),
                "max": serial.theta_max_soft.cpu().tolist(),
            },
            **({"base_limits": {
                "min": serial.theta_min_base.cpu().tolist(),
                "max": serial.theta_max_base.cpu().tolist(),
            }} if self.with_base else {}),
            "n_task_discrete": self.n_task_discrete,
            "n_config_discrete": self.n_config_discrete,
            **({"n_base_random": self.n_base_random} if self.with_base else {}),
            "total_task_points": T,
            "total_config_points": N_total,
            "contact_threshold_m": self.threshold,
            "config_batch_size": self.config_batch_size,
            "dist_sub_batch_size": self.dist_sub_batch_size,
            "task_batch_size": self.task_batch_size,
            "device": str(self.device),
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": round(total_time, 2),
            "stats": stats,
        }
        cfg_path = os.path.join(out_dir, "config.json")
        _save_config(cfg, cfg_path)

        print(f"\n[Method2] 完成！耗时 {total_time/60:.1f} min")
        print(f"  覆盖率       : {stats['coverage_ratio']*100:.1f}%")
        print(f"  平均构型数/点 : {stats['mean_configs_per_point']:.1f}")
        print(f"  总接触对数   : {stats['total_contact_pairs']}")
        print(f"  数据文件     : {data_path}")
        print(f"  配置文件     : {cfg_path}")

        return data, cfg


# ══════════════════════════════════════════════════════════════════════════════
# 资源估算
# ══════════════════════════════════════════════════════════════════════════════

def estimate_generation_resources(dof: int = 4, n_task: int = 20):
    """
    基于实测基准数据，估算两种方法在不同参数下的耗时与存储空间。

    基准数据来源（GPU: NVIDIA RTX 3090 / A5000, leaphand finger_0, DOF=4）：
      - Method 2 (FK, N_c=8, N_t=10 → 1000 pts): 8.34s 实测
      - Method 1 (IK, batchsize=5000, 100 pts 抽样): 0.116 s/pt 实测
      - Method 1 实际使用 batchsize=20000，估算因子 ×3
      - Method 1 with_base 实际数据时长（timestamps from data_with_base_dof_*.npy）: ~68min/finger
      - 数据文件大小: Method1 with_base ~3.7-4.4GB/finger, no_base ~240MB/finger

    Args:
        dof:    手指关节数（leaphand=4）
        n_task: task space 每轴离散数（目标20）
    """
    T = n_task ** 3
    print(f"\n{'═'*72}")
    print(f"  资源估算  —  DOF={dof}, task space: {n_task}^3 = {T:,} 点")
    print(f"{'═'*72}")

    # ── Method 2 (FK) ────────────────────────────────────────────────────────
    # 基准: N_c=8(4096 configs), N_t=10(1000 pts) → 8.34s
    # 缩放: time ∝ N_c^dof × N_t^3
    BASE_T2 = 8.34       # s，基准
    BASE_NC = 8 ** dof   # 4096
    BASE_NT = 10 ** 3    # 1000

    print(f"\n{'─'*72}")
    print(f"  Method 2 (FK-based, 无需SDF模型)")
    print(f"{'─'*72}")
    print(f"  {'N_c':>4}  {'总配置':>9}  {'配置维度':>5}  "
          f"{'估算耗时':>10}  {'估算存储':>10}  备注")
    print(f"  {'─'*4}  {'─'*9}  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*20}")

    rows_m2_nobase = [
        (8,  "均匀网格"),
        (10, "均匀网格"),
        (12, "均匀网格"),
        (15, "均匀网格"),
    ]
    for nc, note in rows_m2_nobase:
        n_configs = nc ** dof
        t_est = BASE_T2 * (n_configs / BASE_NC) * (T / BASE_NT)
        # 存储估算：从基准 1.9MB/1000pts/4096configs 按 pairs 线性外推
        # 基准 contact pairs: 86152 for (4096 configs, 1000 pts)
        # 每 pair ≈ 22 bytes（含 dict overhead，4-dof）
        n_pairs = 86152 * (n_configs / BASE_NC) * (T / BASE_NT)
        size_mb = n_pairs * (dof * 4 + 4) * 3.5 / 1e6  # 3.5× overhead
        t_str = (f"{t_est/3600:.1f}h" if t_est > 3600 else
                 f"{t_est/60:.0f}min" if t_est > 60 else f"{t_est:.0f}s")
        s_str = f"{size_mb/1024:.1f}GB" if size_mb > 1024 else f"{size_mb:.0f}MB"
        print(f"  {nc:>4}  {n_configs:>9,}  {dof:>5}D  {t_str:>10}  {s_str:>10}  {note}")

    # with_base rows
    print(f"  --- with_base=True (手指网格 × 底座随机, dof_total={dof+6}) ---")
    rows_m2_base = [
        (8,  20,  "N_c=8, n_base=20"),
        (8,  50,  "N_c=8, n_base=50"),
        (10, 20,  "N_c=10, n_base=20"),
        (10, 50,  "N_c=10, n_base=50"),
    ]
    for nc, nb, note in rows_m2_base:
        n_configs = (nc ** dof) * nb
        # base 变换增加 FK 开销约 1.2×
        t_est = BASE_T2 * (n_configs / BASE_NC) * (T / BASE_NT) * 1.2
        n_pairs = 86152 * (n_configs / BASE_NC) * (T / BASE_NT) * 1.5  # base 增加覆盖
        size_mb = n_pairs * ((dof+6) * 4 + 4) * 3.5 / 1e6
        t_str = (f"{t_est/3600:.1f}h" if t_est > 3600 else
                 f"{t_est/60:.0f}min" if t_est > 60 else f"{t_est:.0f}s")
        s_str = f"{size_mb/1024:.1f}GB" if size_mb > 1024 else f"{size_mb:.0f}MB"
        print(f"  {nc:>4}  {n_configs:>9,}  {dof+6:>5}D  "
              f"{t_str:>10}  {s_str:>10}  {note}")

    # ── Method 1 (IK) ────────────────────────────────────────────────────────
    # 基准: batchsize=5000 → 0.116s/pt；batchsize=20000 估算 ×3 = 0.35s/pt
    # with_base: 实测约 68min/finger (from Nov 3 timestamps)
    M1_NO_BASE_PER_PT = 0.35   # s/pt  (estimated, batchsize=20000)
    M1_WITH_BASE_PER_PT = 0.50 # s/pt  (estimated, batchsize=20000, 10D)
    # 实测有 with_base 數据: 3.7-4.4GB/finger
    M1_NO_BASE_SIZE_MB  = 240  # MB/finger (from data_finger_no_base.npy)
    M1_WITH_BASE_SIZE_GB = 4.0 # GB/finger (avg of actual data_with_base_dof_*.npy)

    print(f"\n{'─'*72}")
    print(f"  Method 1 (IK-based, 需要预训练SDF模型, batchsize=20000)")
    print(f"{'─'*72}")
    print(f"  {'模式':>18}  {'配置维度':>5}  "
          f"{'估算耗时/点':>11}  {'估算总耗时':>10}  {'估算大小':>10}")
    print(f"  {'─'*18}  {'─'*5}  {'─'*11}  {'─'*10}  {'─'*10}")

    for mode, per_pt, size_str, note in [
        ("no_base",   M1_NO_BASE_PER_PT,   f"~{M1_NO_BASE_SIZE_MB}MB",
         "(实测文件推算)"),
        ("with_base", M1_WITH_BASE_PER_PT,
         f"~{M1_WITH_BASE_SIZE_GB}GB", "(实测文件推算)"),
    ]:
        tot = per_pt * T
        t_str = (f"{tot/3600:.1f}h" if tot > 3600 else f"{tot/60:.0f}min")
        print(f"  {mode:>18}  {dof if 'no' in mode else dof+6:>5}D  "
              f"{per_pt:.3f}s/pt   {t_str:>10}  {size_str:>10}  {note}")

    print(f"\n  注意：")
    print(f"  · 以上耗时均为单根手指（串联链），4根手指需 ×4")
    print(f"  · Method 2 with_base 存储较大，建议事后降采样为 .pt 格式")
    print(f"  · Method 1 no_base/with_base 大小来自实际数据文件")
    print(f"  · GPU 型号不同，实际耗时可能有 ±50% 差异")
    print(f"{'═'*72}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 对比测试：效率基准
# ══════════════════════════════════════════════════════════════════════════════

def run_efficiency_benchmark(
    device,
    robot: ParallelRobotLayer,
    paths: dict,
    serial_idx: int = 0,
    save_dir: str = None,
    test_n_task: int = 10,
    test_n_config: int = 8,
    test_with_base: bool = False,
    test_n_base_random: int = 10,
    method1_batchsize: int = 5000,
    method1_n_points: int = 100,
):
    """
    对比测试 Method 1 和 Method 2 的生成效率与数据质量。

    测试使用较小的离散数来快速评估，结果保存到配置文件中以便对比。

    Args:
        device:              torch device
        robot:               ParallelRobotLayer 实例
        paths:               路径字典
        serial_idx:          手指索引
        save_dir:            保存目录
        test_n_task:         测试用的task space离散数 (默认10 → 1000点)
        test_n_config:       测试用的config space离散数 (默认8)
        test_with_base:      是否测试 with_base 模式
        test_n_base_random:  with_base 模式的底座随机采样数
        method1_batchsize:   Method 1 IK优化批大小
        method1_n_points:    Method 1 只测试前N个task space点

    Returns:
        benchmark_results: dict，含两种方法的耗时和覆盖率等统计
    """
    if save_dir is None:
        save_dir = os.path.join(CUR_DIR, "data")

    results = {}
    out_dir = os.path.join(save_dir, robot.robot, "benchmark",
                           f"finger_{serial_idx}")
    os.makedirs(out_dir, exist_ok=True)

    workspace = robot.space_limits.cpu().numpy()
    serial = robot.serials[serial_idx]
    dof = serial.dof

    print(f"\n{'#'*60}")
    print(f"  效率基准测试 — finger_{serial_idx}  (DOF={dof})")
    print(f"  device: {device}")
    print(f"{'#'*60}")

    # ── Method 2 测试 ────────────────────────────────────────────────────────
    print(f"\n[Benchmark] 测试 Method 2 (FK-based)")
    print(f"  参数: n_task={test_n_task}, n_config={test_n_config}, "
          f"总配置数={test_n_config**dof:,}")
    t_start = time.time()
    gen2 = Method2FKDataGenerator(
        device=device,
        robot=robot,
        paths=paths,
        serial_idx=serial_idx,
        with_base=test_with_base,
        n_task_discrete=test_n_task,
        n_config_discrete=test_n_config,
        n_base_random=test_n_base_random,
        threshold=5e-3,
        config_batch_size=200,
        dist_sub_batch_size=5,
        task_batch_size=100,
    )
    data2, cfg2 = gen2.generate(save_dir=save_dir)
    t2 = time.time() - t_start
    results["method2"] = {
        "time_s": round(t2, 2),
        "stats": cfg2["stats"],
        "n_task": test_n_task,
        "n_config": test_n_config,
        "with_base": test_with_base,
        "total_configs": (test_n_config ** dof * test_n_base_random
                          if test_with_base else test_n_config ** dof),
    }
    print(f"[Benchmark] Method 2耗时: {t2:.2f}s = {t2/60:.2f}min")

    # ── Method 1 测试 ────────────────────────────────────────────────────────
    if _HAS_BPSDF and _HAS_TORCHMIN and os.path.exists(paths.get("model", "")):
        print(f"\n[Benchmark] 测试 Method 1 (IK-based)")
        print(f"  参数: n_task={test_n_task}, batchsize={method1_batchsize}")
        if method1_n_points is not None:
            print(f"  注意: 仅测试前 {method1_n_points} 个task space点")

        t_start = time.time()
        gen1 = Method1IKDataGenerator(
            device=device,
            robot=robot,
            paths=paths,
            serial_idx=serial_idx,
            n_task_discrete=test_n_task,
            batchsize=method1_batchsize,
            epsilon=1e-3,
        )

        # 部分测试
        task_pts = _build_task_grid(workspace, test_n_task, device)
        n_test = min(method1_n_points or len(task_pts), len(task_pts))
        data_partial = {}
        t0 = time.time()
        for i in range(n_test):
            q_v, idx_v = gen1._find_q_for_point(task_pts[i].unsqueeze(0))
            data_partial[i] = {
                "x": task_pts[i].cpu().numpy(),
                "q": q_v.detach().cpu().numpy(),
                "idx": idx_v.detach().cpu().numpy(),
            }
        t1_partial = time.time() - t0
        per_point = t1_partial / n_test

        # 推算全量时间
        T_total = test_n_task ** 3
        estimated_total = per_point * T_total
        t1 = time.time() - t_start

        data_partial = _finalize_data(data_partial, dof)
        stats1 = _compute_stats(data_partial)

        results["method1"] = {
            "time_per_point_s": round(per_point, 3),
            "estimated_full_time_s": round(estimated_total, 1),
            "tested_n_points": n_test,
            "stats_partial": stats1,
            "n_task": test_n_task,
            "batchsize": method1_batchsize,
        }
        print(f"[Benchmark] Method 1: {per_point:.3f}s/点 "
              f"→ 推算全量({T_total}点): {estimated_total/60:.1f}min")
    else:
        print("[Benchmark] 跳过 Method 1（缺少 BP-SDF 模型或依赖库）")
        results["method1"] = {"skipped": True, "reason": "missing model or deps"}

    # ── 对比输出 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  效率对比结果汇总")
    print(f"{'='*60}")
    if "time_s" in results["method2"]:
        T2 = test_n_task**3
        print(f"  Method 2 (FK):")
        print(f"    总耗时     : {results['method2']['time_s']:.2f}s "
              f"({results['method2']['time_s']/60:.2f}min)")
        print(f"    task点数  : {T2}")
        print(f"    配置数    : {results['method2']['total_configs']:,}")
        print(f"    覆盖率    : "
              f"{results['method2']['stats']['coverage_ratio']*100:.1f}%")
        print(f"    接触对/点 : "
              f"{results['method2']['stats']['mean_configs_per_point']:.1f}")
    if "time_per_point_s" in results.get("method1", {}):
        r1 = results["method1"]
        print(f"  Method 1 (IK):")
        print(f"    速度      : {r1['time_per_point_s']:.3f}s/点")
        print(f"    推算({T2}点): {r1['estimated_full_time_s']/60:.1f}min")
        print(f"    接触对/点 : "
              f"{r1['stats_partial']['mean_configs_per_point']:.1f}")

    # 保存基准结果
    benchmark_cfg = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "robot": robot.robot,
        "serial_idx": serial_idx,
        "dof": dof,
        "workspace": {"min": workspace[0].tolist(),
                      "max": workspace[1].tolist()},
        "results": results,
    }
    bench_path = os.path.join(out_dir, "benchmark.json")
    _save_config(benchmark_cfg, bench_path)
    print(f"\n  基准结果已保存: {bench_path}")
    print(f"{'='*60}\n")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ══════════════════════════════════════════════════════════════════════════════

def _make_paths(robot: str, base_rdf: str = None) -> dict:
    """根据机器人名称构建标准路径字典。"""
    if base_rdf is None:
        base_rdf = os.path.join(CUR_DIR, "../../RDF")
    p = {
        "urdf":   os.path.join(base_rdf, f"descriptions/{robot}/*.urdf"),
        "meshes": os.path.join(base_rdf, f"descriptions/{robot}/meshes/*.stl"),
        "points": os.path.join(base_rdf, f"data/{robot}/sdf_points/"),
        "model":  os.path.join(base_rdf, f"models/{robot}/BP_8.pt"),
    }
    if robot == "leap":
        p["meshes"] = os.path.join(
            base_rdf, f"descriptions/{robot}/meshes/visual/*.glb")
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LeapHand Contact Data Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--robot", default="leaphand", type=str,
        choices=["leaphand", "leap"],
        help="机器人型号",
    )
    parser.add_argument(
        "--method", default="both", type=str,
        choices=["method1", "method2", "both", "benchmark", "estimate"],
        help=("生成方法: method1(IK), method2(FK), both(两者), "
              "benchmark(效率测试), estimate(仅打印资源估算，无需加载模型)"),
    )
    parser.add_argument(
        "--finger", default=None, type=int,
        help="手指索引 0-3（None=所有手指）",
    )
    parser.add_argument(
        "--with_base", action="store_true",
        help="是否包含6-DOF底座变换（平移+旋转）",
    )
    parser.add_argument(
        "--n_task", default=20, type=int,
        help="Task space 每轴离散数",
    )
    parser.add_argument(
        "--n_config", default=10, type=int,
        help="Method 2: 手指config space 每DOF离散数",
    )
    parser.add_argument(
        "--n_base_random", default=50, type=int,
        help="Method 2 with_base: 底座随机采样数（每个手指配置）",
    )
    parser.add_argument(
        "--batchsize", default=20000, type=int,
        help="Method 1: IK优化批大小",
    )
    parser.add_argument(
        "--threshold", default=5e-3, type=float,
        help="Method 2: 接触距离阈值（米）",
    )
    parser.add_argument(
        "--save_dir", default=None, type=str,
        help="数据保存根目录",
    )
    parser.add_argument(
        "--rdf_path", default=None, type=str,
        help="RDF目录绝对路径（默认 ../../RDF）",
    )
    args = parser.parse_args()

    # 仅打印估算，不加载模型
    if args.method == "estimate":
        estimate_generation_resources(dof=4, n_task=args.n_task)
        raise SystemExit(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    paths = _make_paths(args.robot, args.rdf_path)
    robot_layer = ParallelRobotLayer(device=device, robot=args.robot, paths=paths)

    finger_indices = (
        list(range(len(robot_layer.serials)))
        if args.finger is None
        else [args.finger]
    )
    save_dir = args.save_dir or os.path.join(CUR_DIR, "data")

    for fi in finger_indices:
        print(f"\n{'='*60}")
        print(f"处理 finger_{fi}  "
              f"({robot_layer.serials[fi].all_links})")
        print(f"{'='*60}")

        if args.method in ("benchmark",):
            run_efficiency_benchmark(
                device=device,
                robot=robot_layer,
                paths=paths,
                serial_idx=fi,
                save_dir=save_dir,
                test_n_task=args.n_task,
                test_n_config=args.n_config,
                test_with_base=args.with_base,
                test_n_base_random=args.n_base_random,
            )

        if args.method in ("method2", "both"):
            gen2 = Method2FKDataGenerator(
                device=device,
                robot=robot_layer,
                paths=paths,
                serial_idx=fi,
                with_base=args.with_base,
                n_task_discrete=args.n_task,
                n_config_discrete=args.n_config,
                n_base_random=args.n_base_random,
                threshold=args.threshold,
            )
            gen2.generate(save_dir=save_dir)

        if args.method in ("method1", "both"):
            if not (os.path.exists(paths["model"])
                    and _HAS_BPSDF and _HAS_TORCHMIN):
                print(f"[警告] 跳过 Method 1 finger_{fi}："
                      f"缺少模型或依赖 (model={paths['model']})")
                continue
            gen1 = Method1IKDataGenerator(
                device=device,
                robot=robot_layer,
                paths=paths,
                serial_idx=fi,
                with_base=args.with_base,
                n_task_discrete=args.n_task,
                batchsize=args.batchsize,
            )
            gen1.generate(save_dir=save_dir)
