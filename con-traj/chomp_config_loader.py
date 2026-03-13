"""
Configuration loader for CHOMP optimizer.
"""
import yaml
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path


class CHOMPConfig:
    """Configuration object for CHOMP optimizer."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration from dictionary."""
        self._config = config_dict
        self._parse_config()
    
    def _parse_config(self):
        """Parse and validate configuration."""
        # Trajectory
        traj = self._config.get('trajectory', {})
        self.T = int(traj.get('num_points', 1000))
        self.D = int(traj.get('dimensions', 2))
        
        # Calculate dt if not explicitly provided
        dt_provided = traj.get('time_step')
        if dt_provided is not None:
            self.dt = float(dt_provided)
        else:
            self.dt = 1.0 / (self.T - 1)
        
        # Task
        task = self._config.get('task', {})
        self.start = np.array(task.get('start', [0.1, 0.1]), dtype=float)
        self.goal = np.array(task.get('goal', [0.6, 0.9]), dtype=float)
        
        # Obstacles
        obs = self._config.get('obstacles', {})
        self.clearance = float(obs.get('clearance', 0.04))
        self.obstacle_shapes = obs.get('shapes', [])
        
        # Smoothness
        smooth = self._config.get('smoothness', {})
        self.velocity_weight = float(smooth.get('velocity_weight', 1.0))
        self.acceleration_weight = float(smooth.get('acceleration_weight', 1.0))
        self.smoothness_damping = float(smooth.get('damping', 1e-4))
        
        # Obstacle cost
        obs_cost = self._config.get('obstacle_cost', {})
        self.lambda_obs = float(obs_cost.get('weight', 0.1))
        
        # Optimizer
        opt = self._config.get('optimizer', {})
        self.step_size = float(opt.get('step_size', 0.001))
        self.num_iterations = int(opt.get('num_iterations', 100))
        
        # Visualization
        vis = self._config.get('visualization', {})
        self.plot_xlim = tuple(vis.get('plot_xlim', [0.0, 1.0]))
        self.plot_ylim = tuple(vis.get('plot_ylim', [0.0, 1.0]))
        self.save_trajectory_plot = bool(vis.get('save_trajectory_plot', True))
        self.trajectory_plot_name = str(vis.get('trajectory_plot_name', 'chomp_2d_result.png'))
        self.save_gradient_plot = bool(vis.get('save_gradient_plot', True))
        self.gradient_plot_name = str(vis.get('gradient_plot_name', 'chomp_2d_final_gradients.png'))
        self.gradient_scale_factor = float(vis.get('gradient_scale_factor', 5.0))
    
    @classmethod
    def from_file(cls, config_path: str) -> 'CHOMPConfig':
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            CHOMPConfig instance
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            config_dict = {}
        
        return cls(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CHOMPConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            CHOMPConfig instance
        """
        return cls(config_dict)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"""CHOMPConfig(
  Trajectory: T={self.T}, D={self.D}, dt={self.dt:.6f}
  Task: start={self.start}, goal={self.goal}
  Obstacles: clearance={self.clearance}, shapes={len(self.obstacle_shapes)}
  Smoothness: vel_weight={self.velocity_weight}, acc_weight={self.acceleration_weight}, damping={self.smoothness_damping}
  Obstacle Cost: lambda_obs={self.lambda_obs}
  Optimizer: step_size={self.step_size}, n_iters={self.num_iterations}
)"""
