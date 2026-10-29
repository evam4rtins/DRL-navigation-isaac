from pathlib import Path

import torch
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.utils.math import euler_xyz_from_quat

from navigation_env.managers.camera_manager import CameraManager
from navigation_env.managers.map_manager import MapManager
from navigation_env.managers.path_manager import PathManager

from cfg.CFG import get_apartment_origins, get_env_apartments_ids
from mdp.scene.go2_scene_cfg import Go2MultiApartmentSceneCfg
from mdp.scene.multi_apartment import restore_grid_origins, use_apartment_origins, restore_clone_plan, use_round_robin_cloning

class NavEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        assert "scene_path" in kwargs, "Must provide scene_path as a keyword argument"
        scene_paths = kwargs["scene_path"]
        scene_paths = [scene_paths] if isinstance(scene_paths, (str, Path)) else list(scene_paths)
        scene_paths = [Path(p) for p in scene_paths]
        num_apts = len(scene_paths)

        self.multi_apartment = isinstance(cfg.scene, Go2MultiApartmentSceneCfg)
        if self.multi_apartment:
            num_apt_prims = sum(1 for name in vars(cfg.scene) if name.startswith("apt_"))
            assert num_apt_prims == num_apts, f"Scene has {num_apt_prims} apartments but {num_apts} scene paths"
            use_apartment_origins(get_apartment_origins())
            use_round_robin_cloning()
        else:
            assert num_apts == 1, "A list of scene paths needs the multi-apartment scene"

        try:
            super().__init__(cfg, render_mode, **kwargs)
        finally: 
            restore_grid_origins()
            restore_clone_plan
        
        env_apt_ids = get_env_apartments_ids(self.num_envs, num_apts).to(self.device)
        if self.multi_apartment:
            expected = get_apartment_origins().to(self.device)[env_apt_ids]
            assert torch.allclose(self.scene.env_origins, expected, atol=1e-4), (
                "env_origins are not the apartment origins: did the grid_transforms patch stop working?"
            )

        map_img_paths = [str(p.parent / (p.stem + "_map.png")) for p in scene_paths]
        map_yaml_paths = [str(p.parent / (p.stem + "_map.yaml")) for p in scene_paths]
        self._map_manager = MapManager(
            map_img_paths, map_yaml_paths, self.num_envs, device=self.device, env_apt_ids=env_apt_ids
            )

        self.path_manager = PathManager(
            scene_path=scene_paths, map_mgr=self._map_manager, num_envs=self.num_envs, device=self.device
            )

        # Action buffer with the past 5 actions (Δx, Δy, Δtheta)
        self._action_buffer = torch.zeros((self.num_envs, 5, 3), dtype=torch.float32, device=self.device)

        # Store past 10 robot linear + angular velocity (x,y,theta)
        self._velocity_buffer = torch.zeros((self.num_envs, 10, 3), dtype=torch.float32, device=self.device)
        self._old_yaw = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)

        self._lidar_buffer = None

        camera_controller = ViewportCameraController(self, cfg=ViewerCfg(origin_type="world"))
        self.camera_manager = CameraManager(
            camera_controller,
            camera_relative_pos=torch.tensor([-0.6, 0.0, 0.8]),
            camera_lookat=torch.tensor([1.6, 0.0, -1.0]),
        )

        self.long_horizon = kwargs.get("use_long_horizon", False)
        self.sample_voronoi_probability = kwargs.get("sample_voronoi_probability", 0.25)
        self.debug_vis = kwargs.get("debug_vis", False)
        self.collision_termination_thresh = 1.0
        self.nominal_weight = 0.0  # scale for nominal action

        self.episode_counter = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # last success rate / path length per apartment, reused at steps where an apartment had no reset
        self._apt_success = torch.zeros(num_apts, device=self.device)
        self._apt_path_len = torch.zeros(num_apts, device=self.device)

    @property
    def env_apt_ids(self) -> torch.Tensor:
        return self._map_manager.env_apt_ids #(num_envs,) apartment index of every env

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        if "log" in self.extras:  # Fixes wrong logging of rewards and dones in the first step after reset
            self.extras.pop("log", None)

        retVal = super().step(action)

        # Update action buffer
        self._action_buffer = torch.roll(self._action_buffer, shifts=1, dims=1)
        self._action_buffer[:, 0, :] = action.to(self.device)

        ## Update linear velocity
        robot: Articulation = self.scene["robot"]
        self._velocity_buffer = torch.roll(self._velocity_buffer, shifts=1, dims=1)
        # store only x, y linear velocity
        self._velocity_buffer[:, 0, :2] = robot.data.root_com_lin_vel_w.torch[:, :2].to(self.device)

        ## Angular velocity
        _, _, new_yaw = euler_xyz_from_quat(robot.data.root_com_quat_w.torch)
        theta = (new_yaw - self._old_yaw) / getattr(self.sim, "dt", 0.005) * getattr(self.sim, "decimation", 16)
        self._velocity_buffer[:, 0, 2] = theta
        self._old_yaw = new_yaw
        #TODO: replace with this:
        #self._velocity_buffer[:, 0, 2] = robot.data.root_com_ang_vel_w.torch[:, 2].to(self.device)

        self.update_follow_camera()
        self._update_lidar_buffer()
        if self.multi_apartment:
            self._log_apartment_stats()

        return retVal

    def _log_apartment_stats(self) -> None:
        if "log" not in self.extras:  # only written at steps where some env was reset
            return
        done = self.reset_buf.bool()
        goal = self.termination_manager.get_term("goal_reached").bool()
        for k, name in enumerate(self.path_manager.map_names):
            ended = done & (self.env_apt_ids == k)
            if ended.any():
                self._apt_success[k] = goal[ended].float().mean()
                self._apt_path_len[k] = self.path_manager.initial_path_length[ended].float().mean()
            self.extras["log"][f"Apartment_success/{name}"] = self._apt_success[k].item()
            self.extras["log"][f"Apartment_path_length/{name}"] = self._apt_path_len[k].item()


    def teleport_robots(self, env_ids: torch.Tensor) -> None:
        ## set cartesian position, quaternion orientation in (x, y, z, w), and linear and angular velocity
        ## reset positions only for specified env indices
        robot: Articulation = self.scene["robot"]

        state = self.path_manager.compute_robot_teleport_state(
            env_ids, self.scene.env_origins, robot.data.default_root_state.torch[0, 2]
        )

        robot.write_root_com_state_to_sim(state, env_ids)

        # reset joints to default state after teleporting root poses
        default_pos = robot.data.default_joint_pos.torch[env_ids]
        default_vel = robot.data.default_joint_vel.torch[env_ids]
        robot.set_joint_position_target(default_pos, None, env_ids)
        robot.write_joint_state_to_sim(default_pos, default_vel, None, env_ids)

        self._map_manager.reset_costmaps(env_ids)  # clear costmaps on teleport to avoid stale obstacle data

        if env_ids[0] == 0:  # only update camera on the first env for now
            self.update_follow_camera(smoothing=1.0)  # snap camera to new position on reset

    def sample_task_on_reset(self, env_ids: torch.Tensor) -> None:
        self.path_manager.sample_nav_task(env_ids, sample_voronoi_prob=self.sample_voronoi_probability)

    def manual_replan(self, env_ids: torch.Tensor) -> None:
        robot = self.scene["robot"]
        self.path_manager.compute_global_plan(
            env_ids, robot.data.root_com_pos_w.torch[env_ids], self.scene.env_origins[env_ids]
        )

    def _update_lidar_buffer(self) -> None:
        if "lidar" in self.scene.sensors:
            lidar = self.scene.sensors["lidar"]
            self._map_manager.update_local_costmap(lidar.data, self.scene.env_origins)

    def update_follow_camera(self, smoothing: float = 0.3):
        if not self.debug_vis and not self.sim.has_gui and self.render_mode != "rgb_array":
            return

        robot_pos = self.scene["robot"].data.root_pos_w.torch[0]
        robot_quat = self.scene["robot"].data.root_quat_w.torch[0]

        self.camera_manager.update(robot_pos, robot_quat, smoothing_factor=smoothing)
