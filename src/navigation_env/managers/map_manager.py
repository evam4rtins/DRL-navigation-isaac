from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import yaml

if TYPE_CHECKING:
    from isaaclab.sensors import RayCasterData

#: Narrowest gap (in meters) the planner is allowed to route through.
MIN_TRAVERSABLE_GAP = 0.20

def _as_tensor(x) -> torch.Tensor:
    return x.torch if hasattr(x, "torch") else x

class MapManager:
    def __init__(
            self,
            map_png_paths: str | Path | list[str | Path],
            map_yaml_paths: str | Path | list[str | Path], 
            num_envs: int, device: torch.device | str, 
            env_apt_ids: torch.Tensor | None = None, 
        ):

        if isinstance(map_png_paths, (str, Path)):
            map_png_paths, map_yaml_paths = [map_png_paths], [map_yaml_paths]
        assert len(map_png_paths) == len(map_yaml_paths)
       
        self.device = device
        self.num_envs = num_envs
        self.num_apts = len(map_png_paths)
        self.map_img_paths = [str(p) for p in map_png_paths]
        self.map_img_path = self.map_img_paths[0]

        self._img_metas, self._map_imgs = [], []
        for png, yml in zip(self.map_img_paths, map_yaml_paths):
            with open(yml, "r") as f:
                self._img_metas.append(yaml.safe_load(f))
            img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Failed to read map image: {png}")
            self._map_imgs.append(img)
        
        resolutions = {m["resolution"] for m in self._img_metas}
        if len(resolutions) != 1:
            raise ValueError(f"All maps must share one resolution, got {sorted(resolutions)}")
        self._resolution = resolutions.pop()

        # per-apartment geometry (K,2)
        shapes = [img.shape for img in self._map_imgs]
        self.map_shape = torch.tensor(shapes, dtype=torch.long, device=device)
        self.map_origin = torch.tensor([m["origin"][:2] for m in self._img_metas], dtype=torch.float32, device=device)
        self.H_max = max(h for h, _ in shapes)
        self.W_max = max(w for _, w in shapes)

        # static occupancy of every map, padded to (K, H_max, W_max)
        self._occupied = np.ones((self.num_apts, self.H_max, self.W_max), dtype=bool)
        for k, img in enumerate(self._map_imgs):
            h, w = img.shape
            self._occupied[k,:h,:w] = img == 0  # true = occupied
        
        # in which apartment each env is
        if env_apt_ids is None:
            env_apt_ids = torch.zeros(num_envs, dtype=torch.long)
        assert env_apt_ids.shape == (num_envs,), "env_apt_ids must have one entry per env"
        assert int(env_apt_ids.max()) < self.num_apts, "env_apt_ids refers to a missing apartment"
        self.env_apt_ids = env_apt_ids.to(device=device, dtype=torch.long)
        self._env_apt_ids_cpu = self.env_apt_ids.cpu().numpy()
   
        # Remember: in costmaps 0 = free space, inf = occupied
        self.local_costmaps = torch.zeros((num_envs, 1, self.H_max, self.W_max), dtype=torch.float32, device=device)

    @property
    def resolution(self) -> float:
        return self._resolution

    @property
    def origin(self) -> tuple[float, float]:
        """ origin of apartment 0 (single-map version) """
        return self.apt_origin(0)

    @property
    def shape(self) -> tuple[int, int]:
        """ (H, W) of apartment 0 (single-map version) """
        return self.apt_shape(0)
    
    def apt_origin(self, apt_id: int) -> tuple[float, float]:
        return self._img_metas[apt_id]["origin"][:2]
    
    def apt_shape(self, apt_id: int) -> tuple[int, int]:
        return self._map_imgs[apt_id].shape
    
    def apt_map_img(self, apt_id: int) -> np.ndarray:
        return self._map_imgs[apt_id].copy()
    
    @property
    def map_img(self) -> np.ndarray:
        """ floor plan of apartment 0 (single-map version) """
        return self.apt_map_img(0)

    @property
    def occupancy_gridmap(self) -> np.ndarray:
        """ ROS occupancy grid of apartment 0 (single map version) """
        return self.apt_occupancy_gridmap(0)
    
    def apt_occupancy_gridmap(self, apt_id: int) -> np.ndarray:
        img = self._map_imgs[apt_id]
        gray_pixels = (img > 1) & (img < 254)
        white_pixels = img >= 254
        black_pixels = img <= 1
        # use -1, [0...100] to represent respectively unknown, free, occupied.
        ogm = img.astype(np.int8, copy=True)
        ogm[gray_pixels] = -1
        ogm[white_pixels] = 0
        ogm[black_pixels] = 100
        return ogm

    def obstacle_masks(self, env_ids: torch.Tensor) -> list[np.ndarray]:
        """Returns a list of masks, one per env, cropped to each map's size

        This is the only costmap data that has to leave the device: the inflation itself runs on the
        host, next to the A* that consumes it (see `PathManager._inflate_and_plan`).
        """
        env_ids_cpu = env_ids.cpu().numpy()
        lidar_hits = (self.local_costmaps[env_ids, 0] != 0.0).cpu().numpy()  # (n, H_max, W_max)
        masks = []
        for i, env_id in enumerate(env_ids_cpu):
            k = self._env_apt_ids_cpu[env_id]
            h, w = self._map_imgs[k].shape
            masks.append(self._occupied[k, :h, :w] | lidar_hits[i, :h, :w])
        return masks

    @property
    def clearance_px(self) -> float:
        """Distance the planner keeps from every obstacle, in cells."""
        return 0.5 * MIN_TRAVERSABLE_GAP / self.resolution

    @staticmethod
    def create_inflated_costmap(
        occupied: np.ndarray, clearance_px: float, free_cells: tuple = (), inflation_scale: int = 2
    ) -> np.ndarray:
        """Inflated A* costmap for a single environment.

        Cells closer than `clearance_px` to an obstacle are inflated

        The clearance is lifted in a `clearance_px` window around each of `free_cells` (the robot and
        its goal), so that an endpoint which ended up inside the band does not become unplannable.
        """
        dist_transform = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        costmap = np.clip(dist_transform.max() - (dist_transform * inflation_scale), 0, 255).astype(np.float32)

        blocked = dist_transform < clearance_px
        radius = int(np.ceil(clearance_px))
        for row, col in free_cells:
            if not blocked[row, col]:
                continue
            window = (slice(max(row - radius, 0), row + radius + 1), slice(max(col - radius, 0), col + radius + 1))
            blocked[window] = occupied[window]  # only the obstacles themselves are left in the way

        costmap[blocked] = np.inf
        return costmap + 1.0  # min_cost = 1.0 for A*

    def reset_costmaps(self, env_ids: torch.Tensor):
        # self.debug_vis()
        self.local_costmaps[env_ids] = 0.0

    def update_local_costmap(self, lidar_scan: RayCasterData, env_origins: torch.Tensor):
        """mark the lidar hits closer than 3m in each env's costmap"""
        thresh = 3.0
        ray_hits_w = lidar_scan.ray_hits_w.torch  # (B, N_rays, 3), warp-backed ProxyArray in IsaacLab 3.0
        pos_w = _as_tensor(lidar_scan.pos_w) # (B, 3)
        scan_dist = torch.norm(ray_hits_w - pos_w.unsqueeze(1), dim=-1)  # (B, N_rays)
        mask = scan_dist < thresh  # (B, N_rays)

        scan_local = ray_hits_w - env_origins.unsqueeze(1)
        scan_map_coords = self.local_to_map_coords(scan_local, self.env_apt_ids)  # (B, N_rays, 2)

        row = scan_map_coords[..., 0]
        col = scan_map_coords[..., 1]

        B, N = mask.shape
        batches = torch.arange(self.num_envs, device=self.device)[:, None].expand(B, N)
        self.local_costmaps[batches[mask], 0, row[mask], col[mask]] = torch.inf

    def _gather(self, values: torch.Tensor, apt_ids, batch_dims: int) -> torch.Tensor:
        # pick the rows of a per-apartment table ``values`` (K, C) for ``apt_ids``, shaped to broadcast over points.
        ids = torch.as_tensor(0 if apt_ids is None else apt_ids, dtype=torch.long, device=values.device)
        rows = values[ids]
        missing = batch_dims - ids.dim()
        assert missing >= 0, "apt_ids has more dims than the points"
        return rows.reshape(*ids.shape, *([1] * missing), values.shape[-1])

    def map_to_local_coords(self, map_coords: torch.Tensor, apt_ids=None) -> torch.Tensor:
        assert len(map_coords.shape) == 2 and map_coords.shape[1] == 2, "Expected map_coords to have shape (N, 2)"
        map_coords = map_coords.clone()
        W = self._gather(self.map_shape[:, 1:2], apt_ids, 1)[..., 0]  # (N,) or (1,): map width of each point
        origin = self._gather(self.map_origin, apt_ids, 1)  # (N, 2) or (1, 2)
        
        map_coords[:, 1] = W - map_coords[:, 1]  # flip x axis
        map_xy = map_coords[:, [1, 0]]  # swap from (row,col) to (x, y)

        scaled = map_xy * self.resolution  # (N, 2)
        return scaled + origin  # (N, 2)

    def local_to_map_coords(self, local_coords: torch.Tensor, apt_ids=None) -> torch.Tensor:
        # data can be (B, N, 2|3) or (N, 2|3)
        if local_coords.size(-1) > 2:
            local_coords = local_coords[..., :2]  # only consider x, y

        local_xy = local_coords.clone()
        batch_dims = local_xy.dim() - 1
        origin = self._gather(self.map_origin, apt_ids, batch_dims)  # broadcastable to local_xy
        shape = self._gather(self.map_shape, apt_ids, batch_dims).to(torch.int32)  # (..., 2): H, W of each point
        H, W = shape[..., 0], shape[..., 1]

        scaled = ((local_xy - origin) / self.resolution).to(torch.int)

        scaled[..., 0] = W - scaled[..., 0]  # flip x axis
        map_coords = scaled[..., [1, 0]]  # swap to (row, col)

        zero = torch.zeros_like(H)
        map_coords[..., 0] = torch.minimum(torch.maximum(map_coords[..., 0], zero), H - 1)  # row
        map_coords[..., 1] = torch.minimum(torch.maximum(map_coords[..., 1], zero), W - 1)  # col
        return map_coords

    # def debug_vis(self):
    #     import matplotlib.pyplot as plt

    #     # check costmaps from two separate batches
    #     local_costmap_1 = self.local_costmaps[0, 0].cpu().numpy()
    #     local_costmap_2 = self.local_costmaps[1, 0].cpu().numpy()
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(local_costmap_1, cmap="gray")
    #     plt.title("Local Costmap with Lidar Updates (Batch 0)")
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(local_costmap_2, cmap="gray")
    #     plt.title("Local Costmap with Lidar Updates (Batch 1)")
    #     plt.show()


#     def debug_vis_global(self):
#         import matplotlib.pyplot as plt

#         # check costmaps from two separate batches
#         global_costmap_1 = self._costmap_backup[0, 0].cpu().numpy()
#         global_costmap_2 = self._global_costmap[0, 0].cpu().numpy()
#         plt.figure(figsize=(12, 6))
#         plt.subplot(1, 2, 1)
#         plt.imshow(global_costmap_1, cmap="gray")
#         plt.title("Base Global Costmap (Batch 0)")
#         plt.subplot(1, 2, 2)
#         plt.imshow(global_costmap_2, cmap="gray")
#         plt.title("Updated Global Costmap (Batch 0)")
#         plt.show()

#     def print_debug(self):
#         t0 = time.time()
#         for _ in range(128):
#             cpu_impl = self._create_inflated_costmap_cpu(self.map_img, inflation_scale=2)
#         print(f"CPU costmap inflation took {time.time() - t0:.4f} seconds")
#         for _ in range(5):
#             gpu_impl = self._create_inflated_costmap_gpu(self._map_tensor, torch.arange(0, self.num_envs), inflation_scale=2)[
#                 0, 0
#             ].cpu()
#         print(f"GPU costmap inflation took {(time.time() - t0)/5:.4f} seconds")
#         import matplotlib.pyplot as plt

#         plt.figure(figsize=(12, 6))
#         plt.subplot(1, 2, 1)
#         plt.title("CPU Inflated Costmap")
#         plt.imshow(cpu_impl, cmap="gray")
#         plt.subplot(1, 2, 2)
#         plt.title("GPU Inflated Costmap")
#         plt.imshow(gpu_impl, cmap="gray")
#         plt.show()


# if __name__ == "__main__":
#     from cfg.CFG import DEVICE, SCENE_USD_PATH

#     map_png = str(SCENE_USD_PATH.parent / SCENE_USD_PATH.stem) + "_map.png"
#     map_yaml = str(SCENE_USD_PATH.parent / SCENE_USD_PATH.stem) + "_map.yaml"
#     costmap_manager = MapManager(map_png, map_yaml, num_envs=128, device=DEVICE)
#     costmap_manager.print_debug()
