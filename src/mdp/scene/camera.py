import torch

import isaaclab.utils.math as math_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
import omni.usd
from pxr import Sdf, Usd, UsdGeom, UsdLux

PARTITION_KIT_ARG = "--/renderer/scenePartitioning/enabled=true"

CAMERA_OFFSET = (0.32, 0.0, 0.05)    # camera position in base frame 

FRONT_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/front_cam",
    width=224, # DINO input: 16x16
    height=224,
    offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0 ,0.0), convention="world"),
    data_types=["rgb"],
    spawn=PinholeCameraCfg(focal_length=1.5, horizontal_aperture=4.0, clipping_range=(0.05, 30.0)),
)

def follow_robot(scene: InteractiveScene) -> None:
    """ Put every camera on its robot's frame """
    robot = scene["robot"]
    pos = robot.data.root_pos_w.torch
    quat = robot.data.root_quat_w.torch
    offset = torch.tensor(CAMERA_OFFSET, device=pos.device).expand_as(pos)
    scene["camera"].set_world_poses(pos + math_utils.quat_apply(quat, offset), quat, convention="world")

def set_scene_partition(num_envs: int) -> int:
    stage = omni.usd.get_context().get_stage()
    n_cams = 0
    for i in range(num_envs):
        env_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}")
        if not env_prim.IsValid():
            continue
        token = f"env_{i}"
        for prim in Usd.PrimRange(env_prim):
            if prim == env_prim:
                name = "primvars:omni:scenePartition"
            elif prim.IsA(UsdGeom.Camera) or prim.HasAPI(UsdLux.LightAPI):
                name = "omni:scenePartition"
                n_cams += int(prim.IsA(UsdGeom.Camera))
            else:
                continue
            attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Token, custom=True, variability=Sdf.VariabilityUniform)
            attr.Set(token)
    return n_cams


_ORIGINAL_SCENE_INIT = InteractiveScene.__init__

def use_scene_partitions() -> None:
    def init_with_partitions(self, *args, **kwargs):
        _ORIGINAL_SCENE_INIT(self, *args, **kwargs)
        n_cams = set_scene_partition(self.num_envs)
        assert n_cams == self.num_envs, f"{n_cams} cameras tagged for {self.num_envs} envs"
    InteractiveScene.__init__ = init_with_partitions


def restore_scene_init() -> None:
    InteractiveScene.__init__ = _ORIGINAL_SCENE_INIT