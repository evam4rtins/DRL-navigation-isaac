"""
Test: an RGB camera on every Go2 in the multi apartement scene.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--maps", nargs="+", default=["31", "36"])
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", default="go2_lidar_multi_full", choices=["go2_lidar_multi", "go2_lidar_multi_full"])
parser.add_argument("--mount", default="sibling", choices=["nested", "sibling"])
parser.add_argument("--res", type=int, default=224, help="image size (224 = DINOv2 input, 16x16 patches)")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--save_every", type=int, default=50)
parser.add_argument("--checkpoint", type=str, default=None, help="drive with the teacher (otherwise zero actions)")
parser.add_argument("--out", type=str, default="logs/tests/camera")
parser.add_argument("--renderer", default="rtx", choices=["rtx", "ovrtx"],
                    help="rtx = default Isaac RTX renderer; ovrtx = OVRTX renderer (isaaclab_ov, needs the ovrtx "
                         "package), which partitions the scene per env by itself")
parser.add_argument("--partition", action="store_true",
                    help="per-env RTX scene partitions: camera i renders only env_i (its robot, its doors/chairs) "
                         "plus the global apartments")
parser.add_argument("--hide_robots", action="store_true",
                    help="make every Go2 invisible (rendering only): the cameras stop seeing the other robots")
parser.add_argument("--no_camera", action="store_true", help="same run without cameras: the speed baseline")
parser.add_argument("--dome_light", type=float, default=0.0, help="add a dome light of this intensity (0 = none)")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_camera
simulation_app = AppLauncher(args).app

import os
import time

import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg

from cfg.CFG import ROOT_DIR, get_scene_usd_paths, set_map_names
from tasks.task_utils import get_env_config

CAM_OFFSET = (0.4, 0.0, 0.1)  # m in the base frame: front of the robot, a bit above the base


def camera_cfg():
    prim = "{ENV_REGEX_NS}/Go2/base/front_cam" if args.mount == "nested" else "{ENV_REGEX_NS}/front_cam"
    return TiledCameraCfg(
        prim_path=prim,
        width=args.res,
        height=args.res,
        # nested: fixed offset on the base. sibling: the pose is set every step, the offset is applied there
        offset=TiledCameraCfg.OffsetCfg(pos=CAM_OFFSET if args.mount == "nested" else (0.0, 0.0, 0.0),
                                        convention="world"),
        data_types=["rgb"],
        **renderer_kwargs(),
        spawn=PinholeCameraCfg(focal_length=1.5, horizontal_aperture=4, clipping_range=(0.05, 30.0)),
    )


def renderer_kwargs():
    if args.renderer == "rtx":
        return {}
    from isaaclab_ov.renderers import OVRTXRendererCfg

    out = os.path.abspath(f"{args.out}_ovrtx_tmp")
    os.makedirs(out, exist_ok=True)
    return {"renderer_cfg": OVRTXRendererCfg(temp_usd_dir=out, log_level="warn",
                                             log_file_path=os.path.join(out, "ovrtx.log"))}


def follow_robots(base):
    """sibling mount: put every camera on its robot's base (+ CAM_OFFSET in the base frame)."""
    robot = base.scene["robot"]
    pos = robot.data.root_pos_w.torch
    quat = robot.data.root_quat_w.torch
    offset = torch.tensor(CAM_OFFSET, device=pos.device).expand_as(pos)
    cam_pos = pos + math_utils.quat_apply(quat, offset)
    base.scene["camera"].set_world_poses(cam_pos, quat, convention="world")


def set_partitions(num_envs):
    """Same attributes as Isaac Lab's IsaacRtxRenderer.prepare_stage (newer versions; not in 3.0.0-beta2):
    primvars:omni:scenePartition = env_i on the env root (inherited by all its geometry) and
    omni:scenePartition = env_i on its cameras and lights. Prims outside /World/envs (the global apartments) have
    no partition and stay visible to every camera."""
    import omni.usd
    from pxr import Sdf, Usd, UsdGeom, UsdLux

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
                n_cams += prim.IsA(UsdGeom.Camera)
            else:
                continue
            attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Token, custom=True, variability=Sdf.VariabilityUniform)
            attr.Set(token)
    print(f"scene partitions: {num_envs} envs, {n_cams} cameras tagged")


def hide_robots(num_envs):
    """Rendering only: physics, contact sensors and the lidar (which does not ray-cast robots) are unchanged.
    The cameras are siblings of the Go2 prims, so they are not hidden with them."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    for i in range(num_envs):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Go2")
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()


def save_grid(rgb, base, maps, path, step):
    n = rgb.shape[0]
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    tiles = []
    apt = base.env_apt_ids.cpu().tolist()
    for i in range(rows * cols):
        if i < n:
            img = cv2.cvtColor(rgb[i, ..., :3], cv2.COLOR_RGB2BGR)
            cv2.putText(img, f"env {i} {maps[apt[i]][-4:]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        else:
            img = np.zeros_like(tiles[0])
        tiles.append(img)
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    cv2.imwrite(os.path.join(path, f"step_{step:03d}.png"), grid)


def main():
    maps = set_map_names(args.maps)
    cfg = get_env_config(args.task)
    cfg.curriculum = None
    cfg.scene.num_envs = args.num_envs
    if not args.no_camera:
        cfg.scene.camera = camera_cfg()
    if args.dome_light > 0:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg

        cfg.scene.light = AssetBaseCfg(prim_path="/World/DomeLight",
                                       spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=args.dome_light))
    cfg.seed = args.seed
    gym.register(
        id="Isaac-camera-test-v0",
        entry_point="navigation_env.NavigationEnv:NavEnv",
        disable_env_checker=True,
        kwargs={"scene_path": get_scene_usd_paths(), "use_long_horizon": False, "sample_voronoi_probability": 0.25},
    )
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    if args.partition:
        # the partitions must be on the stage before the cameras' render products and the Fabric copy of the
        # scene are created (sim.reset inside the env constructor): author them right after the scene is built
        from isaaclab.scene import InteractiveScene

        scene_init = InteractiveScene.__init__

        def init_with_partitions(self, *a, **kw):
            scene_init(self, *a, **kw)
            set_partitions(self.num_envs)

        InteractiveScene.__init__ = init_with_partitions

    env = RslRlVecEnvWrapper(gym.make("Isaac-camera-test-v0", cfg=cfg))
    base = env.unwrapped
    cam = None if args.no_camera else base.scene["camera"]
    if cam is not None:
        print(f"\ncamera prims: {cam.cfg.prim_path}, {cam.num_instances} instances for {args.num_envs} envs "
              f"(should be equal), mount = {args.mount}")

    policy = None
    if args.checkpoint:
        from rsl_rl.runners import OnPolicyRunner

        from policy.NavPolicyv2 import load_policy_checkpoint, make_go2_policy_cfg

        policy_cfg = make_go2_policy_cfg(cfg.obs_groups)
        policy_cfg["num_envs"] = args.num_envs
        runner = OnPolicyRunner(env, policy_cfg, log_dir=None, device=policy_cfg["device"])
        ckpt = args.checkpoint if args.checkpoint.startswith("/") else str(ROOT_DIR / args.checkpoint)
        load_policy_checkpoint(runner, ckpt)
        policy = runner.get_inference_policy(env.device)

    if args.partition:
        import carb

        key = "/rtx/scenePartitioning/showAllPartitionsByDefault"
        print(f"{key} = {carb.settings.get_settings().get(key)}")
    if args.hide_robots:
        hide_robots(args.num_envs)
    out = f"{args.out}_{args.mount}_{args.renderer}" + (f"_light{int(args.dome_light)}" if args.dome_light else "") \
        + ("_hidden" if args.hide_robots else "") + ("_partition" if args.partition else "")
    os.makedirs(out, exist_ok=True)
    obs, _ = env.reset()
    t0, timed = None, 0
    for step in range(args.steps):
        if cam is not None and args.mount == "sibling":
            follow_robots(base)
        with torch.no_grad():
            if policy is None:
                action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            else:
                policy(obs, stochastic_output=True)
                action = policy.output_mean
        obs, _, _, _ = env.step(action)
        if step == 10:  # skip the warm-up steps in the timing
            torch.cuda.synchronize()
            t0 = time.time()
        elif step > 10:
            timed += 1
        if cam is not None and (step % args.save_every == 0 or step == args.steps - 1):
            rgb = cam.data.output["rgb"].torch.cpu().numpy()  # (N, H, W, 3|4) uint8
            if step == 0:
                print(f"rgb {rgb.shape} {rgb.dtype}, mean {rgb[..., :3].mean():.1f} (about 0 = black: no light?)")
            save_grid(rgb, base, maps, out, step)

    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"\n{timed} steps in {dt:.1f} s: {timed / dt:.1f} env steps/s, "
          f"{timed * args.num_envs / dt:.0f} robot steps/s, {args.num_envs} robots, "
          + ("no camera" if cam is None else f"cameras at {args.res}x{args.res}"))
    print(f"images in {out}/")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    simulation_app.close()
