import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--maps", nargs="+", default=["31"])
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--save_every", type=int, default=25)
parser.add_argument("--checkpoint", type=str, default=None, help="drive with the teacher (otherwise zero actions)")
parser.add_argument("--out", type=str, default="logs/tests/vision_task")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.kit_args = ((args.kit_args or "") + " --/renderer/scenePartitioning/enabled=true").strip()
simulation_app = AppLauncher(args).app

import os
import time

import cv2
import gymnasium as gym
import numpy as np
import torch

from cfg.CFG import ROOT_DIR, get_scene_usd_paths, set_map_names
from tasks.task_utils import get_env_config


def save_grid(rgb, apt, maps, path, step):
    n = rgb.shape[0]
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    tiles = []
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
    cfg = get_env_config("go2_vision_multi_full")
    cfg.curriculum = None
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    gym.register(
        id="Isaac-vision-task-test-v0",
        entry_point="navigation_env.NavigationEnv:NavEnv",
        disable_env_checker=True,
        kwargs={"scene_path": get_scene_usd_paths(), "use_long_horizon": False, "sample_voronoi_probability": 0.25},
    )
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(gym.make("Isaac-vision-task-test-v0", cfg=cfg))
    base = env.unwrapped
    cam = base.scene["camera"]
    print(f"\nfront camera: {cam.num_instances} instances for {args.num_envs} envs, "
          f"has_front_camera = {base.has_front_camera}")

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

    os.makedirs(args.out, exist_ok=True)
    apt = base.env_apt_ids.cpu().tolist()
    obs, _ = env.reset()
    t0, timed = None, 0
    for step in range(args.steps):
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
        if step % args.save_every == 0 or step == args.steps - 1:
            rgb = cam.data.output["rgb"].torch.cpu().numpy()
            if step == 0:
                print(f"rgb {rgb.shape} {rgb.dtype}, mean {rgb[..., :3].mean():.1f}")
            save_grid(rgb, apt, maps, args.out, step)

    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"\n{timed} steps in {dt:.1f} s: {timed / dt:.1f} env steps/s, {timed * args.num_envs / dt:.0f} robot steps/s")
    print(f"images in {args.out}/")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    simulation_app.close()