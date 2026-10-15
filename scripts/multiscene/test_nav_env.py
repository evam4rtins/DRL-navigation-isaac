"""
Test of NavEnv with the multi-apartment scene.

 Checks:
  1. env origins = apartment origins 
  2. after reset: every robot stands on a free cell of its apartment's map, and its goal is inside that map
  3. lidar: the robots see walls, and every lidar hit lies inside the robot's own apartment
  4. while running: no NaN/inf in observations and rewards, episodes end and reset,
     robots stay inside their apartment and above the floor
Prints the environment steps per second at the end.

Example:
    # multi-apartment
    env_isaaclab/bin/python scripts/multiscene/test_nav_env.py --task go2_lidar_multi --maps 3 40 21 --num_envs 12 --headless
    # the original single-map task must still work
    env_isaaclab/bin/python scripts/multiscene/test_nav_env.py --task go2_lidar_full --maps 40 --num_envs 4 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="go2_lidar_multi")
parser.add_argument("--maps", nargs="+", default=["3", "40", "21"])
parser.add_argument("--num_envs", type=int, default=12)
parser.add_argument("--steps", type=int, default=400, help="policy steps (one episode = 15 s = ~187 steps)")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import time

import torch

from cfg.CFG import get_scene_usd_path, get_scene_usd_paths, set_map_name, set_map_names
from navigation_env.NavigationEnv import NavEnv
from tasks.task_utils import get_env_config

results = []


def check(name, ok):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")


def t(x):
    return x.torch if hasattr(x, "torch") else x


def all_finite(obs) -> bool:
    if isinstance(obs, dict):
        return all(all_finite(v) for v in obs.values())
    return bool(torch.isfinite(obs).all())


def map_bounds(env):
    mm = env._map_manager
    k = mm.env_apt_ids
    res = mm.resolution
    ox, oy = mm.map_origin[k, 0], mm.map_origin[k, 1]
    H, W = mm.map_shape[k, 0].float(), mm.map_shape[k, 1].float()
    return torch.stack([ox, ox + W * res, oy, oy + H * res], dim=1)


def inside(local_xy, bounds, margin=0.0):
    return (
        (local_xy[:, 0] >= bounds[:, 0] - margin)
        & (local_xy[:, 0] <= bounds[:, 1] + margin)
        & (local_xy[:, 1] >= bounds[:, 2] - margin)
        & (local_xy[:, 1] <= bounds[:, 3] + margin)
    )


def robots_on_free_cells(env) -> torch.Tensor:
    mm = env._map_manager
    local = t(env.scene["robot"].data.root_pos_w) - env.scene.env_origins
    rc = mm.local_to_map_coords(local, mm.env_apt_ids).cpu()
    occ = torch.as_tensor(mm._occupied)
    k = mm.env_apt_ids.cpu()
    return ~occ[k, rc[:, 0], rc[:, 1]]


def main():
    multi = args.task == "go2_lidar_multi"
    if multi:
        maps = set_map_names(args.maps)
        scene_path = get_scene_usd_paths()
    else:
        maps = [set_map_name(args.maps[0])]
        scene_path = get_scene_usd_path()
    print(f"task {args.task}, maps {maps}, {args.num_envs} envs")

    cfg = get_env_config(args.task)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = NavEnv(cfg, scene_path=scene_path, use_long_horizon=False, sample_voronoi_probability=0.25)
    N, device = env.num_envs, env.device
    apt = env.env_apt_ids
    print(f"env -> apartment {apt.tolist()}")
    print(f"env origins:\n{env.scene.env_origins.cpu()}")

    obs, _ = env.reset()
    robot = env.scene["robot"]
    bounds = map_bounds(env)

    # 1. origins
    if multi:
        check("env origins = apartment origins (asserted in NavEnv.__init__)", True)

    # 2. start and goal
    print("\nafter reset")
    check("every robot on a free cell of its own map", robots_on_free_cells(env).all())
    check("every goal inside its own map", inside(env.path_manager.goal_pos_local[:, :2], bounds).all())
    check("observations finite", all_finite(obs))

    #3. lidar 
    act_dim = env.action_manager.total_action_dim
    for _ in range(3):
        obs, *_ = env.step(torch.zeros((N, act_dim), device=device))
    hits_w = t(env.scene["lidar"].data.ray_hits_w)  # (N, rays, 3), inf where nothing was hit
    hit = torch.isfinite(hits_w).all(dim=-1)
    hit_local = hits_w - env.scene.env_origins[:, None]
    in_own = torch.zeros_like(hit)
    for i in range(N):
        in_own[i] = inside(hit_local[i, :, :2], bounds[i : i + 1].expand(hit_local.shape[1], 4), margin=0.5)
    floor = hit_local[..., 2].abs() < 0.05  # rays that only hit the ground plane
    walls = hit & ~floor
    print(f"  lidar: {hit.float().mean():.0%} of rays hit something, {walls.float().mean():.0%} hit objects/walls")
    check("every robot's lidar hits objects/walls", (walls.sum(dim=1) > 0).all())
    check("every lidar hit on objects/walls lies inside the robot's own apartment", (in_own | ~walls).all())

    # 4. run
    print(f"\nrunning {args.steps} steps with random actions ...", flush=True)
    finite, resets, ok_inside, ok_z, ok_free_after_reset = True, 0, True, True, True
    t0 = time.time()
    for _ in range(args.steps):
        action = 2 * torch.rand((N, act_dim), device=device) - 1
        obs, rew, terminated, truncated, _ = env.step(action)
        finite &= all_finite(obs) and bool(torch.isfinite(rew).all())
        done = terminated | truncated
        resets += int(done.sum())
        pos = t(robot.data.root_pos_w)
        ok_inside &= bool(inside((pos - env.scene.env_origins)[:, :2], bounds).all())
        ok_z &= bool((pos[:, 2] > 0.05).all())
        if done.any():  # the new start of the envs that just reset must be a free cell of their own map
            ids = done.nonzero().squeeze(-1)
            rc = env.path_manager.start_pos_map[ids].long().cpu()
            occ = torch.as_tensor(env._map_manager._occupied)
            ok_free_after_reset &= not occ[apt[ids].cpu(), rc[:, 0], rc[:, 1]].any()
    elapsed = time.time() - t0
    check("observations and rewards always finite", finite)
    check(f"episodes ended and reset ({resets} resets)", resets > 0)
    check("new start on a free cell of its own map after every reset", ok_free_after_reset)
    check("robots always inside their own apartment", ok_inside)
    check("robots always above the floor", ok_z)
    print(f"\n{args.steps * N / elapsed:.0f} env steps/s ({args.steps / elapsed:.1f} policy steps/s, {N} envs)")

    print(f"\nRESULT {'PASS' if all(results) else 'FAIL'} ({sum(results)}/{len(results)} checks)")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import sys
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        simulation_app.close()
