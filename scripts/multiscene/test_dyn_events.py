"""
Test: door and chair randomisation with one copy of the doors/chairs per robot.

Builds NavEnv with the task go2_lidar_multi_full (K maps, N robots, env i -> apartment i % K) and checks:
  doors
    1. every env drives its own door joints, as many as its apartment has (envs of the same apartment
       have the same count, apartments differ)
    2. after a reset every door has a target angle in [0, 90] degrees
    3. the doors of envs of the same apartment are randomised independently (envs 0 and K differ)
    3b. a second reset gives the doors new angles
    3c. the doors physically rotate: after stepping, door bodies turned compared to the start
  chairs
    4. the chair view holds the chairs of every env, and there is one collision proxy per
       (apartment, chair), not one per chair name
    5. after a reset most chairs moved away from their authored position (if the copies of the other
       envs blocked the collision check, almost none would move)
    6. envs 0 and K (same apartment) get different chair positions
    7. a second reset moves the chairs again (new positions every episode)
    8. after stepping, every chair is above the floor and inside its own apartment

Example:
    env_isaaclab/bin/python scripts/multiscene/test_dyn_events.py --maps 3 40 21 --num_envs 6 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--maps", nargs="+", default=["3", "40", "21"])
parser.add_argument("--num_envs", type=int, default=6)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import numpy as np
import torch
import warp as wp
from pxr import UsdPhysics

from cfg.CFG import get_scene_usd_paths, set_map_names
from navigation_env.NavigationEnv import NavEnv
from tasks.task_utils import get_env_config

results = []


def check(name, ok):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")


def door_angles(env, env_id):
    """Target angle (degrees) of every door joint of one env."""
    return [
        UsdPhysics.DriveAPI(prim, UsdPhysics.Tokens.angular).GetTargetPositionAttr().Get()
        for prim in env._door_prim_paths[env_id]
    ]


def door_body_quats(env):
    """(x, y, z, w) orientation of every tracked door body, read from the lidar's door view."""
    lidar = env.scene["lidar"]
    for view, target in zip(lidar._mesh_views, lidar._raycast_targets_cfg):
        if "door_" in target.prim_expr:
            transforms = view.get_transforms()
            transforms = wp.to_torch(transforms) if isinstance(transforms, wp.array) else transforms
            return transforms.reshape(-1, 7)[:, 3:7].cpu().numpy().copy()
    raise RuntimeError("the lidar has no door target")


def rotation_deg(q1, q2):
    """Angle (degrees) between two arrays of quaternions, row by row."""
    dot = np.abs(np.sum(q1 * q2, axis=1)).clip(0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def chair_poses(env):
    return wp.to_torch(env._chair_view.get_transforms()).view(-1, 7).cpu().numpy()


def env_chair_xy(env, poses, env_id):
    rows = [row for _, row, _ in env._chairs_of_env.get(env_id, [])]
    return poses[rows, :2]


def map_bounds(env):
    """Local (x_min, x_max, y_min, y_max) of every env's apartment map (same as test_nav_env.py)."""
    mm = env._map_manager
    k = mm.env_apt_ids
    res = mm.resolution
    ox, oy = mm.map_origin[k, 0], mm.map_origin[k, 1]
    H, W = mm.map_shape[k, 0].float(), mm.map_shape[k, 1].float()
    return torch.stack([ox, ox + W * res, oy, oy + H * res], dim=1).cpu().numpy()


def main():
    maps = set_map_names(args.maps)
    K, N = len(maps), args.num_envs
    assert N >= 2 * K, "needs at least two envs per apartment"

    cfg = get_env_config("go2_lidar_multi_full")
    cfg.scene.num_envs = N
    cfg.seed = args.seed
    env = NavEnv(cfg, scene_path=get_scene_usd_paths(), use_long_horizon=False, sample_voronoi_probability=0.25)
    apt = env.env_apt_ids.cpu().tolist()
    print(f"maps {maps}, {N} envs, env -> apartment {apt}")

    quats_start = door_body_quats(env)  
    env.reset()

    # 1. Doors
    print("\ndoors")
    counts = [len(env._door_prim_paths.get(i, [])) for i in range(N)]
    for i in range(N):
        roots = {str(p.GetPath()).split("/")[4] for p in env._door_prim_paths.get(i, [])}
        print(f"  env {i} apt {apt[i]} ({maps[apt[i]]}): {counts[i]} door joints under {sorted(roots)}")
    own = all(str(p.GetPath()).startswith(f"/World/envs/env_{i}/dyn/") for i in range(N) for p in env._door_prim_paths[i])
    same = all(counts[i] == counts[j] for i in range(N) for j in range(N) if apt[i] == apt[j])
    check("1. every env drives its own door joints (under /World/envs/env_i/dyn)", own and sum(counts) > 0)
    check("1. envs of the same apartment have the same number of door joints", same)

    angles = [door_angles(env, i) for i in range(N)]
    in_range = all(a is not None and -1e-3 <= a <= 90 + 1e-3 for env_angles in angles for a in env_angles)
    check("2. every door has a target angle in [0, 90] deg after reset", in_range)
    with_doors = [k for k in range(K) if counts[k] > 0]
    k = with_doors[0] if with_doors else 0
    independent = counts[k] > 0 and not np.allclose(angles[k], angles[k + K])
    print(f"  apt {k}: env {k} angles {np.round(angles[k], 1).tolist()}")
    print(f"  apt {k}: env {k + K} angles {np.round(angles[k + K], 1).tolist()}")
    check(f"3. envs {k} and {k + K} (same apartment) have different door angles", independent)

    # 2. Chairs 
    print("\nchairs")
    if getattr(env, "_chair_view", None) is None:
        check("chair randomisation is active (see the 'Chair randomization disabled' warning)", False)
        print(f"\nRESULT FAIL ({sum(results)}/{len(results)} checks)")
        env.close()
        return
    per_env = [len(env._chairs_of_env.get(i, [])) for i in range(N)]
    per_apt = {apt[i]: per_env[i] for i in range(N)}
    print(f"  chairs per env: {per_env}; proxies: {len(env._chair_proxies)} (expected {sum(per_apt.values())})")
    check("4. chair view holds every env's chairs", env._chair_view.count == sum(per_env))
    check("4. one collision proxy per (apartment, chair)", len(env._chair_proxies) == sum(per_apt.values()))

    poses_1 = chair_poses(env)
    moved = np.linalg.norm(poses_1[:, :2] - env._chair_home[:, :2], axis=1) > 0.01
    print(f"  after reset 1: {moved.mean():.0%} of chairs moved from their authored position")
    check("5. most chairs moved after reset (>= 50%)", moved.mean() >= 0.5)
    k = next((k for k in range(K) if per_env[k] > 0), 0)
    differ = per_env[k] > 0 and not np.allclose(env_chair_xy(env, poses_1, k), env_chair_xy(env, poses_1, k + K), atol=0.01)
    check(f"6. envs {k} and {k + K} (same apartment) have different chair positions", differ)

    env.reset()
    angles_2 = [door_angles(env, i) for i in range(N)]
    changed = [not np.allclose(angles[i], angles_2[i]) for i in range(N) if counts[i] > 0]
    print(f"\ndoors after reset 2: {sum(changed)}/{len(changed)} envs got new door angles")
    print(f"  env {k}: {np.round(angles_2[k], 1).tolist()}")
    check("3b. a second reset gives the doors new angles", all(changed))
    poses_2 = chair_poses(env)
    again = np.linalg.norm(poses_2[:, :2] - poses_1[:, :2], axis=1) > 0.01
    print(f"  reset 2: {again.mean():.0%} of chairs got a new position")
    check("7. a second reset moves the chairs again (>= 50%)", again.mean() >= 0.5)

    act_dim = env.action_manager.total_action_dim
    for _ in range(args.steps):
        env.step(torch.zeros((N, act_dim), device=env.device))
    poses_3 = chair_poses(env)
    turned = rotation_deg(quats_start, door_body_quats(env))
    print(f"\n  door bodies after {args.steps} steps: {np.mean(turned > 10):.0%} turned > 10 deg from the start "
          f"(max {turned.max():.0f} deg)")
    check(f"3c. the doors physically rotate (some door bodies turned > 20 deg)", (turned > 20).any())
    bounds = map_bounds(env)
    origins = env.scene.env_origins.cpu().numpy()
    above, inside = True, True
    for i in range(N):
        for _, row, _ in env._chairs_of_env.get(i, []):
            x, y, z = poses_3[row, :3] - origins[i]
            b = bounds[i]
            above &= bool(z > -0.1)
            inside &= bool(b[0] - 0.5 <= x <= b[1] + 0.5 and b[2] - 0.5 <= y <= b[3] + 0.5)
    check(f"8. after {args.steps} steps every chair is above the floor", above)
    check(f"8. after {args.steps} steps every chair is inside its own apartment", inside)

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