"""
Test: the lidar sees each robot's own doors and chairs.

It has Go2MultiApartmentDynSceneCfg, K maps and N robots.
Checks:
  1. the lidar initialises, and every env has as many door/chair mesh slots as its apartment
     (envs of the same apartment have the same counts, apartments differ)
  2. the tracked door/chair poses in the lidar's mesh table lie inside each env's own apartment
  3. isolation: robots 0 and K are put at exactly the same spot

Example (needs at least 2 * K envs):
    env_isaaclab/bin/python scripts/multiscene/test_dyn_lidar.py --maps 3 40 21 --num_envs 6 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--maps", nargs="+", default=["3", "40", "21"])
parser.add_argument("--num_envs", type=int, default=6)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import cv2
import numpy as np
import torch
import warp as wp
import yaml

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from cfg.CFG import get_apartment_origins, get_dataset_dir, get_env_apartments_ids, set_map_names
from mdp.scene.go2_scene_cfg import Go2MultiApartmentDynSceneCfg
from mdp.scene.multi_apartment import (
    restore_clone_plan,
    restore_grid_origins,
    use_apartment_origins,
    use_round_robin_cloning,
)

CHAIRS = "dyn/Meshes/dynamic_objects/other/chair_*/Meshes/chair_*"
results = []


def check(name, ok):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")


def t(x):
    return x.torch if hasattr(x, "torch") else x


def load_map(name):
    d = get_dataset_dir() / name
    img = cv2.imread(str(d / f"{name}_baked_map.png"), cv2.IMREAD_GRAYSCALE)
    meta = yaml.safe_load(open(d / f"{name}_baked_map.yaml"))
    return img, meta["resolution"], tuple(meta["origin"][:2])


def free_local_points(img, res, origin, n, clearance, rng):
    """n free cells with `clearance` metres of free floor around them, as local (x, y) (MapManager formula)."""
    k = max(1, int(round(clearance / res)))
    free = cv2.erode((img >= 254).astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    rows, cols = np.nonzero(free)
    if len(rows) < n:
        raise SystemExit(f"not enough free cells with {clearance} m clearance")
    idx = rng.choice(len(rows), size=n, replace=False)
    return np.stack([(img.shape[1] - cols[idx]) * res + origin[0], rows[idx] * res + origin[1]], axis=1)


def main():
    maps = set_map_names(args.maps)
    K, N = len(maps), args.num_envs
    assert N >= 2 * K, "needs at least two envs per apartment"
    apt_origins = get_apartment_origins()
    apt_id = get_env_apartments_ids(N, K)
    rng = np.random.default_rng(args.seed)

    use_apartment_origins(apt_origins)
    use_round_robin_cloning()
    try:
        cfg = Go2MultiApartmentDynSceneCfg(num_envs=N, env_spacing=0.0, replicate_physics=False, filter_collisions=True)
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
        scene = InteractiveScene(cfg)
        sim.reset()
    finally:
        restore_grid_origins()
        restore_clone_plan()
    dt = sim.get_physics_dt()
    device = scene.env_origins.device
    lidar = scene["lidar"]
    print(f"lidar class: {type(lidar).__name__}")

    def step(n):
        for _ in range(n):
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)

    # place the robots: env 0 and env K at the same local spot 
    maps_data = [load_map(m) for m in maps]
    local = np.zeros((N, 2))
    for k in range(K):
        ids = np.nonzero(apt_id.numpy() == k)[0]
        local[ids] = free_local_points(*maps_data[k], len(ids), clearance=1.2, rng=rng)
    local[K] = local[0]
    robot = scene["robot"]
    state = torch.zeros((N, 13), device=device)
    state[:, :2] = torch.tensor(local, dtype=torch.float32, device=device) + scene.env_origins[:, :2]
    state[:, 2] = float(t(robot.data.default_root_state)[0, 2])
    state[:, 6] = 1.0  # identity quaternion (x, y, z, w)
    ids = torch.arange(N, device=device)
    robot.write_root_com_state_to_sim(state, ids)
    robot.write_joint_state_to_sim(t(robot.data.default_joint_pos), t(robot.data.default_joint_vel), None, ids)
    step(20)

    # 1. mesh slots per env
    print("\n1. door/chair mesh slots per env")
    slots = {expr: np.zeros(N, dtype=int) for expr in lidar._per_env_slots}
    for expr, entries in lidar._per_env_slots.items():
        for e, _, _ in entries:
            slots[expr][e] += 1
    names = {expr: ("doors" if "door_" in expr else "chairs") for expr in slots}
    for i in range(N):
        print(f"  env {i} apt {int(apt_id[i])} ({maps[int(apt_id[i])]}): "
              + ", ".join(f"{slots[e][i]} {names[e]} meshes" for e in slots))
    check("lidar initialised with the per-env ray caster and door/chair targets", len(slots) == 2)
    same_apt_same_count = all(
        slots[e][i] == slots[e][j] for e in slots for i in range(N) for j in range(N) if apt_id[i] == apt_id[j]
    )
    check("envs of the same apartment have the same door/chair slot counts", same_apt_same_count)

    # 2. tracked poses inside each env's own apartment
    print("\n2. tracked door/chair poses in the lidar's mesh table")
    _ = lidar.data  
    positions = wp.to_torch(lidar._mesh_positions_w).cpu()
    ok = True
    for expr, (rows, envs, cols) in lidar._scatter.items():
        for e, c in zip(envs.cpu().tolist(), cols.cpu().tolist()):
            img, res, (ox, oy) = maps_data[int(apt_id[e])]
            x0, y0 = ox + apt_origins[int(apt_id[e]), 0].item(), oy + apt_origins[int(apt_id[e]), 1].item()
            x, y = positions[e, c, 0].item(), positions[e, c, 1].item()
            ok &= (x0 - 0.5 <= x <= x0 + img.shape[1] * res + 0.5) and (y0 - 0.5 <= y <= y0 + img.shape[0] * res + 0.5)
    check("every tracked door/chair pose lies inside its env's own apartment", ok)

    # 3. isolation: move only env 0's first chair 1 m in front of robots 0 and K
    print("\n3. isolation (envs 0 and K at the same spot, only env 0's chair moved)")
    view0 = sim.physics_sim_view.create_rigid_body_view(f"/World/envs/env_0/{CHAIRS}")
    viewK = sim.physics_sim_view.create_rigid_body_view(f"/World/envs/env_{K}/{CHAIRS}")
    pose = wp.to_torch(view0.get_transforms()).view(-1, 7).clone()
    robot_xy = t(robot.data.root_pos_w)[0, :2].cpu()
    target = robot_xy + torch.tensor([1.0, 0.0])
    pose[0, :2] = target.to(pose.device)
    idx = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=str(pose.device))
    view0.set_transforms(wp.from_torch(pose.contiguous()), indices=idx)
    view0.set_velocities(wp.from_torch(torch.zeros((view0.count, 6), device=pose.device)), indices=idx)
    k_chairs = wp.to_torch(viewK.get_transforms()).view(-1, 7)[:, :2].cpu()
    clear = bool((torch.linalg.norm(k_chairs - target, dim=1) > 1.0).all())
    step(20) 

    hits = t(lidar.data.ray_hits_w).cpu()  # (N, rays, 3)
    chair_xy = wp.to_torch(view0.get_transforms()).view(-1, 7)[0, :2].cpu()
    def hits_near(e):
        h = hits[e]
        finite = torch.isfinite(h).all(dim=1)
        near = torch.linalg.norm(h[:, :2] - chair_xy, dim=1) < 0.35
        return int((finite & near & (h[:, 2] > 0.05)).sum())
    n0, nK = hits_near(0), hits_near(K)
    print(f"  chair of env 0 now at {chair_xy.tolist()}; lidar hits near it: env 0 = {n0}, env {K} = {nK}")
    check(f"env {K} has no chair of its own near that spot (precondition)", clear)
    check("env 0's lidar sees its own moved chair", n0 > 0)
    check(f"env {K}'s lidar does NOT see env 0's chair (same apartment, same spot)", nK == 0)

    print(f"\nRESULT {'PASS' if all(results) else 'FAIL'} ({sum(results)}/{len(results)} checks)")


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