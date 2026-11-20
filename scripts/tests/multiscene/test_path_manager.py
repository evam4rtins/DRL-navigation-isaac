"""
Test of the batched PathManager on the real maps.

1. K = 1: with only the baseline's map and the same seed, the new MapManager + PathManager give
   exactly the baseline saved: Voronoi nodes, sampled tasks, teleport poses, A* paths and path lengths.
2. K maps: for every env
   - start and goal come from its apartment,
   - goal_pos_local is the goal converted with its apartment's map,
   - the A* path is exactly the one its apartment's single-map manager computes for the same start/goal,
   - every path point lies on a free cell of its own map,
   - the teleport pose = local start + env origin.
Example:
    env_isaaclab/bin/python scripts/multiscene/test_path_manager.py --maps 3 40 21
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--maps", nargs="+", default=["3", "40", "21"])
parser.add_argument("--num_envs", type=int, default=12)
parser.add_argument("--baseline", type=str, default="logs/multiscene_baseline/baseline_kujiale_0040_seed0.pt")
try:
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = AppLauncher(args).app
except ImportError: 
    args = parser.parse_args()
    simulation_app = None

import random

import numpy as np
import torch

from cfg.CFG import (
    ROOT_DIR,
    get_apartment_origins,
    get_env_apartments_ids,
    get_scene_usd_paths,
    set_map_name,
    get_scene_usd_path,
    set_map_names,
)
from navigation_env.managers.map_manager import MapManager
from navigation_env.managers.path_manager import PathManager

results = []

def check(name, ok):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")

def map_files(scene_path):
    return str(scene_path.parent / (scene_path.stem + "_map.png")), str(scene_path.parent / (scene_path.stem + "_map.yaml"))

def seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def main():
    # K = 1
    base = torch.load(ROOT_DIR / args.baseline)
    print(f"K = 1 vs baseline ({base['map']}, seed {base['seed']})")
    set_map_name(base["map"])
    seed(base["seed"]) 
    N = base["num_envs"]
    scene_path = get_scene_usd_path()
    mm1 = MapManager(*map_files(scene_path), N, "cpu")
    pm1 = PathManager(scene_path=scene_path, map_mgr=mm1, num_envs=N, device="cpu")
    ids, eo = torch.arange(N), torch.zeros((N, 3))
    pm1.sample_nav_task(ids, sample_voronoi_prob=0.25)
    tel = pm1.compute_robot_teleport_state(ids, eo, robot_z_pos=0.4)
    pm1.compute_global_plan(ids, tel[:, :3].clone(), eo)
    check("Voronoi nodes", torch.equal(pm1.nodes, base["voronoi_nodes"]))
    check("tasks (start/goal map, goal local, yaws)", all(torch.equal(getattr(pm1, k), base[k]) for k in
          ["start_pos_map", "goal_pos_map", "goal_pos_local", "start_yaw", "goal_yaw"]))
    check("teleport poses", torch.equal(tel, base["teleport_state"]))
    check("A* paths and lengths", torch.equal(pm1.path_buf, base["path_buf"]) and torch.equal(pm1.path_len, base["path_len"])
          and torch.equal(pm1.initial_path_length, base["initial_path_length"]))

    # K maps
    maps = set_map_names(args.maps)
    K, N = len(maps), args.num_envs
    apt = get_env_apartments_ids(N, K)
    scene_paths = get_scene_usd_paths()
    files = [map_files(p) for p in scene_paths]
    print(f"\nK = {K} maps {maps}, {N} envs, env -> apartment {apt.tolist()}")
    seed(1)
    mm = MapManager([f[0] for f in files], [f[1] for f in files], N, "cpu", env_apt_ids=apt)
    pm = PathManager(scene_path=scene_paths, map_mgr=mm, num_envs=N, device="cpu")
    singles = []
    for p, f in zip(scene_paths, files):
        m1 = MapManager(*f, 1, "cpu")
        singles.append((m1, PathManager(scene_path=p, map_mgr=m1, num_envs=1, device="cpu")))

    env_origins = get_apartment_origins()[apt]  
    ids = torch.arange(N)
    ok_task, ok_goal_local, ok_path, ok_free, ok_tel = True, True, True, True, True
    for rep in range(5):  
        pm.sample_nav_task(ids, sample_voronoi_prob=0.5)
        tel = pm.compute_robot_teleport_state(ids, env_origins, robot_z_pos=0.4)
        pm.compute_global_plan(ids, tel[:, :3].clone(), env_origins)
        for i in range(N):
            k = int(apt[i])
            m1, p1 = singles[k]
            manual = [(pr["start"]["y"], pr["start"]["x"], pr["goal"]["y"], pr["goal"]["x"]) for pr in p1.manual_tasks["pairs"]]
            s, g = pm.start_pos_map[i].tolist(), pm.goal_pos_map[i].tolist()
            nodes = {tuple(n) for n in p1.nodes.tolist()}
            ok_task &= (s[0], s[1], g[0], g[1]) in manual or (tuple(s) in nodes and tuple(g) in nodes)
            ok_goal_local &= torch.equal(pm.goal_pos_local[i], m1.map_to_local_coords(pm.goal_pos_map[i:i + 1])[0])
            p1.start_pos_map[0], p1.goal_pos_map[0] = pm.start_pos_map[i], pm.goal_pos_map[i]
            p1.goal_pos_local[0] = pm.goal_pos_local[i]
            p1.initial_path_length[0] = p1.current_path_length[0] = p1.point_dist
            p1.path_tensors[0] = None
            local_start = tel[i:i + 1, :3] - env_origins[i:i + 1]
            p1.compute_global_plan(torch.arange(1), local_start, torch.zeros((1, 3)))
            n = int(pm.path_len[i])
            ok_path &= n == int(p1.path_len[0]) and torch.equal(pm.path_buf[i, :n], p1.path_buf[0, :n])
            rc = m1.local_to_map_coords(pm.path_buf[i, : max(n - 1, 1)])
            ok_free &= not m1.obstacle_masks(torch.arange(1))[0][rc[:, 0], rc[:, 1]].any()
            ok_tel &= torch.allclose(tel[i, :2], m1.map_to_local_coords(pm.start_pos_map[i:i + 1])[0] + env_origins[i, :2])
    check("start/goal of every env come from its own apartment", ok_task)
    check("goal_pos_local converted with its apartment's map", ok_goal_local)
    check("A* path == its apartment's single-map manager (same start/goal)", ok_path)
    check("all path points on free cells of its own map", ok_free)
    check("teleport pose = local start + apartment origin", ok_tel)

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
        if simulation_app is not None:
            simulation_app.close()
