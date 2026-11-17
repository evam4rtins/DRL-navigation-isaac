"""
Evaluate a trained policy in the multi-apartment env, with the success rate per apartment.

Examples:
    env_isaaclab/bin/python scripts/multiscene/eval_multi.py --maps 3 \
        --checkpoint ckpts/unitree_go2_nav/old_multi_1maps_policy_<date>.pt --headless
    env_isaaclab/bin/python scripts/multiscene/eval_multi.py --task go2_lidar_multi_full --maps 3 40 21 \
        --checkpoint ckpts/unitree_go2_nav/teacher.pt --tasks_csv eval_tasks.csv --headless
    # videos of the failures in one apartment (1 robot, the camera follows it)
    env_isaaclab/bin/python scripts/multiscene/eval_multi.py --task go2_lidar_multi_full --maps 31 \
        --num_envs 1 --episodes 10 --video --view top --checkpoint ckpts/unitree_go2_nav/teacher.pt --headless
"""

import argparse
 
from isaaclab.app import AppLauncher
 
parser = argparse.ArgumentParser()
parser.add_argument("--maps", nargs="+", required=True)
parser.add_argument("--checkpoint", type=str, required=True, help="path of the .pt policy file")
parser.add_argument("--num_envs", type=int, default=24, help="use a multiple of the number of maps")
parser.add_argument("--episodes", type=int, default=10, help="episodes per env")
parser.add_argument("--task", default="go2_lidar_multi", choices=["go2_lidar_multi", "go2_lidar_multi_full"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--tasks_csv", type=str, default=None, help="write the per-task breakdown to this csv")
parser.add_argument("--video", action="store_true", help="record robot 0, one mp4 per episode")
parser.add_argument("--view", default="follow", choices=["follow", "top"], help="camera for --video")
parser.add_argument("--video_dir", type=str, default="eval_videos")
parser.add_argument("--video_fails", type=int, default=8, help="max videos of failed episodes (timeout/other)")
parser.add_argument("--video_successes", type=int, default=4, help="max videos of successful episodes (arrived)")
parser.add_argument("--min_gap", type=float, default=None,
                    help="override MIN_TRAVERSABLE_GAP (m) of the planner for this evaluation only (default 0.20)")
parser.add_argument("--costmap_on_fail", action="store_true",
                    help="save what A* sees for robot 0 at the end of every failed (timeout) episode")
parser.add_argument("--costmap_every", type=int, default=0,
                    help="also save it every N steps of robot 0 (0 = never)")
parser.add_argument("--costmap_dir", type=str, default="logs/eval/costmaps")
parser.add_argument("--ceiling_z", type=float, default=1.9,
                    help="--view top: hide (rendering only) every mesh of the apartments that starts above this height")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True  
simulation_app = AppLauncher(args).app
 
import csv
import os
 
import cv2
import gymnasium as gym
import numpy as np
import torch
import tqdm
import yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
 
from cfg.CFG import NAVPOINTS_FILE, ROOT_DIR, get_scene_usd_paths, set_map_names
from policy.NavPolicyv2 import load_policy_checkpoint, make_go2_policy_cfg
from tasks.task_utils import get_env_config
 
# arrived: at the goal. progress_5m: goal_reached fired because the robot covered 5 m of its path
# (use_long_horizon=False), not because it was at the goal
OUTCOMES = ["arrived", "progress_5m", "no_path", "timeout", "other"]
 
 
def hide_ceilings(min_z: float) -> int:
    """Make invisible every apartment mesh whose lowest point is above min_z (ceilings, lamps, upper cabinets).
 
    Only the rendering changes: physics colliders and the lidar meshes (already loaded) are not affected.
    """
    import omni.usd
    from pxr import Usd, UsdGeom
 
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    hidden = 0
    for root in stage.GetPrimAtPath("/World").GetChildren():
        if not root.GetName().startswith("apt_"):
            continue
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if not box.IsEmpty() and box.GetMin()[2] > min_z:
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden += 1
    return hidden
 
 
def manual_task_index(maps):
    """(apartment, start row, start col, goal row, goal col) -> index of the hand-made task in nav_points.yaml."""
    navpoints = yaml.safe_load(open(NAVPOINTS_FILE))
    index = {}
    for k, name in enumerate(maps):
        for t, pair in enumerate(navpoints.get(name, {}).get("pairs", [])):
            s, g = pair["start"], pair["goal"]
            index[(k, int(s["y"]), int(s["x"]), int(g["y"]), int(g["x"]))] = t  # map coords are (row=y, col=x)
    return index
 
 
def costmap_snapshot(base, pm, apt):
    """What A* receives for robot 0: obstacles (static map + lidar hits), robot and goal cells, current path."""
    from navigation_env.managers.map_manager import MapManager
 
    mm = base._map_manager
    k = int(apt[0])
    obstacles = mm.obstacle_masks(torch.tensor([0], device=base.device))[0]  # static map | lidar hits, (H, W)
    h, w = obstacles.shape
    static = np.asarray(mm._occupied[k, :h, :w])
    robot_local = base.scene["robot"].data.root_pos_w.torch[0:1, :2] - base.scene.env_origins[0:1, :2]
    robot = mm.local_to_map_coords(robot_local, k).int().cpu().numpy()[0]
    goal = pm.goal_pos_map[0].int().cpu().numpy()
    costmap = MapManager.create_inflated_costmap(obstacles, mm.clearance_px, free_cells=(tuple(robot), tuple(goal)))
    path = pm.path_tensors[0]
    path_rc = None if path is None else mm.local_to_map_coords(path, k).cpu().numpy()
    return dict(static=static, lidar=obstacles & ~static, blocked=np.isinf(costmap) & ~obstacles,
                robot=robot, goal=goal, path=path_rc, clearance_m=mm.clearance_px * mm.resolution)
 
 
def save_costmap(snap, path, title):
    """white = free, black = static map, red = lidar hits, orange = too close to an obstacle (blocked for A*)."""
    import matplotlib
 
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    img = np.ones(snap["static"].shape + (3,))
    img[snap["blocked"]] = (1.0, 0.75, 0.4)
    img[snap["static"]] = (0.0, 0.0, 0.0)
    img[snap["lidar"]] = (0.9, 0.1, 0.1)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    if snap["path"] is not None and len(snap["path"]) > 0:
        ax.plot(snap["path"][:, 1], snap["path"][:, 0], "-", color="tab:blue", linewidth=2)
    ax.plot(snap["robot"][1], snap["robot"][0], "o", color="tab:green", markersize=9)
    ax.plot(snap["goal"][1], snap["goal"][0], "*", color="red", markersize=14)
    ax.set_title(f"{title}\nclearance {snap['clearance_m']:.2f} m (orange = blocked for A*), red = lidar", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
 
 
def dist_to_goal(base, pm):
    """xy distance robot -> goal of every env (same as the goal_reached term)."""
    pos = base.scene["robot"].data.root_com_pos_w.torch[:, :2]
    goal_w = pm.goal_pos_local[:, :2] + base.scene.env_origins[:, :2]
    return torch.linalg.norm(goal_w - pos, dim=-1)
 
 
def task_label(k, start, goal, index):
    key = (k, int(round(start[0])), int(round(start[1])), int(round(goal[0])), int(round(goal[1])))
    t = index.get(key)
    return "random" if t is None else f"task{t:02d}"
 
 
def main():
    if args.min_gap is not None:
        import navigation_env.managers.map_manager as map_manager_module
 
        map_manager_module.MIN_TRAVERSABLE_GAP = args.min_gap  # read by MapManager.clearance_px at every replan
        print(f"planner: MIN_TRAVERSABLE_GAP = {args.min_gap} m")
    maps = set_map_names(args.maps)
    K = len(maps)
    if args.num_envs % K != 0:
        print(f"WARNING: {args.num_envs} envs is not a multiple of {K} maps, apartments get different env counts")
 
    cfg = get_env_config(args.task)
    cfg.curriculum = None
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    gym.register(
        id="Isaac-multi-eval-v0",
        entry_point="navigation_env.EnvDebugWrapper:NavEnvDebugView" if args.video else "navigation_env.NavigationEnv:NavEnv",
        disable_env_checker=True,
        kwargs={"scene_path": get_scene_usd_paths(), "use_long_horizon": False, "sample_voronoi_probability": 0.25,
                "debug_vis": args.video},
    )
    env = RslRlVecEnvWrapper(
        gym.make("Isaac-multi-eval-v0", cfg=cfg, render_mode="rgb_array" if args.video else None)
    )
    base = env.unwrapped
 
    if args.video:
        for sub in ("fails", "successes"):
            os.makedirs(os.path.join(args.video_dir, sub), exist_ok=True)
        if args.view == "top":  
            base.camera_manager.camera_offset = torch.tensor([0.0, 0.0, 7.0], device=base.device)
            base.camera_manager.lookat_offset = torch.tensor([0.3, 0.0, 0.0], device=base.device)
            identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=base.device)  # (x, y, z, w)
 
            def follow_from_above(smoothing: float = 0.3):
                pos = base.scene["robot"].data.root_pos_w.torch[0]
                base.camera_manager.update(pos, identity, smoothing_factor=smoothing)
 
            base.update_follow_camera = follow_from_above
            print(f"hidden {hide_ceilings(args.ceiling_z)} meshes above {args.ceiling_z} m (ceilings) for the top view")
 
    policy_cfg = make_go2_policy_cfg(cfg.obs_groups)
    policy_cfg["num_envs"] = args.num_envs
    runner = OnPolicyRunner(env, policy_cfg, log_dir=None, device=policy_cfg["device"])
    ckpt = args.checkpoint if args.checkpoint.startswith("/") else str(ROOT_DIR / args.checkpoint)
    load_policy_checkpoint(runner, ckpt)
    policy = runner.get_inference_policy(env.device)
 
    apt = base.env_apt_ids
    pm = base.path_manager
    goal_thresh = base.termination_manager.get_term_cfg("goal_reached").params.get("threshold_m", 0.2)
    episodes_done = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    counts = torch.zeros((K, len(OUTCOMES)), dtype=torch.long)
    task_index = manual_task_index(maps)
    task_counts = {}  # (apartment, label) -> one count per outcomes entry
 
    env.unwrapped.seed(args.seed)
    obs, _ = env.reset()
    
    plan_len = pm.current_path_length.clone()
    goal_dist = dist_to_goal(base, pm)
    start = pm.start_pos_map.clone().cpu()
    goal = pm.goal_pos_map.clone().cpu()
 
    if args.costmap_on_fail or args.costmap_every:
        os.makedirs(args.costmap_dir, exist_ok=True)
    snapshot, step0 = None, 0
    frames = []
    saved = {"fails": 0, "successes": 0}
    limit = {"fails": args.video_fails, "successes": args.video_successes}
 
    def videos_left() -> bool:
        return any(saved[k] < limit[k] for k in saved)
    fps = max(1, round(1.0 / base.step_dt))
 
    with tqdm.tqdm(total=env.num_envs * args.episodes, desc="Evaluating") as pbar:
        while not bool((episodes_done >= args.episodes).all()):
            with torch.no_grad():
                policy(obs, stochastic_output=True)
                action = policy.output_mean
            obs, _, dones, _ = env.step(action)
 
            frame = None
            if args.video and videos_left():
                frame = base.render()
 
            done_ids = dones.bool().nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                reached = base.termination_manager.get_term("goal_reached")
                timed_out = base.termination_manager.get_term("timeout")
                no_path = reached & (plan_len < goal_thresh)
                # distance before the last step (+0.3 m for the motion of that step)
                at_goal = reached & ~no_path & (goal_dist < goal_thresh + 0.3)
                progress = reached & ~no_path & ~at_goal
                keep = done_ids[episodes_done[done_ids] < args.episodes]
                for i in keep.tolist():
                    if at_goal[i]:
                        col = 0
                    elif progress[i]:
                        col = 1
                    elif no_path[i]:
                        col = 2
                    else:
                        col = 3 if timed_out[i] else 4
                    k = int(apt[i])
                    counts[k, col] += 1
                    label = task_label(k, start[i].tolist(), goal[i].tolist(), task_index)
                    task_counts.setdefault((k, label), [0] * len(OUTCOMES))[col] += 1
 
                    if args.costmap_on_fail and i == 0 and snapshot is not None and OUTCOMES[col] in ("timeout", "other"):
                        name = f"ep{int(episodes_done[0]):03d}_{maps[k]}_{label}_{OUTCOMES[col]}_end.png"
                        save_costmap(snapshot, os.path.join(args.costmap_dir, name),
                                     f"{maps[k]} {label}: last step before {OUTCOMES[col]}")
                    kind = "fails" if OUTCOMES[col] in ("timeout", "other") else "successes" if col == 0 else None
                    if args.video and i == 0 and frames and kind is not None and saved[kind] < limit[kind]:
                        # (this step's frame is not in `frames`: it already shows the next episode)
                        name = f"ep{int(episodes_done[0]):03d}_{maps[k]}_{label}_{OUTCOMES[col]}.mp4"
                        h, w = frames[0].shape[:2]
                        path = os.path.join(args.video_dir, kind, name)
                        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        for f in frames:
                            writer.write(cv2.cvtColor(f[..., :3], cv2.COLOR_RGB2BGR))
                        writer.release()
                        saved[kind] += 1
                        print(f"  video: {kind}/{name} ({len(frames)} frames)")
                if args.video and bool((done_ids == 0).any()):
                    frames = []  # robot 0 starts a new episode; this step's frame is its first one
                pbar.update(len(keep))
                episodes_done[done_ids] += 1
            if frame is not None:
                frames.append(frame)
            if args.costmap_on_fail or args.costmap_every:
                snapshot = costmap_snapshot(base, pm, apt)  # robot 0, after this step
                step0 = 0 if bool((done_ids == 0).any()) else step0 + 1
                if args.costmap_every and step0 % args.costmap_every == 0 and episodes_done[0] < args.episodes:
                    k0 = int(apt[0])
                    name = f"ep{int(episodes_done[0]):03d}_{maps[k0]}_step{step0:03d}.png"
                    save_costmap(snapshot, os.path.join(args.costmap_dir, name), f"{maps[k0]} step {step0}")
            plan_len = pm.current_path_length.clone()
            goal_dist = dist_to_goal(base, pm)
            start = pm.start_pos_map.clone().cpu()
            goal = pm.goal_pos_map.clone().cpu()
 
    print(f"\ncheckpoint: {args.checkpoint}")
    print(f"maps: {maps}, {args.num_envs} envs, {args.episodes} episodes per env (deterministic, no curriculum)\n")
    # success = goal_reached fired (old definition, comparable with earlier logs); at_goal = really at the goal
    print("success = goal_reached termination (arrived + progress_5m + no_path, as before); "
          "at_goal = arrived / (episodes - no_path)")
    print(f"{'apartment':15} {'episodes':>8} {'success':>8} {'at_goal':>8} "
          + " ".join(f"{o:>11}" for o in OUTCOMES))
    for k, name in enumerate(maps + ["ALL"]):
        c = counts[k] if k < K else counts.sum(dim=0)
        n = int(c.sum())
        succ = (c[0] + c[1] + c[2]).item() / max(n, 1)
        at_goal = c[0].item() / max(n - int(c[2]), 1)
        print(f"{name:15} {n:8d} {succ:8.1%} {at_goal:8.1%} " + " ".join(f"{int(x):11d}" for x in c))
 
    # per task: the hand-made tasks that fail most (arrived / (episodes - no_path)); progress_5m is not arrived
    print("\nhand-made tasks with the lowest arrival rate (at least 3 episodes)")
    rows = []
    for (k, label), c in sorted(task_counts.items()):
        tried = sum(c) - c[2]
        rate = c[0] / tried if tried else float("nan")
        rows.append(dict(apartment=maps[k], task=label, episodes=sum(c), arrived=c[0], progress_5m=c[1],
                         no_path=c[2], timeout=c[3], other=c[4], arrival_rate=round(rate, 3)))
    worst = sorted((r for r in rows if r["task"] != "random" and r["episodes"] >= 3 and r["arrival_rate"] == r["arrival_rate"]),
                   key=lambda r: r["arrival_rate"])[:15]
    for r in worst:
        print(f"  {r['apartment']:14} {r['task']:7} {r['episodes']:3d} ep  arrived {r['arrived']:3d}  "
              f"5m {r['progress_5m']:3d}  timeout {r['timeout']:3d}  no_path {r['no_path']:3d}  "
              f"arrival {r['arrival_rate']:.0%}")
    if args.tasks_csv:
        with open(args.tasks_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nper-task breakdown written to {args.tasks_csv}")
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
 