"""
Geometry of the hand-made tasks: are the ones the teacher never solves impossible / badly placed, or just hard?

For every hand-made task of nav_points.yaml (training apartments in the eval csv):
  - A* path on the static floor plan (no doors/chairs), with the robot needing at least --min_clear m of room
  - path length (m), narrowest point along the path (m to the nearest non-free cell),
    room around the start and around the goal (m)
Run:
    env_isaaclab/bin/python scripts/multiscene/inspect_tasks.py --tasks_csv logs/eval/eval_tasks_2100.csv
"""

import argparse
import csv
import math

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyastar2d
import yaml

from cfg.CFG import NAVPOINTS_FILE, get_dataset_dir

parser = argparse.ArgumentParser()
parser.add_argument("--tasks_csv", required=True, help="per-task csv written by eval_multi.py --tasks_csv")
parser.add_argument("--min_clear", type=float, default=0.15, help="room (m) the robot needs around its centre")
parser.add_argument("--out", default="logs/stats/task_geometry")
args = parser.parse_args()


def load_map(name):
    d = get_dataset_dir() / name
    img = cv2.imread(str(d / f"{name}_baked_map.png"), cv2.IMREAD_GRAYSCALE)
    res = float(yaml.safe_load(open(d / f"{name}_baked_map.yaml"))["resolution"])
    free = img >= 254
    clearance = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) * res  # m to the nearest non-free cell
    return img, res, clearance


def plan(clearance, res, start, goal, min_clear):
    weights = np.where(clearance >= min_clear, 1.0, np.inf).astype(np.float32)
    weights[start] = weights[goal] = 1.0  # judge the endpoints separately (room_start / room_goal)
    path = pyastar2d.astar_path(weights, start, goal, allow_diagonal=True)
    if path is None:
        return None, float("nan"), float("nan")
    steps = np.diff(path, axis=0)
    length = float(np.sum(np.hypot(steps[:, 0], steps[:, 1]))) * res
    inner = path[3:-3] if len(path) > 6 else path
    narrowest = float(clearance[inner[:, 0], inner[:, 1]].min())
    return path, length, narrowest


def main():
    rows = [r for r in csv.DictReader(open(args.tasks_csv)) if r["task"] != "random"]
    def ok(r):  # success as in training: at the goal or 5 m of progress (older csv files have no progress_5m)
        return int(r["arrived"]) + int(r.get("progress_5m") or 0)

    def tried(r):  # episodes in which the planner found a path
        return int(r["episodes"]) - int(r["no_path"])

    dead = {(r["apartment"], r["task"]) for r in rows if tried(r) >= 3 and ok(r) == 0}
    rate = {(r["apartment"], r["task"]): round(ok(r) / tried(r), 3) if tried(r) else "" for r in rows}
    navpoints = yaml.safe_load(open(NAVPOINTS_FILE))

    out_rows, drawings = [], []
    for name in sorted({r["apartment"] for r in rows}):
        img, res, clearance = load_map(name)
        for t, pair in enumerate(navpoints.get(name, {}).get("pairs", [])):
            label = f"task{t:02d}"
            s, g = pair["start"], pair["goal"]
            start, goal = (int(s["y"]), int(s["x"])), (int(g["y"]), int(g["x"]))  # (row, col)
            path, length, narrowest = plan(clearance, res, start, goal, args.min_clear)
            straight = math.hypot(goal[0] - start[0], goal[1] - start[1]) * res
            is_dead = (name, label) in dead
            out_rows.append(dict(apartment=name, task=label, dead=is_dead, arrival_rate=rate.get((name, label), ""),
                                 path_found=path is not None, path_m=round(length, 2), straight_m=round(straight, 2),
                                 narrowest_m=round(narrowest, 3), room_start_m=round(float(clearance[start]), 3),
                                 room_goal_m=round(float(clearance[goal]), 3)))
            if is_dead:
                drawings.append((name, label, img, start, goal, path, out_rows[-1]))

    with open(f"{args.out}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    # dead vs other tasks
    def summary(sel, key):
        vals = [r[key] for r in sel if isinstance(r[key], float) and not math.isnan(r[key])]
        return f"{np.median(vals):6.2f}" if vals else "   n/a"

    groups = {"dead": [r for r in out_rows if r["dead"]], "other": [r for r in out_rows if not r["dead"]]}
    print(f"{'':6} {'tasks':>5} {'no path':>8}   median: {'path m':>7} {'straight m':>10} {'narrowest m':>11} "
          f"{'room start':>10} {'room goal':>9}")
    for g, sel in groups.items():
        no_path = sum(not r["path_found"] for r in sel)
        print(f"{g:6} {len(sel):5d} {no_path:8d}           {summary(sel, 'path_m'):>7} {summary(sel, 'straight_m'):>10} "
              f"{summary(sel, 'narrowest_m'):>11} {summary(sel, 'room_start_m'):>10} {summary(sel, 'room_goal_m'):>9}")

    print("\ndead tasks")
    for r in groups["dead"]:
        flags = []
        if not r["path_found"]:
            flags.append(f"NO PATH with {args.min_clear} m of room")
        if r["room_goal_m"] < args.min_clear:
            flags.append("goal too close to an obstacle")
        if r["room_start_m"] < args.min_clear:
            flags.append("start too close to an obstacle")
        if r["path_found"] and r["narrowest_m"] < 0.25:
            flags.append("narrow passage")
        print(f"  {r['apartment']:14} {r['task']}  path {r['path_m']:5.1f} m  narrowest {r['narrowest_m']:.2f} m  "
              f"room start {r['room_start_m']:.2f} / goal {r['room_goal_m']:.2f} m  {'; '.join(flags)}")

    # drawings of the dead tasks
    cols = 5
    nrows = max(1, math.ceil(len(drawings) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(3.6 * cols, 3.8 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (name, label, img, start, goal, path, r) in zip(np.atleast_1d(axes).ravel(), drawings):
        ax.imshow(img, cmap="gray")
        if path is not None:
            ax.plot(path[:, 1], path[:, 0], "-", color="tab:blue", linewidth=1.5)
        ax.plot(start[1], start[0], "o", color="tab:green", markersize=7)
        ax.plot(goal[1], goal[0], "*", color="tab:red", markersize=11)
        ax.set_title(f"{name} {label}\npath {r['path_m']} m, narrowest {r['narrowest_m']} m", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}_dead_tasks.png", dpi=120)
    print(f"\nwrote {args.out}.csv and {args.out}_dead_tasks.png")


if __name__ == "__main__":
    main()