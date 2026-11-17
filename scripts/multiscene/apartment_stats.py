"""
For each apartment:
  - map size (m), free floor area (m^2)
  - narrowness: share of the free floor closer than 0.5 m to a wall
  - number of door joints and chairs 
  - number of hand-made tasks and their mean start-goal distance (straight line, m)
Writes:
  - apartment_stats.csv
  - apartment_layouts.png: the floor plans side by side, titled train / unseen + key numbers
and prints the averages of the training and the unseen apartments.

Run:
    env_isaaclab/bin/python scripts/multiscene/apartment_stats.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--maps", nargs="+", default=None, help="default: training_envs + eval_envs of dataset_cfg.yaml")
parser.add_argument("--out", default="apartment_stats")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app 

import csv
import math

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pxr import Usd

from cfg.CFG import NAVPOINTS_FILE, get_dataset_dir, load_cfg, resolve_map_name

DYN = "Meshes/dynamic_objects"


def count_doors_chairs(usd_path):
    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot().GetChildren()[0]
    dyn = root.GetPrimAtPath(DYN)
    if not dyn.IsValid():
        return 0, 0
    chairs = doors = 0
    for prim in Usd.PrimRange(dyn):
        if prim.GetName().startswith("chair_") and prim.GetParent().GetName() == "other":
            chairs += 1
        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            doors += 1
    return doors, chairs


def map_stats(name):
    d = get_dataset_dir() / name
    img = cv2.imread(str(d / f"{name}_baked_map.png"), cv2.IMREAD_GRAYSCALE)
    meta = yaml.safe_load(open(d / f"{name}_baked_map.yaml"))
    res = float(meta["resolution"])
    free = img >= 254
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) * res
    free_area = free.sum() * res * res
    narrow = float((dist[free] < 0.5).mean()) if free.any() else float("nan")
    return img, res, img.shape[1] * res, img.shape[0] * res, free_area, narrow


def task_stats(name, res, navpoints):
    pairs = navpoints.get(name, {}).get("pairs", [])
    if not pairs:
        return 0, float("nan")
    dists = [math.hypot(p["goal"]["x"] - p["start"]["x"], p["goal"]["y"] - p["start"]["y"]) * res for p in pairs]
    return len(pairs), float(np.mean(dists))


def main():
    cfg = load_cfg()
    train = [resolve_map_name(m) for m in cfg.get("training_envs", [])]
    unseen = [resolve_map_name(m) for m in cfg.get("eval_envs", [])]
    maps = [resolve_map_name(m) for m in args.maps] if args.maps else train + unseen
    navpoints = yaml.safe_load(open(NAVPOINTS_FILE))

    rows, images = [], []
    for name in maps:
        img, res, w, h, area, narrow = map_stats(name)
        doors, chairs = count_doors_chairs(get_dataset_dir() / name / f"{name}_baked.usda")
        n_tasks, task_dist = task_stats(name, res, navpoints)
        split = "unseen" if name in unseen else ("train" if name in train else "other")
        rows.append(dict(map=name, split=split, width_m=round(w, 1), height_m=round(h, 1),
                         free_area_m2=round(area, 1), narrow_share=round(narrow, 3), doors=doors, chairs=chairs,
                         tasks=n_tasks, mean_task_dist_m=round(task_dist, 2)))
        images.append(img)
        print(f"{name:14} {split:6}  {w:5.1f} x {h:5.1f} m  free {area:6.1f} m2  narrow {narrow:5.1%}  "
              f"doors {doors:2}  chairs {chairs:2}  tasks {n_tasks:2}  mean task dist {task_dist:4.1f} m")

    # averages per split
    keys = ["free_area_m2", "narrow_share", "doors", "chairs", "mean_task_dist_m"]
    print("\naverages")
    for split in ("train", "unseen"):
        sel = [r for r in rows if r["split"] == split]
        if sel:
            print(f"  {split:6} ({len(sel):2}): " + "  ".join(f"{k} {np.nanmean([r[k] for r in sel]):.2f}" for k in keys))

    with open(f"{args.out}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cols = 6
    nrows = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(3.2 * cols, 3.4 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, r, img in zip(np.atleast_1d(axes).ravel(), rows, images):
        ax.imshow(img, cmap="gray")
        color = "tab:red" if r["split"] == "unseen" else "black"
        ax.set_title(f"{r['map']} ({'UNSEEN' if r['split'] == 'unseen' else r['split']})\n"
                     f"{r['free_area_m2']} m², doors {r['doors']}, chairs {r['chairs']}", fontsize=8, color=color)
    fig.tight_layout()
    fig.savefig(f"{args.out}_layouts.png", dpi=120)
    print(f"\nwrote {args.out}.csv and {args.out}_layouts.png")


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
