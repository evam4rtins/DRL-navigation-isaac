"""
Test of the batched MapManager on the real maps.

1. K = 1: the new MapManager with only map 40 gives exactly the baseline saved with the
   original code: conversions and obstacle mask.
2. K maps: every apartment behaves exactly like its own single-map MapManager
   (conversions per point and batched, obstacle masks, lidar costmaps with apartment origins).
3. Prints the size of the padded maps and the lidar-costmap memory for a training-size run.

Example: 
    env_isaaclab/bin/python scripts/multiscene/test_map_manager.py --maps 3 40 21
"""

import argparse
import types

import numpy as np
import torch

from cfg.CFG import ROOT_DIR, get_apartment_origins, get_dataset_dir, get_env_apartments_ids, set_map_names
from navigation_env.managers.map_manager import MapManager

parser = argparse.ArgumentParser()
parser.add_argument("--maps", nargs="+", default=["3", "40", "21"])
parser.add_argument("--num_envs", type=int, default=12)
parser.add_argument("--baseline", type=str, default="logs/multiscene_baseline/baseline_kujiale_0040_seed0.pt")
args = parser.parse_args()

results = []

def check(name, ok):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")


def map_files(name):
    d = get_dataset_dir() / name
    return str(d / f"{name}_baked_map.png"), str(d / f"{name}_baked_map.yaml")


class Proxy:
    """Stands in for IsaacLab's warp-backed sensor data (has a .torch view)."""

    def __init__(self, t):
        self.torch = t

# k = 1
print("1. k = 1 vs baseline")
base = torch.load(ROOT_DIR / args.baseline)
png, yml = map_files(base["map"])
mm1 = MapManager(png, yml, base["num_envs"], "cpu") 
check("resolution / origin / shape", mm1.resolution == base["resolution"]
      and list(mm1.origin) == list(base["origin"]) and tuple(mm1.shape) == tuple(base["shape"]))
local = mm1.map_to_local_coords(base["conv_px"])
check("map_to_local_coords == baseline", torch.equal(local, base["conv_local"]))
check("local_to_map_coords == baseline", torch.equal(mm1.local_to_map_coords(local), base["conv_back"]))
mask0 = mm1.obstacle_masks(torch.arange(1))[0]
check("obstacle mask (static) == baseline", np.array_equal(mask0, base["obstacle_mask_env0"].numpy()))

# k maps
maps = set_map_names(args.maps)
K, N = len(maps), args.num_envs
apt = get_env_apartments_ids(N, K)
files = [map_files(m) for m in maps]
print(f"\n2. K = {K} maps {maps}, {N} envs, env -> apartment {apt.tolist()}")
mm = MapManager([f[0] for f in files], [f[1] for f in files], N, "cpu", env_apt_ids=apt)
singles = [MapManager(p, y, N, "cpu") for p, y in files]

for k in range(K):
    H, W = mm.apt_shape(k)
    px = torch.stack(torch.meshgrid(torch.arange(0, H, 3), torch.arange(0, W, 3), indexing="ij"), -1).reshape(-1, 2).float()
    loc_b, loc_s = mm.map_to_local_coords(px, k), singles[k].map_to_local_coords(px)
    back_b = mm.local_to_map_coords(loc_b, torch.full((len(px),), k))
    ok = torch.equal(loc_b, loc_s) and torch.equal(back_b, singles[k].local_to_map_coords(loc_s))
    check(f"apt {k} ({maps[k]}): conversions == single-map manager, round trip "
          f"max error {(back_b.float() - px).abs().max().item():.0f} px", ok)

pts = torch.rand(N, 200, 3) * 24 - 12
out = mm.local_to_map_coords(pts, apt)
check("batched (B, N, 3) local_to_map per env == its apartment's manager",
      all(torch.equal(out[i], singles[int(apt[i])].local_to_map_coords(pts[i])) for i in range(N)))

masks = mm.obstacle_masks(torch.arange(N))
check("obstacle masks: cropped to each env's map and == its apartment's static map",
      all(masks[i].shape == mm.apt_shape(int(apt[i]))
          and np.array_equal(masks[i], singles[int(apt[i])].obstacle_masks(torch.arange(1))[0]) for i in range(N)))

env_origins = get_apartment_origins()[apt]
hits_local = torch.rand(N, 300, 3) * 6 - 3
scan = types.SimpleNamespace(ray_hits_w=Proxy(hits_local + env_origins[:, None]), pos_w=Proxy(env_origins.clone()))
mm.update_local_costmap(scan, env_origins)
ok = True
for i in range(N):
    k = int(apt[i])
    s1 = MapManager(*files[k], 1, "cpu")
    s1.update_local_costmap(types.SimpleNamespace(ray_hits_w=Proxy(hits_local[i:i + 1]), pos_w=Proxy(torch.zeros(1, 3))),
                            torch.zeros(1, 3))
    H, W = mm.apt_shape(k)
    ok &= torch.equal(mm.local_costmaps[i, 0, :H, :W], s1.local_costmaps[0, 0])
    ok &= not torch.isinf(mm.local_costmaps[i, 0, H:, :]).any().item() and not torch.isinf(mm.local_costmaps[i, 0, :, W:]).any().item()
check("lidar costmaps (world hits, apartment origins) == each apartment's own manager, nothing outside the map", ok)
lidar_masks = mm.obstacle_masks(torch.arange(N))
check("obstacle masks now include the lidar hits", all(lidar_masks[i].sum() >= masks[i].sum() for i in range(N)))

# sizes
print(f"\n3. padded map size H_max x W_max = {mm.H_max} x {mm.W_max}")
for n in (64, 256):
    print(f"   lidar costmaps for {n} envs: {n * mm.H_max * mm.W_max * 4 / 1e6:.0f} MB")

print(f"\nRESULT {'PASS' if all(results) else 'FAIL'} ({sum(results)}/{len(results)} checks)")
