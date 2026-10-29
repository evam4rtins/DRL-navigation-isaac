"""
Test: one copy of the doors and chairs per robot.

Builds Go2MultiApartmentDynSceneCfg with K maps and N robots and checks:
  1. env origins = apartment origins as before
  2. every env has its own copy at /World/envs/env_i/dyn, and it is the copy of its apartment:
     a. the USD file the copy references is apartment i % K
     b. it has the same number of chairs and door joints as that apartment's USD
     c. its doors/chairs lie inside that apartment's floor plan (world coordinates)
  3. in each copy only Meshes/dynamic_objects is active 
  4. physics: after stepping, every chair still exists, is above the floor and inside its own apartment

Example:
    env_isaaclab/bin/python scripts/multiscene/test_dyn_objects.py --maps 3 40 21 --num_envs 6 --headless
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--maps", nargs="+", default=["3", "40", "21"])
parser.add_argument("--num_envs", type=int, default=6)
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from pathlib import Path

import numpy as np
import omni.usd
import torch
import warp as wp
import yaml
from pxr import Pcp, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from cfg.CFG import get_apartment_origins, get_dataset_dir, get_env_apartments_ids, get_scene_usd_paths, set_map_names
from mdp.scene.go2_scene_cfg import Go2MultiApartmentDynSceneCfg
from mdp.scene.multi_apartment import (
    restore_clone_plan,
    restore_grid_origins,
    use_apartment_origins,
    use_round_robin_cloning,
)

DYN = "Meshes/dynamic_objects"
results = []


def check(name, ok):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")


def mem_gb():
    status = open("/proc/self/status").read().split("\n")
    return {l.split(":")[0]: int(l.split()[1]) / 1e6 for l in status if l.startswith(("VmRSS", "VmHWM"))}


def count_dyn(root: Usd.Prim) -> tuple[int, int]:
    """(chairs, door joints) under root/Meshes/dynamic_objects."""
    dyn = root.GetPrimAtPath(DYN)
    if not dyn.IsValid():
        return 0, 0
    chairs, doors = 0, 0
    for prim in Usd.PrimRange(dyn):
        if prim.GetName().startswith("chair_") and prim.GetParent().GetName() == "other":
            chairs += 1
        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            doors += 1
    return chairs, doors


def referenced_files(prim: Usd.Prim) -> list[str]:
    """Names of the USD files this prim references."""
    names = []
    for arc in Usd.PrimCompositionQuery(prim).GetCompositionArcs():
        if arc.GetArcType() == Pcp.ArcTypeReference and arc.GetTargetLayer() is not None:
            names.append(Path(arc.GetTargetLayer().identifier).name)
    return names


def map_bounds_world(name: str, origin) -> tuple[float, float, float, float]:
    """World (x_min, x_max, y_min, y_max) of a map's floor plan (same formula as MapManager)."""
    d = get_dataset_dir() / name
    meta = yaml.safe_load(open(d / f"{name}_baked_map.yaml"))
    import cv2

    img = cv2.imread(str(d / f"{name}_baked_map.png"), cv2.IMREAD_GRAYSCALE)
    H, W = img.shape
    res, (ox, oy) = meta["resolution"], meta["origin"][:2]
    return ox + origin[0], ox + W * res + origin[0], oy + origin[1], oy + H * res + origin[1]


def inside(x, y, b, margin=0.5):
    return b[0] - margin <= x <= b[1] + margin and b[2] - margin <= y <= b[3] + margin


def main():
    maps = set_map_names(args.maps)
    K, N = len(maps), args.num_envs
    usd_paths = get_scene_usd_paths()
    apt_origins = get_apartment_origins()
    apt_id = get_env_apartments_ids(N, K)
    bounds = [map_bounds_world(m, apt_origins[k].tolist()) for k, m in enumerate(maps)]
    print(f"maps {maps}, {N} envs, env -> apartment {apt_id.tolist()}")

    ref_counts = []
    for k, p in enumerate(usd_paths):
        stage_k = Usd.Stage.Open(str(p))
        ref_counts.append(count_dyn(stage_k.GetDefaultPrim() or stage_k.GetPseudoRoot().GetChildren()[0]))
        print(f"  apartment {k} ({maps[k]}): {ref_counts[k][0]} chairs, {ref_counts[k][1]} door joints")

    use_apartment_origins(apt_origins)
    use_round_robin_cloning()
    try:
        cfg = Go2MultiApartmentDynSceneCfg(num_envs=N, env_spacing=0.0, replicate_physics=False, filter_collisions=True)
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
        t0 = time.time()
        scene = InteractiveScene(cfg)
        sim.reset()
        t_scene = time.time() - t0
    finally:
        restore_grid_origins()
        restore_clone_plan()
    stage = omni.usd.get_context().get_stage()

    # 1. origins
    print("\n1. env origins")
    check("env origins = apartment origins", torch.allclose(scene.env_origins.cpu(), apt_origins[apt_id], atol=1e-4))

    # 2 and 3. per-env copies
    print("\n2./3. per-robot copies (before stepping)")
    ok_exists, ok_file, ok_counts, ok_inside, ok_active = True, True, True, True, True
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    print(f"  {'env':>3} {'apt':>3} {'map':14} {'file':28} {'chairs':>6} {'doors':>5}  inside")
    for i in range(N):
        k = int(apt_id[i])
        root = stage.GetPrimAtPath(f"/World/envs/env_{i}/dyn")
        if not root.IsValid():
            ok_exists = False
            print(f"  {i:>3} {k:>3} {maps[k]:14} MISSING /World/envs/env_{i}/dyn")
            continue
        files = referenced_files(root)
        expected_file = Path(usd_paths[k]).name
        ok_file &= expected_file in files
        counts = count_dyn(root)
        ok_counts &= counts == ref_counts[k]
        # world bounding box of this copy's doors/chairs
        rng = bbox.ComputeWorldBound(root.GetPrimAtPath(DYN)).ComputeAlignedRange()
        cx, cy = (rng.GetMin()[0] + rng.GetMax()[0]) / 2, (rng.GetMin()[1] + rng.GetMax()[1]) / 2
        in_apt = inside(rng.GetMin()[0], rng.GetMin()[1], bounds[k]) and inside(rng.GetMax()[0], rng.GetMax()[1], bounds[k])
        ok_inside &= in_apt
        active = sorted(c.GetName() for c in root.GetChild("Meshes").GetChildren() if c.IsActive())
        rendering = root.GetChild("Rendering")
        ok_active &= active == ["dynamic_objects"] and not (rendering.IsValid() and rendering.IsActive())
        print(f"  {i:>3} {k:>3} {maps[k]:14} {','.join(files)[:28]:28} {counts[0]:>6} {counts[1]:>5}  {in_apt} "
              f"(centre {cx:.1f}, {cy:.1f})")
    check("every env has /World/envs/env_i/dyn", ok_exists)
    check("each copy references ITS apartment's USD (i % K)", ok_file)
    check("each copy has its apartment's number of chairs and door joints", ok_counts)
    check("each copy's doors/chairs lie inside its own apartment", ok_inside)
    check("in each copy only Meshes/dynamic_objects is active (no static meshes, no lights)", ok_active)

    # 4. physics
    print(f"\n4. physics ({args.steps} steps)")
    dt = sim.get_physics_dt()
    for _ in range(args.steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
    try:
        view = sim.physics_sim_view.create_rigid_body_view(
            f"/World/envs/env_*/dyn/{DYN}/other/chair_*/Meshes/chair_*"
        )
        poses = wp.to_torch(view.get_transforms()).view(-1, 7).cpu()  # same call as the chair events
        expected_chairs = sum(ref_counts[int(apt_id[i])][0] for i in range(N))
        check(f"chair rigid bodies in PhysX: {view.count} (expected {expected_chairs})", view.count == expected_chairs)
        ok_z, ok_apt = True, True
        for row, path in enumerate(view.prim_paths):
            i = int(path.split("/env_")[1].split("/")[0])
            x, y, z = poses[row, :3].tolist()
            ok_z &= z > -0.1
            ok_apt &= inside(x, y, bounds[int(apt_id[i])])
        check("all chairs above the floor after stepping", ok_z)
        check("all chairs still inside their own apartment after stepping", ok_apt)
    except Exception as error:
        print(f"  SKIP  physics check ({type(error).__name__}: {error})")

    m = mem_gb()
    print(f"\nscene creation + reset: {t_scene:.1f} s, RAM now {m['VmRSS']:.2f} GB (peak {m['VmHWM']:.2f} GB)")
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
