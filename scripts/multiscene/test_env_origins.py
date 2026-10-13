""" 
Test: global apartments + env origins at the apartments.

Checks:
1. scene.env_origins[i] == apartment origin of env i   (the grid_transforms replacement works)
2. a robot teleported to local free cells (local + env_origins) ends up inside its apartment and did not fall through the floor (z > 0; robots without a locomotion policy may tip over)
3. a plot of the floor plans at their world positions with the robots on top, and optionally a rendered top-down image (--image)

"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--maps", nargs="+", default=["3", "40"])
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--image", action="store_true", help="also render a top-down image (needs --enable_cameras)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_ceiling", action="store_true", help="hide the ceilings in the top-down image")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import CameraCfg

from cfg.CFG import ROOT_DIR, get_apartment_origins, get_dataset_dir, get_env_apartments_ids, set_map_names
from mdp.scene.go2_scene_cfg import Go2MultiApartmentSceneCfg
from mdp.scene.multi_apartment import use_apartment_origins

OUT = ROOT_DIR/ "logs" / "multiscene_shared_chairs_doors"
OUT.mkdir(parents=True, exist_ok=True)

def t(x):
    return x.torch if hasattr(x, "torch") else x

def load_map(name):
    d = get_dataset_dir() / name
    img = cv2.imread(str(d / f"{name}_baked_map.png"), cv2.IMREAD_GRAYSCALE)
    meta = yaml.safe_load(open(d / f"{name}_baked_map.yaml" ))
    return img, meta["resolution"], tuple(meta["origin"][:2])

def free_local_points(img, res, origin, n, rng):
    k = max(1, int(round(0.35 / res)))
    free = cv2.erode((img >= 254).astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    rows, cols = np.nonzero(free)
    idx = rng.choice(len(rows), size=n, replace=False)
    x = (img.shape[1] - cols[idx]) * res + origin[0]
    y = rows[idx] * res + origin[1]
    return np.stack([x, y], axis=1)

def main():
    maps = set_map_names(args.maps)
    K, N = len(maps), args.num_envs
    apartment_origins = get_apartment_origins()
    apt_id = get_env_apartments_ids(N,K)
    print(f"maps: {maps}")
    print(f"apartment origins:\n{apartment_origins}")
    print(f"env -> apartment: {apt_id.tolist()}")

    use_apartment_origins(apartment_origins)

    cfg = Go2MultiApartmentSceneCfg(num_envs=N, env_spacing=0.0, replicate_physics=False, filter_collisions=True)
    cfg.light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=1000.0))
    if args.image:
        # one top-down camera per apartment, centred on its floor plan, ~25 m above it
        for k in range(K):
            img, res, (ox, oy) = load_map(maps[k])
            H, W = img.shape
            cx = apartment_origins[k, 0].item() + ox + W * res / 2   # floor-plan centre in world x
            cy = apartment_origins[k, 1].item() + oy + H * res / 2   # floor-plan centre in world y
            setattr(cfg, f"topcam_{k}", CameraCfg(
                prim_path=f"/World/TopCam_{k}",
                width=1600, height=1600, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955,
                                                    clipping_range=(1.0, 100.0)),
                offset=CameraCfg.OffsetCfg(pos=(cx, cy, 25.0),
                                            rot=(0.0, 0.7071068, 0.0, 0.7071068), convention="world"),
            ))

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    scene = InteractiveScene(cfg)

    # hide the ceilings so the top-down image shows the inside (visualisation only, never in training)
    if args.no_ceiling:
        for k in range(K):
            ceiling = sim.stage.GetPrimAtPath(f"/World/apt_{k}/Meshes/static_objects/ceiling")
            if ceiling.IsValid():
                ceiling.SetActive(False)

    sim.reset()

    # check 1: env origins
    env_origins = scene.env_origins.cpu()
    expected = apartment_origins[apt_id]
    ok_origins = torch.allclose(env_origins, expected, atol=1e-4)
    print(f"\nscene.env_origins:\n{env_origins}")
    print(f"CHECK 1 env_origins == apartment origin of each env: {'PASS' if ok_origins else 'FAIL'}")

    # teleport every robot to a free cell of its own apartment: world = local + env_origins
    rng = np.random.default_rng(args.seed)
    maps_data = [load_map(m) for m in maps]
    local_xy = np.zeros((N,2))
    for k in range(K):
        ids = np.nonzero(apt_id.numpy() == k)[0]
        if len(ids):
            local_xy[ids] = free_local_points(*maps_data[k], len(ids), rng)

    print("teleporting robots ...", flush=True)
    robot = scene["robot"]
    device = scene.env_origins.device
    z0 = float(t(robot.data.default_root_state)[0,2])
    state = torch.zeros((N,13), device=device)
    state[:,:2] = torch.tensor(local_xy, dtype=torch.float32, device=device) + scene.env_origins[:,:2]
    state[:,2] = z0
    state[:,6] = 1.0
    env_ids = torch.arange(N, device=device)
    robot.write_root_com_state_to_sim(state, env_ids)
    robot.write_joint_state_to_sim(t(robot.data.default_joint_pos), t(robot.data.default_joint_vel), None, env_ids)

    print(f"stepping {args.steps} steps ...", flush=True)
    dt = sim.get_physics_dt()
    for _ in range(args.steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

    # check 2: every robot inside its own apartment (not through the floor)
    pos_w = t(robot.data.root_pos_w).cpu()
    local = pos_w[:,:2] - env_origins[:,:2]
    ok_inside = True
    print(f"\n{'env':>3} {'apt':>3} {'map':14} {'world x':>8} {'world y':>8} {'local x':>8} {'local y':>8} {'z':>5}  inside")
    for i in range(N):
        k = int(apt_id[i])
        img, res, (ox, oy) = maps_data[k]
        H, W = img.shape
        inside = (ox <= local[i, 0] <= ox + W * res) and (oy <= local[i,1] <= oy + H * res) and pos_w[i,2] > 0.0
        ok_inside &= bool(inside)
        print(f"{i:>3} {k:>3} {maps[k]:14} {pos_w[i,0]:8.2f} {pos_w[i,1]:8.2f} {local[i,0]:8.2f} {local[i,1]:8.2f} {pos_w[i,2]:5.2f}  {inside}")
    print(f"CHECK 2 robots inside their apartment (not through the floor): {'PASS' if ok_inside else 'FAIL'}")

    # plot floor plans at theirs world positions and the robots
    fig, ax = plt.subplots(figsize=(6 + 3 * K, 6))
    for k in range(K):
        img, res, (ox, oy) = maps_data[k]
        H, W = img.shape
        x0, y0 = ox + apartment_origins[k, 0].item(), oy + apartment_origins[k, 1].item()
        ax.imshow(img[:, ::-1], cmap="gray", origin="lower", extent=(x0, x0 + W * res, y0, y0 + H * res))
        ax.text(x0, y0 + H * res + 0.5, f"apt_{k}: {maps[k]}", fontsize=9)
    colors = plt.cm.tab10(apt_id.numpy() % 10)
    ax.scatter(pos_w[:, 0], pos_w[:, 1], c=colors, s=60, edgecolors="k", zorder=3)
    for i in range(N):
        ax.annotate(f"env_{i}", (pos_w[i, 0], pos_w[i, 1]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_aspect("equal"), ax.set_xlabel("world x [m]"), ax.set_ylabel("world y [m]")
    ax.set_title("Floor plans at their world position, robots after the steps")
    fig.tight_layout()
    fig.savefig(OUT / "layout.png", dpi=150)
    print(f"\nsaved {OUT / 'layout.png'}")

    if args.image:
        for _ in range(5):  
            sim.step()
            scene.update(dt)
        for k in range(K):
            rgb = t(scene[f"topcam_{k}"].data.output["rgb"])[0].cpu().numpy()[..., :3]
            f = OUT / f"topdown_apt{k}.png"
            cv2.imwrite(str(f), cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
            print(f"saved {f}")

    print(f"\nRESULT {'PASS' if ok_origins and ok_inside else 'FAIL'}")

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