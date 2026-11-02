"""
Evaluate a trained policy in the multi-apartment env, with the success rate per apartment.

Used to tell an env bug from a training problem: take a policy trained on on apartment and run it
  (a) in a scene with only that apartment         -> its reference success rate
  (b) in the 3-apartment scene                    -> its apartment should give about the same number
If (b) is much worse than (a) for its own apartment, the env is wrong when K > 1.
If (b) is about equal to (a), the env is fine and the multi-apartment failure is a training problem.

Example :
    env_isaaclab/bin/python scripts/multiscene/eval_multi.py --maps 3 \
        --checkpoint ckpts/unitree_go2_nav/old_multi_1maps_policy_<date>.pt --headless
    env_isaaclab/bin/python scripts/multiscene/eval_multi.py --maps 3 40 21 \
        --checkpoint ckpts/unitree_go2_nav/old_multi_1maps_policy_<date>.pt --headless
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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
import tqdm
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from cfg.CFG import ROOT_DIR, get_scene_usd_paths, set_map_names
from policy.NavPolicyv2 import load_policy_checkpoint, make_go2_policy_cfg
from tasks.task_utils import get_env_config


def main():
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
        entry_point="navigation_env.NavigationEnv:NavEnv",
        disable_env_checker=True,
        kwargs={"scene_path": get_scene_usd_paths(), "use_long_horizon": False, "sample_voronoi_probability": 0.25},
    )
    env = RslRlVecEnvWrapper(gym.make("Isaac-multi-eval-v0", cfg=cfg))
    base = env.unwrapped

    policy_cfg = make_go2_policy_cfg(cfg.obs_groups)
    policy_cfg["num_envs"] = args.num_envs
    runner = OnPolicyRunner(env, policy_cfg, log_dir=None, device=policy_cfg["device"])
    ckpt = args.checkpoint if args.checkpoint.startswith("/") else str(ROOT_DIR / args.checkpoint)
    load_policy_checkpoint(runner, ckpt)
    policy = runner.get_inference_policy(env.device)

    apt = base.env_apt_ids
    goal_thresh = base.termination_manager.get_term_cfg("goal_reached").params.get("threshold_m", 0.2)
    plan_len = base.path_manager.current_path_length.clone()
    episodes_done = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    counts = torch.zeros((K, 4), dtype=torch.long)

    env.unwrapped.seed(args.seed)
    obs, _ = env.reset()
    with tqdm.tqdm(total=env.num_envs * args.episodes, desc="Evaluating") as pbar:
        while not bool((episodes_done >= args.episodes).all()):
            with torch.no_grad():
                policy(obs, stochastic_output=True)
                action = policy.output_mean  
            obs, _, dones, _ = env.step(action)
            done_ids = dones.bool().nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                reached = base.termination_manager.get_term("goal_reached")
                timed_out = base.termination_manager.get_term("timeout")
                no_path = reached & (plan_len < goal_thresh)
                arrived = reached & ~no_path
                keep = done_ids[episodes_done[done_ids] < args.episodes]
                for i in keep.tolist():
                    col = 0 if arrived[i] else 1 if no_path[i] else 2 if timed_out[i] else 3
                    counts[int(apt[i]), col] += 1
                pbar.update(len(keep))
                episodes_done[done_ids] += 1
            plan_len = base.path_manager.current_path_length.clone()

    print(f"\ncheckpoint: {args.checkpoint}")
    print(f"maps: {maps}, {args.num_envs} envs, {args.episodes} episodes per env (deterministic, no curriculum)\n")
    print(f"{'apartment':15} {'episodes':>8} {'success':>8} {'arrived':>8} {'no_path':>8} {'timeout':>8} {'other':>8}")
    for k, name in enumerate(maps):
        n = int(counts[k].sum())
        succ = (counts[k, 0] + counts[k, 1]).item() / max(n, 1)
        print(f"{name:15} {n:8d} {succ:8.1%} " + " ".join(f"{int(c):8d}" for c in counts[k]))
    n = int(counts.sum())
    print(f"{'ALL':15} {n:8d} {(counts[:, 0] + counts[:, 1]).sum().item() / max(n, 1):8.1%}")
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