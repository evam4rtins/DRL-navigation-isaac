"""
Record the lidar teacher driving in go2_vision_multi_full: the dataset for the offline student.

At every env step, for every robot, it saves what the student will see and what the teacher says:
  student inputs : dino (DINOv2 tokens, bf16), global_plan, velocity_buffer, action_buffer
  targets        : teacher_action (mean of the teacher's distribution), teacher_value (critic), lidar (fp16)
  bookkeeping    : apt (apartment index), done, time_out (the step ended the episode / by timeout)

Run :
    env_isaaclab/bin/python scripts/vision/record_teacher.py --maps 3 4 8 20 21 24 --num_envs 64 --steps 1000 \
        --out logs/datasets/teacher_a --checkpoint logs/rsl_rl/training/multi_18maps_full_11-11_11-30/model_2100.pt --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--maps", nargs="+", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1000, help="env steps to record (samples = steps x num_envs)")
parser.add_argument("--chunk_steps", type=int, default=50)
parser.add_argument("--pool", type=int, default=2, help="DINOv2 patch pooling (2 = 8x8 patches, 65 tokens)")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--sample", action="store_true",
                    help="drive with actions sampled from the teacher (more varied states); labels are always the mean")
parser.add_argument("--warmup_steps", type=int, default=20, help="steps run before recording (envs settle, resets spread)")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.kit_args = ((args.kit_args or "") + " --/renderer/scenePartitioning/enabled=true").strip()
simulation_app = AppLauncher(args).app

import json
import os
import time

import gymnasium as gym
import torch

from cfg.CFG import ROOT_DIR, get_scene_usd_paths, set_map_names
from tasks.task_utils import get_env_config

STUDENT_INPUTS = ["dino", "global_plan", "velocity_buffer", "action_buffer"]
FP16 = {"lidar", "global_plan", "velocity_buffer", "action_buffer"}

def main():
    maps = set_map_names(args.maps)
    cfg = get_env_config("go2_vision_multi_full")
    cfg.curriculum = None
    cfg.scene.num_envs = args.num_envs
    cfg.observations.dino.tokens.params["pool"] = args.pool
    cfg.seed = args.seed
    gym.register(
        id="Isaac-record-teacher-v0",
        entry_point="navigation_env.NavigationEnv:NavEnv",
        disable_env_checker=True,
        kwargs={"scene_path": get_scene_usd_paths(), "use_long_horizon": False, "sample_voronoi_probability": 0.25},
    )
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from policy.NavPolicyv2 import load_policy_checkpoint, make_go2_policy_cfg

    env = RslRlVecEnvWrapper(gym.make("Isaac-record-teacher-v0", cfg=cfg))
    base = env.unwrapped
    policy_cfg = make_go2_policy_cfg(cfg.obs_groups)
    policy_cfg["num_envs"] = args.num_envs
    runner = OnPolicyRunner(env, policy_cfg, log_dir=None, device=policy_cfg["device"])
    ckpt = args.checkpoint if args.checkpoint.startswith("/") else str(ROOT_DIR / args.checkpoint)
    load_policy_checkpoint(runner, ckpt)
    actor = runner.get_inference_policy(env.device)
    critic = runner.alg.critic

    os.makedirs(args.out, exist_ok=True)
    apt = base.env_apt_ids.clone().cpu()
    obs, _ = env.reset()

    def teacher(obs):
        with torch.no_grad():
            actor(obs, stochastic_output=True)
            mean = actor.output_mean.clone()
            value = critic(obs)
            value = value[0] if isinstance(value, (tuple, list)) else value
        return mean, value.reshape(-1, 1)

    for _ in range(args.warmup_steps):
        mean, _ = teacher(obs)
        obs, _, _, _ = env.step(mean)

    meta = dict(maps=maps, num_envs=args.num_envs, steps=args.steps, chunk_steps=args.chunk_steps, pool=args.pool,
                checkpoint=args.checkpoint, sample=args.sample, seed=args.seed, keys={}, chunks=[])
    chunk, n_chunk, t0 = {}, 0, time.time()

    def flush():
        nonlocal chunk, n_chunk
        if not chunk:
            return
        data = {k: torch.stack(v) for k, v in chunk.items()}
        data["apt"] = apt
        path = os.path.join(args.out, f"chunk_{n_chunk:04d}.pt")
        torch.save(data, path)
        meta["chunks"].append(os.path.basename(path))
        if not meta["keys"]:
            meta["keys"] = {k: [list(v.shape[2:]) if v.dim() > 2 else [], str(v.dtype)] for k, v in data.items()
                            if k != "apt"}
        chunk, n_chunk = {}, n_chunk + 1

    for step in range(args.steps):
        mean, value = teacher(obs)
        row = {k: obs[k] for k in STUDENT_INPUTS + ["lidar"]}
        row["teacher_action"] = mean
        row["teacher_value"] = value
        action = actor.output_mean if not args.sample else actor(obs, stochastic_output=True)
        obs, _, dones, extras = env.step(mean if not args.sample else action)
        row["done"] = dones.bool()
        row["time_out"] = extras.get("time_outs", torch.zeros_like(dones)).bool()
        for k, v in row.items():
            v = v.detach()
            v = v.to(torch.float16) if k in FP16 else v
            chunk.setdefault(k, []).append(v.cpu())
        if (step + 1) % args.chunk_steps == 0:
            flush()
            rate = (step + 1) * args.num_envs / (time.time() - t0)
            print(f"[record] {step + 1}/{args.steps} steps, {(step + 1) * args.num_envs} samples, "
                  f"{rate:.0f} samples/s", flush=True)
    flush()
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved {len(meta['chunks'])} chunks, {args.steps * args.num_envs} samples in {args.out}")
    print("keys:", meta["keys"])
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    simulation_app.close()