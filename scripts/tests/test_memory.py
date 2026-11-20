"""
GPU memory of the vision task with a PPO-sized buffer.

Run (pool 2 = 8x8 patches, 4 = 4x4, 1 = full 16x16):
    env_isaaclab/bin/python scripts/vision/test_memory.py --maps 31 --num_envs 64 --pool 2 --headless
    env_isaaclab/bin/python scripts/vision/test_memory.py --maps 31 36 --num_envs 256 --pool 2 --steps_per_env 188 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--maps", nargs="+", default=["31"])
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--pool", type=int, default=2, help="DINOv2 patch pooling: 1 = 16x16, 2 = 8x8, 4 = 4x4")
parser.add_argument("--steps_per_env", type=int, default=188, help="rollout length (Matteo's PPO: 188)")
parser.add_argument("--fill_steps", type=int, default=30, help="env steps actually run (speed); <= steps_per_env")
parser.add_argument("--mini_batches", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.kit_args = ((args.kit_args or "") + " --/renderer/scenePartitioning/enabled=true").strip()
simulation_app = AppLauncher(args).app

import time

import gymnasium as gym
import torch

from cfg.CFG import get_scene_usd_paths, set_map_names
from mdp.observations.dino import DinoTokens, num_dino_tokens
from policy.adapter import QueryAdapter
from tasks.task_utils import get_env_config


def device_used_gb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**30


def report(stage: str, base: float | None = None):
    used = device_used_gb()
    extra = f"  (+{used - base:.2f})" if base is not None else ""
    print(f"[memory] {stage:38} {used:6.2f} GB used of {torch.cuda.mem_get_info()[1] / 2**30:.1f}{extra}")
    return used


def main():
    set_map_names(args.maps)
    report("start (Isaac Sim launched)")
    cfg = get_env_config("go2_vision_multi_full")
    cfg.curriculum = None
    cfg.scene.num_envs = args.num_envs
    cfg.observations.dino.tokens.params["pool"] = args.pool
    gym.register(
        id="Isaac-vision-memory-test-v0",
        entry_point="navigation_env.NavigationEnv:NavEnv",
        disable_env_checker=True,
        kwargs={"scene_path": get_scene_usd_paths(), "use_long_horizon": False, "sample_voronoi_probability": 0.25},
    )
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(gym.make("Isaac-vision-memory-test-v0", cfg=cfg))
    obs = env.get_observations()
    m_env = report("env built (+ DINOv2 loaded)")

    T, N = args.steps_per_env, args.num_envs
    buffer = {}
    print(f"\nbuffer: {T} steps x {N} envs = {T * N} samples")
    for key in obs.keys():
        t = obs[key]
        buffer[key] = torch.zeros((T, *t.shape), dtype=t.dtype, device=t.device)
        print(f"  {key:16} {str(tuple(t.shape[1:])):14} {str(t.dtype):15} {buffer[key].numel() * buffer[key].element_size() / 2**30:6.2f} GB")
    tokens_expected = num_dino_tokens(224, args.pool)
    print(f"  dino tokens per image: {obs['dino'].shape[1] // DinoTokens.EMBED_DIM} (expected {tokens_expected})")
    m_buf = report("rollout buffer allocated", m_env)

    actions = torch.zeros(N, env.num_actions, device=env.device)
    steps = min(args.fill_steps, T)
    t0 = None
    for s in range(steps):
        obs, _, _, _ = env.step(actions)
        for key in buffer:
            buffer[key][s] = obs[key]
        if s == 4:
            torch.cuda.synchronize()
            t0 = time.time()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / max(steps - 5, 1)
    print(f"\nstep with cameras + DINOv2: {dt * 1000:.0f} ms -> {N / dt:.0f} robot steps/s")
    report(f"after {steps} env steps", m_env)

    adapter = QueryAdapter(token_dim=DinoTokens.EMBED_DIM).to(env.device)
    head = torch.nn.Sequential(torch.nn.Linear(adapter.out_dim, 256), torch.nn.ELU(), torch.nn.Linear(256, 3)).to(env.device)
    params = list(adapter.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)
    mb = T * N // args.mini_batches
    idx = torch.randint(0, steps * N, (mb,), device=env.device)
    tokens = buffer["dino"][:steps].flatten(0, 1)[idx]
    torch.cuda.reset_peak_memory_stats()
    before = device_used_gb()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = head(adapter(tokens)).float().pow(2).mean()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        print(f"\nPPO-like update on {mb} samples: torch peak +{torch.cuda.max_memory_allocated() / 2**30:.2f} GB "
              f"(device {before:.2f} -> {device_used_gb():.2f} GB)")
    except torch.cuda.OutOfMemoryError:
        print(f"\nPPO-like update on {mb} samples: OUT OF MEMORY")
    report("after the update", m_env)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    simulation_app.close()
