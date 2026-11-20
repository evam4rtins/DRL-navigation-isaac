"""
The lidar teacher drives go2_vision_multi_full and every step is written to disk in the format of student.data.
"""

from __future__ import annotations

import json
import os
import time

import gymnasium as gym
import torch

from cfg.CFG import ROOT_DIR, get_scene_usd_paths, set_map_names
from student.config import ExperimentCfg
from student.data import SAMPLE_KEYS
from tasks.task_utils import get_env_config

FP16_KEYS = {"lidar", "global_plan", "velocity_buffer", "action_buffer"}
OBS_KEYS = [k for k in SAMPLE_KEYS if k not in ("teacher_action", "teacher_value")]


def make_vision_env(maps, num_envs: int, pool: int, seed: int):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    maps = set_map_names(maps)
    env_cfg = get_env_config("go2_vision_multi_full")
    env_cfg.curriculum = None
    env_cfg.scene.num_envs = num_envs
    env_cfg.observations.dino.tokens.params["pool"] = pool
    env_cfg.seed = seed
    env_id = "Isaac-student-vision-v0"
    if env_id not in gym.registry:
        gym.register(id=env_id, entry_point="navigation_env.NavigationEnv:NavEnv", disable_env_checker=True,
                     kwargs={"scene_path": get_scene_usd_paths(), "use_long_horizon": False,
                             "sample_voronoi_probability": 0.25})
    return RslRlVecEnvWrapper(gym.make(env_id, cfg=env_cfg)), env_cfg, maps


def load_teacher(env, env_cfg, checkpoint: str):
    from rsl_rl.runners import OnPolicyRunner

    from policy.NavPolicyv2 import load_policy_checkpoint, make_go2_policy_cfg

    policy_cfg = make_go2_policy_cfg(env_cfg.obs_groups)
    policy_cfg["num_envs"] = env.num_envs
    runner = OnPolicyRunner(env, policy_cfg, log_dir=None, device=policy_cfg["device"])
    path = checkpoint if checkpoint.startswith("/") else str(ROOT_DIR / checkpoint)
    load_policy_checkpoint(runner, path)
    actor = runner.get_inference_policy(env.device)
    critic = runner.alg.critic

    @torch.no_grad()
    def teacher(obs):
        sampled = actor(obs, stochastic_output=True)
        mean = actor.output_mean.clone()
        value = critic(obs)
        value = value[0] if isinstance(value, (tuple, list)) else value
        return mean, sampled, value.reshape(-1, 1)

    return teacher


def run(cfg: ExperimentCfg) -> None:
    r = cfg.record
    assert r.maps, "record.maps is empty: give the apartments of this batch (--set record.maps=[3,4,8,20])"
    env, env_cfg, maps = make_vision_env([str(m) for m in r.maps], r.num_envs, cfg.student.pool, r.seed)
    teacher = load_teacher(env, env_cfg, cfg.teacher_checkpoint)
    os.makedirs(r.out, exist_ok=True)
    apt = env.unwrapped.env_apt_ids.clone().cpu()

    obs, _ = env.reset()
    for _ in range(r.warmup_steps):
        mean, _, _ = teacher(obs)
        obs, _, _, _ = env.step(mean)

    meta = dict(maps=maps, num_envs=r.num_envs, steps=r.steps, chunk_steps=r.chunk_steps, pool=cfg.student.pool,
                checkpoint=cfg.teacher_checkpoint, sample=r.sample, seed=r.seed, keys={}, chunks=[])
    chunk, n_chunk, t0 = {}, 0, time.time()

    def flush():
        nonlocal chunk, n_chunk
        if not chunk:
            return
        data = {k: torch.stack(v) for k, v in chunk.items()}
        data["apt"] = apt
        name = f"chunk_{n_chunk:04d}.pt"
        torch.save(data, os.path.join(r.out, name))
        meta["chunks"].append(name)
        meta["keys"] = meta["keys"] or {k: [list(v.shape[2:]), str(v.dtype)] for k, v in data.items() if k != "apt"}
        chunk, n_chunk = {}, n_chunk + 1

    for step in range(r.steps):
        mean, sampled, value = teacher(obs)
        row = {k: obs[k] for k in OBS_KEYS}
        row["teacher_action"], row["teacher_value"] = mean, value
        obs, _, dones, extras = env.step(sampled if r.sample else mean)
        row["done"] = dones.bool()
        terms = env.unwrapped.termination_manager
        row["time_out"] = terms.get_term("timeout").clone().bool()
        row["goal_reached"] = terms.get_term("goal_reached").clone().bool()
        for k, v in row.items():
            v = v.detach()
            chunk.setdefault(k, []).append((v.to(torch.float16) if k in FP16_KEYS else v).cpu())
        if (step + 1) % r.chunk_steps == 0:
            flush()
            rate = (step + 1) * r.num_envs / (time.time() - t0)
            print(f"[record] {step + 1}/{r.steps} steps, {(step + 1) * r.num_envs} samples, {rate:.0f} samples/s",
                  flush=True)
    flush()
    with open(os.path.join(r.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved {len(meta['chunks'])} chunks, {r.steps * r.num_envs} samples in {r.out}")
    env.close()