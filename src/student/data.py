from __future__ import annotations
import glob
import os
import random
from typing import Iterator
import torch
from tensordict import TensorDict

from student.config import DataCfg
from student.models import IMAGE_GROUP, OTHER_GROUPS


TARGET_KEYS = ["teacher_action", "teacher_value", "lidar"]
SAMPLE_KEYS = [IMAGE_GROUP] + OTHER_GROUPS + TARGET_KEYS


def list_chunks(cfg: DataCfg) -> list[str]:
    paths: list[str] = []
    for d in cfg.dirs:
        paths += sorted(glob.glob(os.path.join(d, "**", "chunk_*.pt"), recursive=True))
    if cfg.max_chunks:
        paths = paths[: cfg.max_chunks]
    if not paths:
        raise FileNotFoundError(f"no chunk_*.pt under {cfg.dirs}")
    return paths


def load_chunk(path: str, split: str, cfg: DataCfg) -> dict[str, torch.Tensor]:
    """Samples of one chunk for one split (train / val), flattened over (steps, robots)."""
    d = torch.load(path, map_location="cpu")
    n = d["teacher_action"].shape[1]
    is_val = (torch.arange(n) % cfg.val_mod) == cfg.val_mod - 1
    keep = is_val if split == "val" else ~is_val
    return {k: d[k][:, keep].flatten(0, 1) for k in SAMPLE_KEYS}


def concat(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def load_validation(paths: list[str], cfg: DataCfg) -> dict[str, torch.Tensor]:
    return concat([load_chunk(p, "val", cfg) for p in paths[:: cfg.val_chunk_every]])


def train_groups(paths: list[str], cfg: DataCfg, rng: random.Random) -> Iterator[dict[str, torch.Tensor]]:
    order = paths[:]
    rng.shuffle(order)
    for i in range(0, len(order), cfg.chunks_in_ram):
        yield concat([load_chunk(p, "train", cfg) for p in order[i:i + cfg.chunks_in_ram]])


def batches(data: dict[str, torch.Tensor], batch: int, device: str, shuffle: bool) -> Iterator[TensorDict]:
    n = data["teacher_action"].shape[0]
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch):
        idx = order[i:i + batch]
        out = {}
        for k, v in data.items():
            v = v[idx].to(device, non_blocking=True)
            out[k] = v.float() if (k in OTHER_GROUPS or k == "lidar") else v
        yield TensorDict(out, batch_size=[len(idx)])
