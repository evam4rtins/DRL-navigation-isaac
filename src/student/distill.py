"""
Offline distillation: behaviour cloning of the teacher on recorded data.
"""
from __future__ import annotations

import json
import math
import random
import time

import torch
import yaml

from student import data as D
from student.config import ExperimentCfg, to_dict
from student.losses import compute_losses, weighted_total
from student.models import build_student, student_parameters
from student.targets import fov_mask


@torch.no_grad()
def evaluate(actor, critic, val, cfg: ExperimentCfg, stats: dict) -> dict[str, float]:
    actor.eval()
    critic.eval()
    sums: dict[str, float] = {}
    n = 0
    for b in D.batches(val, cfg.distill.batch, cfg.device, shuffle=False):
        m = b.batch_size[0]
        losses = compute_losses(actor, critic, b, cfg.transfer, stats)
        if actor.adapter is not None:
            losses["action_mse_shuffled_image"] = compute_losses(
                actor, critic, b, cfg.transfer, stats, shuffle_image=True)["action_mse"]
        losses["action_mse_mean_baseline"] = (stats["action_mean"] - b["teacher_action"]).pow(2).mean()
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + v.item() * m
        n += m
    actor.train()
    critic.train()
    return {k: v / n for k, v in sums.items()}


def run(cfg: ExperimentCfg) -> None:
    if not cfg.distill.enabled:
        print(f"distillation is disabled for '{cfg.name}' (e.g. the scratch condition): nothing to do")
        return
    torch.manual_seed(cfg.distill.seed)
    rng = random.Random(cfg.distill.seed)
    out = cfg.run_dir("distill")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)

    paths = D.list_chunks(cfg.data)
    val = D.load_validation(paths, cfg.data)
    first = D.concat([D.load_chunk(p, "train", cfg.data) for p in paths[: cfg.data.chunks_in_ram]])
    stats = {
        "value_mean": first["teacher_value"].mean().item(),
        "value_std": first["teacher_value"].std().item() + 1e-6,
        "action_mean": first["teacher_action"].mean(0).to(cfg.device),
        "fov": fov_mask(cfg.transfer.lidar_bins, cfg.transfer.fov_deg).to(cfg.device),
    }
    if cfg.distill.overfit:
        first = {k: v[: cfg.distill.overfit] for k, v in first.items()}
        val = first
        print(f"OVERFIT check on {cfg.distill.overfit} samples")
    samples_per_epoch = len(first["teacher_action"]) if cfg.distill.overfit else \
        len(first["teacher_action"]) * math.ceil(len(paths) / cfg.data.chunks_in_ram)
    print(f"{len(paths)} chunks, ~{samples_per_epoch} train samples per epoch, {len(val['teacher_action'])} val samples")

    sample = next(D.batches(first, 4, cfg.device, shuffle=False))
    actor, critic = build_student(cfg, sample)
    actor.to(cfg.device)
    critic.to(cfg.device)
    params = student_parameters(actor, critic)
    print(f"student '{cfg.name}': {sum(p.numel() for p in params) / 1e6:.2f} M parameters, image {cfg.student.image}, "
          f"weights action {cfg.transfer.action} / value {cfg.transfer.value} / lidar {cfg.transfer.lidar}")

    opt = torch.optim.AdamW(params, lr=cfg.distill.lr, weight_decay=cfg.distill.weight_decay)
    total_steps = cfg.distill.epochs * math.ceil(samples_per_epoch / cfg.distill.batch) + 10
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.distill.lr, total_steps=total_steps)

    history, best = [], float("inf")
    for epoch in range(cfg.distill.epochs):
        t0, running, nb = time.time(), 0.0, 0
        groups = [first] if cfg.distill.overfit else D.train_groups(paths, cfg.data, rng)
        for group in groups:
            for b in D.batches(group, cfg.distill.batch, cfg.device, shuffle=True):
                loss = weighted_total(compute_losses(actor, critic, b, cfg.transfer, stats), cfg.transfer)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                if sched.last_epoch < total_steps - 1:
                    sched.step()
                running, nb = running + loss.item(), nb + 1
        metrics = evaluate(actor, critic, val, cfg, stats)
        metrics.update(epoch=epoch + 1, train_loss=running / max(nb, 1), minutes=round((time.time() - t0) / 60, 2))
        history.append(metrics)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in metrics.items()),
              flush=True)
        if metrics["action_mse"] < best:
            best = metrics["action_mse"]
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "config": to_dict(cfg),
                        "stats": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in stats.items()},
                        "epoch": epoch + 1}, out / "best.pt")
        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=1)
    print(f"\nbest validation action_mse {best:.4f} -> {out / 'best.pt'}")
