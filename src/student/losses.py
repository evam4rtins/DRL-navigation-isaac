"""
One loss per transfer signal:
  action : MSE between the student's mean action and the teacher's mean action
  value  : MSE between the student's critic and the teacher's critic, divided by the teacher value variance
  lidar  : MSE between the adapter's lidar head and the teacher's lidar profile (representation transfer)
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from student.config import TransferCfg
from student.models import StudentModel
from student.targets import lidar_profile


def compute_losses(actor: StudentModel, critic: StudentModel, batch: TensorDict, transfer: TransferCfg,
                   stats: dict, shuffle_image: bool = False) -> dict[str, torch.Tensor]:
    device = batch["teacher_action"].device
    obs = batch
    if shuffle_image and actor.adapter is not None:
        obs = batch.clone(recurse=False)
        obs["dino"] = batch["dino"][torch.randperm(batch.batch_size[0], device=device)]
    out: dict[str, torch.Tensor] = {}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        action = actor(obs)  # deterministic: the mean of the distribution
        features = actor.adapter_features
        out["action_mse"] = (action.float() - batch["teacher_action"]).pow(2).mean()
        if transfer.value > 0:
            value = critic(obs).float()
            out["value_mse"] = ((value - batch["teacher_value"]) / stats["value_std"]).pow(2).mean()
        if transfer.lidar > 0 and features is not None:
            pred = torch.sigmoid(actor.adapter.predict_lidar(features).float())
            err = (pred - lidar_profile(batch["lidar"], transfer.lidar_bins)).pow(2)
            out["lidar_mse"] = err.mean()
            out["lidar_mse_fov"] = err[:, stats["fov"]].mean()
            out["lidar_mse_out"] = err[:, ~stats["fov"]].mean()
    return out


def weighted_total(losses: dict[str, torch.Tensor], transfer: TransferCfg) -> torch.Tensor:
    total = transfer.action * losses["action_mse"]
    if "value_mse" in losses:
        total = total + transfer.value * losses["value_mse"]
    if "lidar_mse" in losses:
        total = total + transfer.lidar * losses["lidar_mse"]
    return total
