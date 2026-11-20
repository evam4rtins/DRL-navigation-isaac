from __future__ import annotations
from typing import Any
from rsl_rl.modules import HiddenState
import torch
from tensordict import TensorDict

from policy.adapter import QueryAdapter
from policy.NavPolicyv2 import ENCODERS_HIDDEN_DIMS, MLPModelWithEncoders, resolve_legacy_obs_groups
from student.config import ExperimentCfg

IMAGE_GROUP = "dino"
OTHER_GROUPS = ["global_plan", "velocity_buffer", "action_buffer"]
TOKEN_DIM = 384  # DINOv2 Vit-S/14

def student_obs_groups(image: bool) -> dict[str, list[str]]:
    groups = ([IMAGE_GROUP] if image else []) + OTHER_GROUPS
    return {"actor": list(groups), "critic": list(groups)}

class StudentModel(MLPModelWithEncoders):
   
    def __init__(self, obs: TensorDict, obs_groups: dict, obs_set: str, output_dim: int,
                 adapter_cfg: dict | None = None, **kwargs: Any) -> None:
        self.adapter_cfg = dict(adapter_cfg or {})
        self.adapter_features = None
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)

    @property
    def adapter(self) -> QueryAdapter | None:
        return self.encoders[IMAGE_GROUP] if IMAGE_GROUP in self.encoders else None
    
    def get_latent(self, obs, masks=None, hidden_state=None):
        parts = []
        for group in self.obs_groups:
            out = self.encoders[group](obs[group])
            if group == IMAGE_GROUP:
                self.adapter_features = out
            parts.append(out.float())
        return self.obs_normalizer(torch.cat(parts, dim=-1))
    
    def _get_encoder_output_dim(self, group: str) -> int:
        if group == IMAGE_GROUP:
            return self.adapter_cfg.get("num_queries", 8) * self.adapter_cfg.get("dim", 128)
        return super()._get_encoder_output_dim(group)

    def _build_encoder(self, group: str):
        if group == IMAGE_GROUP:
            return QueryAdapter(token_dim=TOKEN_DIM, **self.adapter_cfg)
        return super()._build_encoder(group)
    

def build_student(cfg: ExperimentCfg, sample_obs: TensorDict, num_actions: int = 3) -> tuple[StudentModel, StudentModel]:
    """ Actor and critic of the student. """
    obs_groups = resolve_legacy_obs_groups(student_obs_groups(cfg.student.image))
    a = cfg.student.adapter
    adapter_cfg = dict(num_queries=a.num_queries, dim=a.dim, heads=a.heads, layers=a.layers, 
                       lidar_rays=cfg.transfer.lidar_bins if cfg.transfer.lidar > 0 else 0)
    
    common = dict(obs_groups=obs_groups, activation="elu", obs_normalization=False, encoders_hidden_dims=ENCODERS_HIDDEN_DIMS,
                  adapter_cfg=adapter_cfg)
    actor = StudentModel(sample_obs, obs_set="actor", output_dim=num_actions,
                         hidden_dims=cfg.student.actor_hidden_dims, last_activation="tanh",
                         distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
                         **common)
    critic = StudentModel(sample_obs, obs_set="critic", output_dim=1, hidden_dims=cfg.student.critic_hidden_dims,
                          last_activation=None, distribution_cfg=None, **common)
    if cfg.student.image and cfg.student.shared_adapter:
        critic.encoders[IMAGE_GROUP] = actor.encoders[IMAGE_GROUP]
    return actor, critic


def student_parameters(actor: StudentModel, critic: StudentModel) -> list[torch.nn.Parameter]:
    """Parameters of actor + critic, each once (the shared adapter would otherwise appear twice)."""
    seen, params = set(), []
    for p in list(actor.parameters()) + list(critic.parameters()):
        if id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    return params