from __future__ import annotations
import ast
import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class AdapterCfg:
    num_queries: int = 8
    dim: int = 128
    heads: int = 4
    layers: int = 1

@dataclass
class StudentCfg:
    image: bool = True # false = no camera at all
    pool: int = 2 # DINOv2 patch pooling 
    shared_adapter: bool = True # actor and critic use the same adapter
    adapter: AdapterCfg = field(default_factory=AdapterCfg)
    actor_hidden_dims: list[int] = field(default_factory=lambda : [256, 128, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 128])

@dataclass
class TransferCfg:
    """ Weight of each transfer signal in distillation (0 = not used) """
    action: float = 1.0
    value: float = 0.0
    lidar: float = 0.0
    lidar_bins: int = 72  # polar bins of the lidar target (72 = 5 deg)
    fov_deg: float = 106.0  # camera horizontal field of view

@dataclass
class DataCfg:
    dirs: list[str] = field(default_factory=lambda: ["logs/datasets/teacher_v1"])
    max_chunks: int = 0   # 0 = all
    chunks_in_ram: int = 6
    val_mod: int = 8   # robots with env index % val_mod == val_mod - 1 are validation
    val_chunk_every: int = 3  # validation samples are taken from every k-th chunk only

@dataclass
class DistillCfg:
    enabled: bool = True   # False = from scratch
    epochs: int = 10
    batch: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    overfit: int = 0 
    seed: int = 0

@dataclass
class RecordCfg:
    maps: list[str] = field(default_factory=list)
    num_envs: int = 64
    steps: int = 1000
    chunk_steps: int = 50
    warmup_steps: int = 200
    sample: bool = False
    out: str = "logs/datasets/teacher_v1/a"
    seed: int = 0

@dataclass
class ExperimentCfg:
    name: str = "actions"
    teacher_checkpoint: str = "logs/rsl_rl/training/multi_18maps_full_11-11_11-30/model_2100.pt"
    out_dir: str = "logs/student"
    device: str = "cuda"
    student: StudentCfg = field(default_factory=StudentCfg)
    transfer: TransferCfg = field(default_factory=TransferCfg)
    data: DataCfg = field(default_factory=DataCfg)
    distill: DistillCfg = field(default_factory=DistillCfg)
    record: RecordCfg = field(default_factory=RecordCfg)

    def run_dir(self, phase: str) -> Path:
        return Path(self.out_dir) / self.name / phase

def _merge(dc, values: dict):
    """Recursively set dataclass fields from a dict, refusing unknown keys (typos fail loudly)."""
    names = {f.name: f for f in dataclasses.fields(dc)}
    for key, value in values.items():
        if key not in names:
            raise KeyError(f"unknown config field '{key}' in {type(dc).__name__}; known: {sorted(names)}")
        current = getattr(dc, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dc, key, value)
    return dc


def _parse_override(item: str) -> dict:
    key, _, raw = item.partition("=")
    if not _:
        raise ValueError(f"override '{item}' must look like section.field=value")
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw  # plain string
    out: dict = {}
    node = out
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return out


def load_config(path: str | None, overrides: list[str] | None = None) -> ExperimentCfg:
    cfg = ExperimentCfg()
    if path:
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        _merge(cfg, _parse_override(item))
    return cfg


def to_dict(cfg: ExperimentCfg) -> dict:
    return copy.deepcopy(dataclasses.asdict(cfg))

