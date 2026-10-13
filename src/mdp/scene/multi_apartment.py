import torch
import isaaclab.cloner as cloner
from cfg.CFG import get_env_apartments_ids

_ORIGINAL_GRID_TRANSFORMS = cloner.grid_transforms

def use_apartment_origins(apartment_origins: torch.Tensor) -> None:
    origins = apartment_origins.detach().clone().to(torch.float32)

    def apartment_transforms(N: int, spacing: float = 1.0, up_axis: str = "z", device="cpu"):
        _, quat = _ORIGINAL_GRID_TRANSFORMS(N, spacing, up_axis=up_axis, device=device)
        pos = origins[get_env_apartments_ids(N, origins.shape[0])].to(device)
        return pos, quat

    cloner.grid_transforms = apartment_transforms

def restore_grid_origins() -> None:
    cloner.grid_transforms = _ORIGINAL_GRID_TRANSFORMS