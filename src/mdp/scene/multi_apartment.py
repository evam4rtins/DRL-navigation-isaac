import torch
import isaaclab.cloner as cloner
import isaaclab.sim as sim_utils
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

def spawn_static_apartment(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """ spawn an apartment USD and deactivate its doors and chairs """
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    dyn = prim.GetPrimAtPath("Meshes/dynamic_objects")
    if dyn.IsValid():
        dyn.SetActive(False)
    return prim