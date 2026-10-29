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

_ORIGINAL_MAKE_CLONE_PLAN = cloner.make_clone_plan

def use_round_robin_cloning() -> None:
    """ give env i the variant i % K of every multi-variant asset"""

    def round_robin_clone_plan(sources, destinations, num_clones, clone_strategy=None, device="cpu"):
        return _ORIGINAL_MAKE_CLONE_PLAN(sources, destinations, num_clones, cloner.sequential, device)
    
    cloner.make_clone_plan = round_robin_clone_plan

def restore_clone_plan() -> None:
    cloner.make_clone_plan = _ORIGINAL_MAKE_CLONE_PLAN

def spawn_dynamic_only(prim_path, cfg, translation=None, orientation=None, **kwargs):
    " spawn an apartment USD and keep only its doors and chairs"
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    meshes = prim.GetChild("Meshes")
    if not meshes.IsValid():
        raise RuntimeError(f"{prim_path}/Meshes not found. Children: {[c.GetName() for c in prim.GetChildren()]} ")
    for child in meshes.GetChildren():
        if child.GetName() != "dynamic_objects":
            child.SetActive(False)
    rendering = prim.GetChild("Rendering")
    if rendering.IsValid():
        rendering.SetActive(False)
    return prim