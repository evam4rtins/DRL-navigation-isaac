from __future__ import annotations

import torch
import warp as wp
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.cloner.cloner_utils import iter_clone_plan_matches
from isaaclab_physx.physics import PhysxManager
from isaaclab_physx.sensors.ray_caster.multi_mesh_ray_caster import MultiMeshRayCaster
 

def _has_rigid_body_api(prim) -> bool:
    return bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))

class PerEnvMultiMeshRayCaster(MultiMeshRayCaster):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._per_env_slots: dict[str, list[tuple[int, int, str]]] = {}
        self._scatter: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    # which body moves which mesh
    def _build_mesh_records(self, target_cfg, plan, dummy_mesh_id):
        if plan is None or not target_cfg.track_mesh_transforms:
            return super()._build_mesh_records(target_cfg, plan, dummy_mesh_id)
        
        records_per_env = [[] for _ in range(self._num_envs)]   
        slots: list[tuple[int, int, str]] = [] 
        found = False

        for source_root, destination_template, source_path, env_ids in iter_clone_plan_matches(plan, target_cfg.prim_expr):
            found = True
            source_prims = sim_utils.find_matching_prims(source_path)
            meshes: list[tuple[int, str]] = []
            for source_prim in source_prims:
                owner = sim_utils.get_first_matching_ancestor_prim(str(source_prim.GetPath()), _has_rigid_body_api)
                owner_path = None if owner is None else str(owner.GetPath())
                if owner_path and (owner_path == source_root or owner_path.startswith(f"{source_root}/")):
                    bodies = [(source_prim, owner)]
                else:
                    bodies = [(p,p) for p in sim_utils.get_all_matching_child_prims(str(source_prim.GetPath()), _has_rigid_body_api)]
                if not bodies:
                    raise RuntimeError(f"'{source_prim.GetPath()}' has no rigid body to track it with.")
                for geometry_prim, body_prim in bodies:
                    mesh_id = self._load_target_prim_warp_mesh(geometry_prim, target_cfg, reference_prim=body_prim)
                    dummy_mesh_id = mesh_id if dummy_mesh_id is None else dummy_mesh_id
                    meshes.append((mesh_id, str(body_prim.GetPath())[len(source_root):]))
        
            for env_id in env_ids:
                for mesh_id, body_suffix in meshes:
                    slot = len(records_per_env[env_id])
                    records_per_env[env_id].append((mesh_id, (1.0e9, 1.0e9, 1.0e9), (0.0, 0.0, 0.0, 1.0)))
                    slots.append((env_id, slot, destination_template.format(env_id) + body_suffix))
            
        if not found:
            return super()._build_mesh_records(target_cfg, plan, dummy_mesh_id)
        
        self._per_env_slots[target_cfg.prim_expr] = slots
        unique_bodies = list(dict.fromkeys(path for _,_, path in slots))
        return records_per_env, dummy_mesh_id, unique_bodies
    
    # one view over exactly those bodies 
    def _create_tracked_target_view(self, target_prim_paths):
        paths = [target_prim_paths] if isinstance(target_prim_paths, str) else list(target_prim_paths)
        physics_sim_view = PhysxManager.get_physics_sim_view()
        if physics_sim_view is None:
            raise RuntimeError("PhysX simulation view is not initialized.")
        return physics_sim_view.create_rigid_body_view(paths)
 
    def _initialize_warp_meshes(self):
        super()._initialize_warp_meshes()
        # map every row of each view to the (env, column) slots it has to fill in the mesh table
        mesh_offset = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            expr = target_cfg.prim_expr
            if target_cfg.track_mesh_transforms:
                if expr not in self._per_env_slots:
                    raise RuntimeError(f"PerEnvMultiMeshRayCaster: tracked target '{expr}' is not env-scoped.")
                row_of = {path: row for row, path in enumerate(view.prim_paths)}
                missing = [p for _, _, p in self._per_env_slots[expr] if p not in row_of]
                if missing:
                    raise RuntimeError(f"Bodies missing from the PhysX view for '{expr}', e.g. {missing[:3]}")
                rows, envs, cols = zip(*((row_of[p], e, mesh_offset + s) for e, s, p in self._per_env_slots[expr]))
                self._scatter[expr] = tuple(torch.tensor(x, dtype=torch.long, device=self._device)
                                            for x in (rows, envs, cols))
            mesh_offset += self._num_meshes_per_env[expr]
 
    # copy poses into their own slots 
    def _update_mesh_transforms(self):
        if not hasattr(self, "_mesh_views") or not self._scatter:
            return
        positions = wp.to_torch(self._mesh_positions_w)  # (num_envs, num_meshes, 3), shares memory
        orientations = wp.to_torch(self._mesh_orientations_w)  # (num_envs, num_meshes, 4), quat (x, y, z, w)
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                continue
            transforms = view.get_transforms()
            transforms = wp.to_torch(transforms) if isinstance(transforms, wp.array) else transforms
            transforms = transforms.reshape(-1, 7)  # (pos, quat x y z w), one row per body
            rows, envs, cols = self._scatter[target_cfg.prim_expr]
            positions[envs, cols] = transforms[rows, :3]
            orientations[envs, cols] = transforms[rows, 3:7]
 