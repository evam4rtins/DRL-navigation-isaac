from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from isaaclab.envs import ManagerBasedRLEnv
import timm

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

class DinoTokens:
    MODEL = "vit_small_patch14_dinov2.lvd142m"
    EMBED_DIM = 384
    PATCH = 14

    def __init__(self, chunk: int = 256):
        self.model = None
        self.chunk = chunk
    
    def _ensure_model(self, device, res: int):
        if self.model is not None:
            return
        
        self.model = timm.create_model(
            self.MODEL, pretrained=True, num_classes=0, img_size=res,
            cache_dir=Path("~/.cache/torch/hub/timm").expanduser(),
        ).to(device).eval().requires_grad_(False)

        self.mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, env: ManagerBasedRLEnv, pool: int = 2) -> torch.Tensor:
        rgb = env.scene["camera"].data.output["rgb"].torch[..., :3] # (N, H, W, 3)
        n, h, w, _ = rgb.shape
        assert h % self.PATCH == 0 and w % self.PATCH == 0, f"image {h} x {w} is not a multiple of {self.PATCH}"
        self._ensure_model(rgb.device, h)
        gh = h // self.PATCH
        gw = w // self.PATCH
        assert gh % pool == 0 and gw % pool == 0, f"pool {pool} does not divide the {gh}x{gw} patch grid"

        out = []
        for i in range(0, n, self.chunk):
            x = rgb[i:i + self.chunk].permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - self.mean) / self.std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = self.model.forward_features(x)  # (B, 1 [+ registers] + gh*gw, D)
            cls, patches = tokens[:, :1], tokens[:, -gh * gw:]
            if pool > 1:
                grid = patches.transpose(1, 2).reshape(-1, self.EMBED_DIM, gh, gw)
                patches = F.avg_pool2d(grid.float(), pool).flatten(2).transpose(1, 2)
            out.append(torch.cat([cls.float(), patches.float()], dim=1).to(torch.bfloat16))
        return torch.cat(out).flatten(1)  # (N, T * D)


def num_dino_tokens(res: int = 224, pool: int = 2) -> int:
    return 1 + (res // DinoTokens.PATCH // pool) ** 2