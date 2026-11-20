"""
What is transferred from the teacher to the student.

- action: the mean of the teacher's action distribution
- value : the teacher's critic output
- lidar : the teacher's lidar observation turned into a polar range profile
"""

import torch

def lidar_profile(lidar: torch.Tensor, bins: int) -> torch.Tensor:
    """ Closest normalised distance per yaw bin. """

    k = lidar.shape[1] // 2
    dist, yaw = lidar[:, :k].float(), lidar[:, k:].float()
    idx = (yaw * bins).long().clamp_(0, bins - 1)
    out = torch.ones(lidar.shape[0], bins, device=lidar.device)
    return out.scatter_reduce_(1, idx, dist, reduce="amin", include_self=True)


def fov_mask(bins: int, fov_deg: float) -> torch.Tensor:
    """(bins,) True for the yaw sectors inside the camera's horizontal field of view."""
    centre = (torch.arange(bins) + 0.5) / bins
    degrees_from_ahead = torch.minimum(centre, 1 - centre) * 360.0
    return degrees_from_ahead <= fov_deg / 2
