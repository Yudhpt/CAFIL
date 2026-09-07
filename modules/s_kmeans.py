"""Spherical k-means used to build CAFIL's global concept dictionary."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def spherical_kmeans(x: torch.Tensor, k: int, *, iters: int = 30) -> torch.Tensor:
    """Cluster ``[N, D]`` vectors on the unit sphere and return ``[k, D]`` centers.

    Initialization is deterministic for a fixed input: after choosing the
    first point, each following center is the point farthest from existing
    centers. Empty clusters retain their preceding center.
    """
    if x.ndim != 2:
        raise ValueError(f"spherical_kmeans expects [N,D], got {tuple(x.shape)}")
    if x.shape[0] == 0:
        raise ValueError("spherical_kmeans received an empty tensor")
    points = F.normalize(x.float(), dim=-1)
    k = max(1, min(int(k), int(points.shape[0])))
    centers = [points[0]]
    for _ in range(1, k):
        distance = torch.stack([1.0 - points @ center for center in centers], dim=1).min(dim=1).values
        centers.append(points[int(distance.argmax())])
    centers_tensor = torch.stack(centers)
    for _ in range(max(1, int(iters))):
        assignment = (points @ centers_tensor.T).argmax(dim=1)
        updated = centers_tensor.clone()
        for index in range(k):
            members = points[assignment == index]
            if len(members):
                updated[index] = F.normalize(members.mean(dim=0), dim=-1)
        if torch.allclose(updated, centers_tensor, atol=1.0e-5):
            break
        centers_tensor = updated
    return F.normalize(centers_tensor, dim=-1)
