"""Frozen local DINOv2 patch-token encoder used by CAFIL Stage I.

CAFIL uses DINO only as a fixed semantic target for Slot Attention. This
module deliberately exposes one operation, :meth:`DinoTeacher.patch_tokens`;
foreground masks, part proposals, and attention diagnostics are not part of
the released method.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="xFormers is not available.*", category=UserWarning)


class DinoTeacher(nn.Module):
    """Load a local DINOv2 model and return frozen patch features.

    Args:
        model_name: DINOv2 hub entrypoint, for example ``dinov2_vitb14``.
        dino_img_size: Square resolution used by the pretrained DINO model.
        torch_home: Directory containing the local torch hub checkout and
            checkpoint cache.
        local_only: Reject a missing local model instead of downloading it.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        dino_img_size: int = 224,
        torch_home: str | None = None,
        local_only: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.dino_img_size = int(dino_img_size)
        self.patch_size = 14

        home = Path(torch_home) if torch_home else Path.home() / ".cache" / "torch"
        home.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(home)
        local_repo = home / "hub" / "facebookresearch_dinov2_main"
        if local_only and not local_repo.exists():
            raise FileNotFoundError(f"Local DINO repository not found: {local_repo}")

        checkpoint_names = {
            "dinov2_vits14": "dinov2_vits14_pretrain.pth",
            "dinov2_vitb14": "dinov2_vitb14_pretrain.pth",
            "dinov2_vitl14": "dinov2_vitl14_pretrain.pth",
            "dinov2_vitg14": "dinov2_vitg14_pretrain.pth",
        }
        checkpoint_name = checkpoint_names.get(self.model_name)
        checkpoint_path = home / "hub" / "checkpoints" / str(checkpoint_name)
        if local_only and checkpoint_name and not checkpoint_path.exists():
            raise FileNotFoundError(f"Local DINO checkpoint not found: {checkpoint_path}")

        self.dino = torch.hub.load(str(local_repo), self.model_name, source="local")
        self.dino.eval()
        for parameter in self.dino.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return DINO patch embeddings with shape ``[B, N, D]``.

        ``images`` must already use DINO/ImageNet normalization. Inputs are
        resized only when needed, so Stage I accepts dataset-native image
        resolutions while DINO still sees its pretrained grid.
        """
        if images.shape[-2:] != (self.dino_img_size, self.dino_img_size):
            images = F.interpolate(images, size=(self.dino_img_size, self.dino_img_size), mode="bilinear", align_corners=False)
        features = self.dino.forward_features(images)
        if isinstance(features, dict) and "x_norm_patchtokens" in features:
            return features["x_norm_patchtokens"]
        return self.dino.get_intermediate_layers(images, n=1)[0][:, 1:, :]
