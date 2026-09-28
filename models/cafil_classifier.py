"""
CAFIL Stage-II 图像分类器
=========================

定义 Stage-II 使用的 **纯图像分类模型** ``CAFILImageClassifier``，以及从 YAML
配置实例化的工厂函数 ``build_cafil_classifier_from_config``。

流水线位置
----------
- **阶段**：Stage-II 训练与推理

设计说明
--------
"""
from __future__ import annotations

import torch
from torch import nn

from models.visual_backbone import build_visual_backbone


class CAFILImageClassifier(nn.Module):
    """
    CAFIL Stage-II 图像分类器（ResNet 骨干）。

    ----------
    backbone_name : str
        见 ``build_visual_backbone``。
    num_classes : int
        分类类别数。
    pretrained : bool
        是否加载 ImageNet 预训练（训练初值；从 checkpoint 恢复时由外层控制）。
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        backbone, feat_dim = build_visual_backbone(
            backbone_name,
            int(num_classes),
            pretrained=bool(pretrained),
        )
        self.feat_dim = feat_dim
        self.backbone = backbone

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取 ResNet 全局平均池化后的特征向量 φ（不经过 fc 分类层）。

        手动展开 ``forward`` 而非 ``backbone(x)`` 的原因
        -----------------------------------------------
        标准 ResNet 路径：conv1→bn→relu→maxpool→layer1-4→avgpool→flatten。

        形状
        ----
        输入 ``x``：[B, 3, H, W]（通常 H=W=224）
        输出 ``feat``：[B, feat_dim]（ResNet-50 为 2048）
        """
        # stem：7×7 conv + BN + ReLU + 3×3 maxpool，空间尺寸 /4
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        # 四个残差 stage；layer4 输出 [B, 2048, H/32, W/32]
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # 全局平均池化 → [B, 2048, 1, 1]，flatten 为 [B, 2048]
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        ----------
        images : Tensor [B, 3, H, W]
            输入图像 batch。

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(logits, features)``，其中 ``features`` 是全局池化后的 φ。
        """
        feat = self._features(images)
        logits = self.backbone.fc(feat)  # 标准 ImageNet fc：2048 → num_classes
        return logits, feat


def build_cafil_classifier_from_config(cfg: dict, *, pretrained: bool = False) -> CAFILImageClassifier:
    """
    从 YAML 配置 dict 构建 ``CAFILImageClassifier``。

    ----------
    cfg : dict
        需含 ``model`` 段（``backbone``、``num_classes``）。
    pretrained : bool
        是否 ImageNet 预训练；从 checkpoint 加载权重时通常为 False。

    Returns
    -------
    CAFILImageClassifier
        已按配置组装好的模型实例。

    Notes
    -----
    模型不读取 ``loss``：CAFIL 训练目标与图像模型显式分离。
    """
    model_cfg = cfg["model"]
    return CAFILImageClassifier(
        backbone_name=str(model_cfg["backbone"]),
        num_classes=int(model_cfg["num_classes"]),
        pretrained=bool(pretrained),
    )
