"""
视觉骨干网络构建（Stage-II）
============================

为 CAFIL Stage-II 分类器提供 **torchvision ResNet** 骨干 + 线性分类头。
Stage-I 使用 DINO，Stage-II 默认切换到 ResNet18/50 做最终任务学习。

流水线位置
----------
- **阶段**：Stage-II 模型初始化
- **输入**：骨干名称（``resnet18`` / ``resnet50``）、类别数、是否 ImageNet 预训练
- **输出**：``(backbone_module, feat_dim)``，其中 ``backbone.fc`` 已替换为 ``num_classes`` 维线性层

本模块直接返回 CAFIL Stage-II 所需的分类器骨干与特征维度，不暴露额外的实验接口。
"""
from __future__ import annotations

from torch import nn
from torchvision import models


def build_visual_backbone(
    name: str,
    num_classes: int,
    *,
    pretrained: bool = True,
) -> tuple[nn.Module, int]:
    """
    构造带线性分类头的 torchvision ResNet 骨干。

    Parameters
    ----------
    name : str
        ``"resnet18"`` 或 ``"resnet50"``（大小写不敏感）。
    num_classes : int
        分类类别数（Waterbirds/CelebA 等为 2）。
    pretrained : bool, default True
        是否加载 ImageNet 预训练权重；推理加载已有 checkpoint 时通常设为 False。

    Returns
    -------
    tuple[nn.Module, int]
        - ``backbone``：完整 ResNet，``fc`` 层输出 ``num_classes`` 维 logits 所需维度
        - ``feat_dim``：全局平均池化后特征维度（ResNet18=512，ResNet50=2048）

    Raises
    ------
    ValueError
        不支持的 ``name``。

    Notes
    -----
    ``CAFILImageClassifier`` 会手动走 conv→layer4→avgpool 提取特征，再调用 ``backbone.fc``；
    因此这里保留标准 ResNet 结构，仅替换最后一层 ``fc`` 的 out_features。
    """
    backbone_key = str(name).lower()
    if backbone_key == "resnet50":
        # ResNet50 使用 IMAGENET1K_V2 预训练权重
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
    elif backbone_key == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    # 记录原 fc 输入维度
    feat_dim = int(backbone.fc.in_features)
    # 将 ImageNet 1000 类头替换为任务相关 num_classes 类
    backbone.fc = nn.Linear(feat_dim, int(num_classes))
    return backbone, feat_dim
