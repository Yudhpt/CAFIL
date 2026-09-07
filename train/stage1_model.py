r"""CAFIL Stage-I 训练模型：冻结 DINO + Slot Attention 区域分解。

## 管线定位

本模块是 **Stage-I 的唯一前向图**（``train/train_stage1.py`` 与 ``concept_infer.py`` 共用）。
目标：在无标注环境下，将每张图像的 DINO patch token 分解为 K 个可重建的 slot 区域单元，
为后续 concept_infer 导出 ``P.npy``（概念分布）与 ``consscore.npy``（共识分数）提供 slot 表征。

数据流::

    images [B,3,H,W]
      → DinoTeacher.patch_tokens (冻结) → dino_tokens [B,N,D_dino]
      → token_proj (可训练)              → slot_inputs [B,N,D_slot]
      → SlotAttention (可训练)           → slots [B,K,D_slot], slot_masks [B,K,N]
      → recon_head + einsum              → recon_tokens [B,N,D_dino]
      → mean-pool(dino_tokens).detach    → probe_features [B,D_dino] → classifier → logits

## 损失项（``DinoSlotStage1Output.losses``）

总损失::

    L = λ_recon·L_recon + λ_cls·L_cls + λ_overlap·L_overlap + λ_entropy·L_entropy

| 损失 | 公式含义 | 梯度流向 | 默认 λ |
|------|----------|----------|--------|
| **L_recon** | ``1 - cos_sim(recon_token, dino_token)`` 逐 patch 平均 | token_proj, slot_attention, recon_head | 1.0 |
| **L_cls** | 冻结 probe 上的 CE(logits, y)；probe = mean(dino_tokens).detach() | **仅 classifier**（不回传 slot） | 0.2 |
| **L_overlap** | slot_masks 归一化后的 Gram 矩阵非对角项均值；惩罚多 slot 覆盖同一 patch | slot_attention | 0.05 |
| **L_entropy** | 每个 slot 空间分布的归一化熵均值；**越低**表示 slot 越聚焦 | slot_attention | 0.02 |

**设计意图**：分类 probe 故意接在 **detach 的 DINO 特征** 上，使 cls 损失只训练轻量 classifier，
不干扰 slot 分解；slot 的质量由 recon + 空间正则驱动。这与论文 Stage-I probe 协议一致。

## concept_infer 中的复用

``concept_infer._collect`` 调用 ``model(images, y)`` 并收集：

- ``out.slots [N,K,D_slot]`` → 全局球面 k-means → 概念字典 U → 样本分布 P
- ``out.slot_masks [N,K,N_patch]`` → 可视化 / s_kmeans 的 v1 熵过滤
- ``out.logits`` → frozen-probe NLL，参与 pi_consensus 的 ``min(z_ell, z_d)`` 共识分数

## 张量形状（默认配置）

| 张量 | 形状 | 说明 |
|------|------|------|
| dino_tokens | [B, 196, 768] | DINOv2-ViT-B/14, patch 14×14 |
| slots | [B, 8, 256] | K=8 slot, D_slot=256 |
| slot_masks | [B, 8, 196] | 即 SlotAttention 的 attn_own |
| recon_tokens | [B, 196, 768] | slot 加权重建的 DINO 特征 |
| logits | [B, num_classes] | 默认 num_classes=2 |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import DinoTeacher, SlotAttention, SlotAttentionConfig


@dataclass
class DinoSlotStage1Config:
    """Stage-I 模型与损失权重配置；由 ``utils.stage1._build_model_cfg`` 从 YAML 解析。"""

    dino_model_name: str = "dinov2_vitb14"
    dino_img_size: int = 224          # DINO 输入边长；N_dino = (dino_img_size // 14)²
    dino_torch_home: str = str(Path.home() / ".cache" / "torch")
    dino_dim: int = 768               # DINOv2-ViT-B patch token 维度
    slot_dim: int = 256               # SlotAttention 工作维度
    num_slots: int = 8                # K；concept_infer summary 中的 num_slots
    num_classes: int = 2              # 数据集类别数
    slot_attention: SlotAttentionConfig = field(default_factory=SlotAttentionConfig)
    lambda_recon: float = 1.0         # 重建损失权重（主损失）
    lambda_cls: float = 0.2           # 冻结 probe 分类损失（仅训 classifier）
    lambda_overlap: float = 0.05      # slot 空间重叠惩罚
    lambda_entropy: float = 0.02      # slot 空间熵惩罚（鼓励 compact mask）
    use_mock_dino: bool = False       # True 时用 RGB 投影假 token，便于无 GPU/无权重调试


@dataclass
class DinoSlotStage1Output:
    """``DinoSlotStage1.forward`` 的完整输出；train_stage1 与 concept_infer 按需取用各字段。"""

    loss: torch.Tensor                # 加权总损失标量
    logits: torch.Tensor              # [B, C] 冻结 probe 分类 logits
    slots: torch.Tensor               # [B, K, D_slot] 区域 slot 向量 → concept_infer 建 U/P
    slot_masks: torch.Tensor          # [B, K, N] token ownership (= attn_own)
    probe_features: torch.Tensor      # [B, D_dino] mean-pool 的 DINO 特征（已 detach）
    recon_tokens: torch.Tensor        # [B, N, D_dino] slot 加权重建 token
    dino_tokens: torch.Tensor         # [B, N, D_dino] 冻结 DINO 目标
    losses: Dict[str, torch.Tensor]   # 分项损失字典，含 loss_total/recon/cls/overlap/entropy
    aux: Dict[str, torch.Tensor]      # 诊断量：slot_eff_area, slot_peak, slot_entropy, acc


class _MockDinoTokens(nn.Module):
    """无 DINO 权重时的占位教师：将 resize 后的 RGB patch 线性投影为假 token。

    仅用于 ``use_mock_dino=True`` 的单元测试/CI；不参与正式训练复现。
    """

    def __init__(self, token_dim: int, n_tokens: int = 196) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.n_tokens = int(n_tokens)
        self.proj = nn.Linear(3, self.token_dim)

    @torch.no_grad()
    def patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        bsz = int(images.shape[0])
        side = int(round(float(self.n_tokens) ** 0.5))
        x = F.interpolate(images, size=(side, side), mode="bilinear", align_corners=False)
        x = x.flatten(2).transpose(1, 2)  # [B, N, 3]
        return self.proj(x).detach().reshape(bsz, self.n_tokens, self.token_dim)


class DinoSlotStage1(nn.Module):
    """CAFIL Stage-I：冻结 DINOv2 patch token + 可训练 Slot Attention 区域分解。

    训练目标：slot 应能重建 DINO 语义（L_recon），且各 slot 空间上互不重叠、分布紧凑。
    分类 probe 接在 detach 的 DINO 池化特征上，用于 concept_infer 导出 probe NLL，
    但不驱动 slot 学习——避免 shortcut 分类器污染无监督区域分解。
    """

    def __init__(self, cfg: DinoSlotStage1Config) -> None:
        super().__init__()
        self.cfg = cfg
        # --- 冻结 DINO 教师：仅提供 patch token 目标，参数不参与优化 ---
        if bool(cfg.use_mock_dino):
            self.dino = _MockDinoTokens(token_dim=int(cfg.dino_dim), n_tokens=196)
        else:
            self.dino = DinoTeacher(
                model_name=str(cfg.dino_model_name),
                dino_img_size=int(cfg.dino_img_size),
                torch_home=str(cfg.dino_torch_home),
                local_only=True,
            )
        for p in self.dino.parameters():
            p.requires_grad_(False)

        # D_dino → D_slot 线性投影 + LayerNorm
        self.token_proj = nn.Sequential(
            nn.LayerNorm(int(cfg.dino_dim)),
            nn.Linear(int(cfg.dino_dim), int(cfg.slot_dim)),
        )
        # 核心：K 个 slot 与 N 个 patch 竞争
        self.slot_attention = SlotAttention(
            num_slots=int(cfg.num_slots),
            slot_dim=int(cfg.slot_dim),
            cfg=cfg.slot_attention,
        )
        # slot 向量 → DINO 维，供重建损失
        self.recon_head = nn.Sequential(
            nn.LayerNorm(int(cfg.slot_dim)),
            nn.Linear(int(cfg.slot_dim), int(cfg.dino_dim)),
        )
        # 冻结 probe 分类头：接 detach 的 DINO 池化特征
        self.classifier = nn.Sequential(
            nn.LayerNorm(int(cfg.dino_dim)),
            nn.Linear(int(cfg.dino_dim), int(cfg.num_classes)),
        )
        # ImageNet 归一化常数（与 DINO 预训练一致）
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> "DinoSlotStage1":
        super().train(mode)
        # DINO 始终 eval（关闭 dropout 等）；即使 Stage-I 处于 train 模式
        self.dino.eval()
        return self

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """将输入图像按 ImageNet 均值/方差归一化，与 DINO 预训练输入分布对齐。"""
        return (images - self.image_mean.to(images.device)) / self.image_std.to(images.device)

    @staticmethod
    def _slot_overlap_loss(slot_masks: torch.Tensor) -> torch.Tensor:
        """Slot 空间重叠惩罚 L_overlap。

        先将每个 slot 的 attn_own 沿 patch 维归一化为概率分布 ``slot_mass[b,k,n]``，
        再计算 Gram 矩阵 ``G[b,k,k'] = Σ_n slot_mass[b,k,n]·slot_mass[b,k',n]``。
        取非对角项均值并除以 K(K-1) 归一化 → 鼓励不同 slot 覆盖不同 patch 区域。

        输入: slot_masks ``[B, K, N]``（即 attn_own，沿 slot 维已 softmax）
        输出: 标量损失（batch 均值）
        """
        slot_mass = slot_masks / slot_masks.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        gram = torch.bmm(slot_mass, slot_mass.transpose(1, 2))  # [B, K, K]
        num_slots = int(slot_masks.shape[1])
        eye = torch.eye(num_slots, device=slot_masks.device, dtype=slot_masks.dtype).unsqueeze(0)
        return (gram * (1.0 - eye)).sum(dim=(1, 2)).mean() / float(max(num_slots * (num_slots - 1), 1))

    @staticmethod
    def _slot_entropy(slot_masks: torch.Tensor) -> torch.Tensor:
        """Slot 空间分布熵 L_entropy（越低 → slot 越 compact / 越聚焦）。

        对每个 slot 的 ``slot_mass[b,k,:]`` 计算 Shannon 熵，再除以 ``log(N)`` 归一化到 [0,1]，
        最后对 batch 与 slot 取均值。训练时 **最小化** 此损失，使 slot 倾向于集中覆盖少量 patch。

        输入: slot_masks ``[B, K, N]``
        输出: 标量（batch 均值）
        """
        slot_mass = slot_masks / slot_masks.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        entropy = -(slot_mass * slot_mass.clamp_min(1.0e-8).log()).sum(dim=-1)  # [B, K]
        return (entropy / torch.log(slot_masks.new_tensor(float(slot_masks.shape[-1])).clamp_min(2.0))).mean()

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> DinoSlotStage1Output:
        """Stage-I 前向：DINO token 提取 → slot 分解 → 重建 + probe 分类 + 空间正则。

        Args:
            images: ``[B, 3, H, W]`` 输入图像（任意 H,W，DINO 内部会 resize）。
            labels: ``[B]`` 可选类别标签；为 None 时 L_cls=0。

        Returns:
            DinoSlotStage1Output，含总损失、slot 表征、mask、分项损失与诊断量。
        """
        # Step 1：冻结 DINO 提取 patch token（不参与 autograd）
        with torch.no_grad():
            dino_tokens = self.dino.patch_tokens(self._normalize_images(images)).detach()  # [B,N,D_dino]

        # Step 2：投影 + Slot Attention 分解
        slot_inputs = self.token_proj(dino_tokens)  # [B,N,D_slot]
        slots, slot_masks, _slot_read = self.slot_attention(slot_inputs)  # slots [B,K,D], masks [B,K,N]

        # Step 3：slot 加权重建 DINO token → L_recon（主监督信号）
        slot_recon = self.recon_head(slots)  # [B,K,D_dino]
        recon_tokens = torch.einsum("bkn,bkd->bnd", slot_masks, slot_recon)  # [B,N,D_dino]
        # 逐 patch 余弦距离：1 - cos_sim(recon, dino)，鼓励 slot 组合还原 DINO 语义
        recon_loss = 1.0 - (F.normalize(recon_tokens, dim=-1) * F.normalize(dino_tokens, dim=-1)).sum(dim=-1)
        loss_recon = recon_loss.mean()

        # Step 4：冻结 probe 分类 — probe_features 来自 detach 的 DINO 均值池化
        # 梯度 **不回传** token_proj / slot_attention（因 dino_tokens 已 detach）
        probe_features = dino_tokens.mean(dim=1).detach()  # [B,D_dino]
        logits = self.classifier(probe_features)  # [B,C]
        loss_cls = logits.new_zeros(())
        if labels is not None:
            loss_cls = F.cross_entropy(logits, labels)

        # Step 5：slot 空间正则 — 减少重叠、鼓励 compact
        loss_overlap = self._slot_overlap_loss(slot_masks)
        loss_entropy = self._slot_entropy(slot_masks)

        # Step 6：加权总损失
        loss = (
            float(self.cfg.lambda_recon) * loss_recon
            + float(self.cfg.lambda_cls) * loss_cls
            + float(self.cfg.lambda_overlap) * loss_overlap
            + float(self.cfg.lambda_entropy) * loss_entropy
        )

        # Step 7：无梯度诊断量（训练日志 / concept_infer 不使用，但 train_stage1 可记录）
        with torch.no_grad():
            slot_mass = slot_masks / slot_masks.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            entropy = -(slot_mass * slot_mass.clamp_min(1.0e-8).log()).sum(dim=-1)
            eff_area = (entropy.exp() / float(slot_masks.shape[-1])).mean()  # 有效覆盖面积估计
            peak = slot_mass.max(dim=-1).values.mean()  # mask 峰值强度
            pred = logits.argmax(dim=-1)
            acc = (pred == labels).float().mean() if labels is not None else logits.new_zeros(())
        return DinoSlotStage1Output(
            loss=loss,
            logits=logits,
            slots=slots,
            slot_masks=slot_masks,
            probe_features=probe_features,
            recon_tokens=recon_tokens,
            dino_tokens=dino_tokens,
            losses={
                "loss_total": loss,
                "loss_recon": loss_recon,
                "loss_cls": loss_cls,
                "loss_overlap": loss_overlap,
                "loss_entropy": loss_entropy,
            },
            aux={
                "slot_eff_area": eff_area.detach(),
                "slot_peak": peak.detach(),
                "slot_entropy": loss_entropy.detach(),
                "acc": acc.detach(),
            },
        )
