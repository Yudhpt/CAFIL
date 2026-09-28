r"""Slot Attention 模块 — CAFIL Stage-I 的区域分解核心。

## 在 CAFIL 管线中的位置

**Stage-I 训练（``train/stage1_model.py`` → ``DinoSlotStage1``）**
    1. 冻结 DINOv2 提取 patch token：``[B, N, D_dino]``（默认 N=196, D_dino=768）。
    2. ``token_proj`` 线性投影到 slot 维度：``[B, N, D_slot]``（默认 D_slot=256）。
    3. **本模块** 将 N 个 patch token 竞争性地分配给 K 个 slot（默认 K=8），输出：
       - ``slots [B, K, D_slot]``：每个 slot 的区域语义向量；
       - ``attn_own [B, K, N]``：token ownership（沿 slot 维 softmax，每个 token 的 K 路权重和为 1）。
    4. ``recon_head(slots)`` + ``attn_own`` 加权重建 DINO token → ``loss_recon``（余弦距离）。
    5. ``attn_own`` 还参与 ``loss_overlap``（slot 间空间重叠惩罚）与 ``loss_entropy``（鼓励 slot 聚焦）。

**concept_infer（``train/concept_infer.py``）**
    - 加载 Stage-I checkpoint 后调用 ``model(images)``，收集 ``out.slots`` 与 ``out.slot_masks``（即 ``attn_own``）。
    - ``slots`` 经全局球面 k-means 得到概念字典 U，再 softmax 分配得到样本级概念分布 ``P [N, R]``；
      该 P 是 Stage-II 的 ``P.npy`` 输入，**不经过** SlotAttention 的参数再训练。
    - ``slot_masks`` 可用于可视化 slot 空间覆盖，以及 ``s_kmeans`` 的注意力熵过滤。

## 张量形状约定

| 符号 | 含义 | 典型值 |
|------|------|--------|
| B | batch size | 任意 |
| N | patch token 数（H×W） | 196 (=14×14) |
| K | slot 数 | 8 |
| D | slot / token 特征维 | 256 |

## 两种 attention 归一化

同一组 logits ``dots[b,k,n] = q(slot_k)·k(token_n)/√D`` 派生两种语义：

- **attn_own** = softmax(dots, dim=slot)：解释「第 n 个 patch 属于哪个 slot」→ 用于重建、overlap/entropy 损失、外部 prior。
- **attn_read** = normalize(attn_own, dim=token)：解释「slot k 从哪些 patch 读取信息」→ 仅用于 GRU 更新，保证每个 slot 的 update 量级稳定。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SlotAttentionConfig:
    """Slot Attention 超参；由 ``DinoSlotStage1Config.slot_attention`` 传入。

    Attributes:
        num_iterations: 竞争-更新循环次数；越多 slot 越稳定，但算力线性增加。
        epsilon: ``attn_read`` 归一化时分母的下界，防止除零。
        mlp_hidden_size: slot 更新后 MLP 残差分支的隐层宽度。
        qkv_bias: Q/K/V 线性层是否带 bias。
        slot_init_scale: ``slot_init_mode=random`` 时高斯噪声的缩放系数。
        slot_init_mode: ``random``/``gaussian`` 训练时加噪、推理时确定性；``learned_query`` 始终用可学习均值。
    """

    num_iterations: int = 3
    epsilon: float = 1.0e-8
    mlp_hidden_size: int = 512
    qkv_bias: bool = False
    slot_init_scale: float = 1.0
    slot_init_mode: str = "random"


class SlotAttention(nn.Module):
    """迭代式 Slot Attention：将 patch token 序列分解为 K 个可竞争的区域 slot。

    参考 Locatello et al. (2020) 的对象中心表征思路；CAFIL 将其用于**无监督区域分解**，
    而非生成式建模。可训练参数：Q/K/V 投影、GRU、MLP、可学习 slot 初始化 ``slots_mu/log_sigma``。

    输入:
        tokens: ``[B, N, D]``，Stage-I 中为 ``token_proj(dino_tokens)``。

    输出:
        slots: ``[B, K, D]`` — 每个 slot 的区域语义 embedding，供 ``recon_head`` 与 concept_infer 使用。
        attn_own: ``[B, K, N]`` — token ownership；``sum_k attn_own[b,k,n] = 1``。
        attn_read: ``[B, K, N]`` — slot 读取权重；``sum_n attn_read[b,k,n] = 1``（仅内部 GRU 用）。
    """

    def __init__(self, num_slots: int, slot_dim: int, cfg: SlotAttentionConfig | None = None) -> None:
        super().__init__()
        self.num_slots = int(num_slots)  # K：concept_infer 中 ``num_slots`` 写入 summary.json
        self.slot_dim = int(slot_dim)    # D：与 token_proj 输出维一致
        self.cfg = cfg or SlotAttentionConfig()

        # 三路 LayerNorm：分别归一化输入 token、当前 slot 状态、MLP 前的 slot。
        self.norm_inputs = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        # Cross-attention 投影：K/V 来自 patch token，Q 来自 slot 查询向量。
        # 这与标准 Transformer 自注意力不同——slot 与 token 角色不对称。
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=bool(self.cfg.qkv_bias))
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=bool(self.cfg.qkv_bias))
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=bool(self.cfg.qkv_bias))

        # 每轮迭代：attn_read 加权 V → GRU 融合历史 slot → MLP 残差修正。
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, int(self.cfg.mlp_hidden_size)),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.cfg.mlp_hidden_size), slot_dim),
        )

        # 可学习 slot 先验：mu 为各 slot 的初始中心，log_sigma 控制训练期随机扰动幅度。
        self.slots_mu = nn.Parameter(torch.empty(1, self.num_slots, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.empty(1, self.num_slots, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_log_sigma)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """执行迭代式 slot-token 竞争。

        这里显式拆开两种 attention 语义，但二者都来自同一个 ``dots=q(slot)·k(token)``：

        - ``attn_own = softmax(dots, dim=slot)``：每个 token 分给哪个 slot。后续 DINO prior、
          env proxy、重建和可视化都使用它，因此空间约束会直接回传到同一组 slot query / token key。
        - ``attn_read``：把 ``attn_own`` 在 token 维归一化，得到每个 slot 读取哪些 token 的分布。
          GRU 更新只用这张图，保证每个 slot 的 update 量级稳定。

        这不是两套独立 mask head；它只是同一 logits 的两种归一化视图。
        """
        if tokens.dim() != 3:
            raise ValueError(f"SlotAttention expects tokens [B,N,D], got shape={tuple(tokens.shape)}")
        batch_size, _, dim = tokens.shape
        if dim != self.slot_dim:
            raise ValueError(f"Token dim mismatch: expected {self.slot_dim}, got {dim}")

        # Step 0：对全部 patch token 做一次 K/V 投影（迭代内复用，减少计算）。
        normalized_tokens = self.norm_inputs(tokens)       # [B, N, D]
        key_tokens = self.to_k(normalized_tokens)          # [B, N, D]
        value_tokens = self.to_v(normalized_tokens)        # [B, N, D]

        # Step 1：初始化 K 个 slot 向量；训练期随机模式有助于打破对称、避免 slot collapse。
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)  # [B, K, D]
        init_mode = str(getattr(self.cfg, "slot_init_mode", "random")).strip().lower()
        if init_mode in {"learned_query", "query", "deterministic"}:
            slots = mu
        elif init_mode in {"random", "gaussian"}:
            sigma = self.slots_log_sigma.exp().expand(batch_size, self.num_slots, -1)
            init_noise = torch.randn_like(mu) if self.training else torch.zeros_like(mu)
            slots = mu + sigma * init_noise * float(self.cfg.slot_init_scale)
        else:
            raise ValueError(f"Unsupported slot_init_mode: {self.cfg.slot_init_mode}")

        # 占位；最后一轮迭代的 attention 即为返回值。
        attn_own = tokens.new_zeros(batch_size, self.num_slots, tokens.shape[1])   # [B, K, N]
        attn_read = tokens.new_zeros(batch_size, self.num_slots, tokens.shape[1])  # [B, K, N]
        scale = self.slot_dim**-0.5  # scaled dot-product，与 Transformer 一致

        for _ in range(int(self.cfg.num_iterations)):
            # --- 竞争阶段 ---
            slots_prev = slots
            slot_queries = self.to_q(self.norm_slots(slots))  # [B, K, D]
            dots = torch.einsum("bkd,bnd->bkn", slot_queries, key_tokens) * scale  # [B, K, N]

            # attn_own[b,k,n]：patch n 被 slot k「占有」的概率；沿 k 维 softmax → 每个 token 的 K 路权重和为 1。
            # Stage-I 的 recon / overlap / entropy 损失、以及 concept_infer 的 slot_mean 聚合都依赖此张量。
            attn_own = F.softmax(dots, dim=1)

            # attn_read[b,k,n]：slot k 从 patch n 读取信息的权重；沿 n 维归一化 → 每个 slot 的 N 路权重和为 1。
            # 梯度仍通过 attn_own → dots 回传；read 只是 renormalize，无额外参数。
            attn_read = attn_own + float(self.cfg.epsilon)
            attn_read = attn_read / attn_read.sum(dim=-1, keepdim=True).clamp_min(float(self.cfg.epsilon))

            # --- 聚合阶段 ---
            updates = torch.einsum("bkn,bnd->bkd", attn_read, value_tokens)  # [B, K, D]

            # --- 更新阶段 ---
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(batch_size, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))  # 残差 MLP 细化 slot 表征

        # slots → recon_head → concept_infer 的 U/P；attn_own → stage1 空间正则与可视化。
        return slots, attn_own, attn_read
