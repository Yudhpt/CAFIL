#!/usr/bin/env python3
"""Infer the two Stage-I artifacts consumed by CAFIL Stage II.

The command loads a reviewed Stage-I checkpoint, clusters frozen slot features
into a global concept dictionary, and writes exactly two arrays:

* ``P.npy``: per-image soft concept assignments, shape ``[N, R]``.
* ``consscore.npy``: per-image pi-consensus score, shape ``[N]``.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataloader.stage1 import build_dataloader
from modules.s_kmeans import spherical_kmeans
from utils.stage1 import _build_model_cfg, get_device, load_cfg, set_seed
from train.stage1_model import DinoSlotStage1


def _log(message: str) -> None:
    """Print a stable, flushed pipeline log message."""
    print(f"[concept_infer] {message}", flush=True)


def _load_yaml_defaults(path: str) -> dict[str, Any]:
    """Read optional CLI defaults from a concept-inference YAML file."""
    if not path.strip():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when --config is supplied") from exc
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"--config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"--config must contain a YAML mapping: {config_path}")
    return payload


def _parse_args() -> argparse.Namespace:
    """Parse the minimal Stage-I-to-Stage-II concept inference contract."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    known, _ = pre.parse_known_args()
    defaults = _load_yaml_defaults(known.config)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=str(known.config), help="Optional concept-inference defaults YAML.")
    parser.add_argument("--ckpt", required=True, help="Reviewed Stage-I checkpoint.")
    parser.add_argument("--config-name", default=defaults.get("config_name", "stage1"))
    parser.add_argument("--prototype-split", default=defaults.get("prototype_split", "train"))
    parser.add_argument("--eval-split", default=defaults.get("eval_split", "train"))
    parser.add_argument("--output-dir", default=defaults.get("output_dir", ""))
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--num-prototypes", type=int, default=int(defaults.get("num_prototypes", defaults.get("R", 8))))
    parser.add_argument("--temperature", type=float, default=float(defaults.get("temperature", 0.1)))
    parser.add_argument("--max-prototype-samples", type=int, default=int(defaults.get("max_prototype_samples", 0)))
    parser.add_argument("--max-eval-samples", type=int, default=int(defaults.get("max_eval_samples", 0)))
    parser.add_argument("--pipeline-random-state", type=int, default=int(defaults.get("pipeline_random_state", 42)))
    parser.add_argument("--density-k-frac", type=float, default=float(defaults.get("density_k_frac", 0.05)))
    parser.add_argument("--density-k-min", type=int, default=int(defaults.get("density_k_min", 10)))
    parser.add_argument("--concept-dict-backend", choices=["torch", "faiss_spherical"], default=str(defaults.get("concept_dict_backend", "torch")))
    parser.add_argument("--density-backend", choices=["exact_js", "faiss_hellinger_gpu", "torch_hellinger_gpu"], default=str(defaults.get("density_backend", "exact_js")))
    parser.add_argument("--density-chunk", type=int, default=int(defaults.get("density_chunk", 4096)))
    args, overrides = parser.parse_known_args()
    args.overrides = list(overrides)
    return args


@torch.inference_mode()
def _collect(model: DinoSlotStage1, loader: Any, device: torch.device, *, max_samples: int, desc: str) -> dict[str, torch.Tensor]:
    """Collect the slots, class labels, and frozen-probe NLL used by pi-consensus."""
    model.eval()
    collected_slots, collected_labels, collected_nll = [], [], []
    total = 0
    for batch in tqdm(loader, desc=desc, dynamic_ncols=True, leave=False):
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        output = model(images, labels)
        take = int(images.shape[0]) if max_samples <= 0 else min(int(images.shape[0]), max(0, max_samples - total))
        if take <= 0:
            break
        collected_slots.append(output.slots[:take].cpu())
        collected_labels.append(labels[:take].cpu())
        collected_nll.append(F.cross_entropy(output.logits[:take], labels[:take], reduction="none").cpu())
        total += take
        if max_samples > 0 and total >= max_samples:
            break
    if not collected_slots:
        raise RuntimeError("Concept inference collected no samples")
    return {"slots": torch.cat(collected_slots), "labels": torch.cat(collected_labels), "probe_nll": torch.cat(collected_nll)}


def _per_class_zscore(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Standardize a scalar signal within each class before consensus fusion."""
    result = np.zeros_like(values, dtype=np.float64)
    for class_id in np.unique(labels):
        index = np.where(labels == class_id)[0]
        class_values = values[index].astype(np.float64)
        result[index] = (class_values - class_values.mean()) / (class_values.std() + 1.0e-8)
    return result.astype(np.float32)


def _output_dir(args: argparse.Namespace) -> Path:
    """Use an explicit output directory or a stable directory beside the checkpoint."""
    if str(args.output_dir).strip():
        return Path(args.output_dir).expanduser().resolve()
    return Path(args.ckpt).expanduser().resolve().parent / "concept_infer_outputs"
def _faiss_spherical_kmeans(x: torch.Tensor, k: int, *, random_state: int, iters: int = 30, nredo: int = 3) -> torch.Tensor:
    """FAISS 球面 KMeans：从 slot 向量聚类得到全局原型中心 U。"""
    try:
        import faiss
    except Exception as exc:
        raise RuntimeError("FAISS spherical KMeans requested but faiss is unavailable.") from exc

    if x.ndim != 2:
        raise ValueError(f"_faiss_spherical_kmeans expects [N,D], got {tuple(x.shape)}")
    x_n = F.normalize(x.float().cpu(), dim=-1)
    arr = np.ascontiguousarray(x_n.numpy().astype(np.float32, copy=False))
    if arr.shape[0] == 0:
        raise ValueError("_faiss_spherical_kmeans received an empty tensor.")
    k = max(1, min(int(k), int(arr.shape[0])))
    km = faiss.Kmeans(
        d=int(arr.shape[1]),
        k=int(k),
        niter=max(int(iters), 1),
        nredo=max(int(nredo), 1),
        seed=int(random_state),
        spherical=True,
        gpu=False,
    )
    km.train(arr)
    centers = torch.from_numpy(np.asarray(km.centroids, dtype=np.float32).reshape(int(k), int(arr.shape[1])))
    return F.normalize(centers, dim=-1).cpu()


def _global_concept_dictionary(
    slots: torch.Tensor,
    *,
    num_prototypes: int,
    backend: str = "torch",
    random_state: int = 0,
) -> torch.Tensor:
    """构建全局 concept 字典 U：对所有 slot 向量做球面 KMeans（torch 或 faiss 后端）。"""
    slots_n = F.normalize(slots.float().cpu().reshape(-1, int(slots.shape[-1])), dim=-1)
    backend_key = str(backend).strip().lower()
    if backend_key == "faiss_spherical":
        return _faiss_spherical_kmeans(slots_n, int(num_prototypes), random_state=int(random_state)).cpu()
    return spherical_kmeans(slots_n, int(num_prototypes)).cpu()


def _global_prototype_distribution(slots: torch.Tensor, prototypes: torch.Tensor, *, temperature: float) -> np.ndarray:
    """计算每样本对 U 的软分配 P_i：slot-原型相似度 softmax 后在 slot 维平均，再行归一化。"""
    slot_scores = _slot_concept_scores(slots, prototypes, temperature=temperature)
    prob = slot_scores.mean(dim=1)
    P = prob.numpy().astype(np.float32)
    row_sum = P.sum(axis=1, keepdims=True).clip(min=1.0e-12)
    return (P / row_sum).astype(np.float32)


def _slot_concept_scores(
    slots: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """返回 Eq.3 的逐 slot concept 概率 ``[N,K,R]``，不改变 P 的正式计算。"""
    slots_n = F.normalize(slots.float().cpu(), dim=-1)
    proto = F.normalize(prototypes.float().cpu(), dim=-1)
    tau = max(float(temperature), 1.0e-6)
    sim = torch.einsum("nkd,rd->nkr", slots_n, proto)
    return torch.softmax(sim / tau, dim=-1)


def _knn_radius_density_from_p(
    P: np.ndarray,
    labels: np.ndarray,
    *,
    k_frac: float,
    k_min: int,
    backend: str = "exact_js",
    chunk: int = 4096,
    eps: float = 1.0e-12,
    k_rule: str = "linear",
    k_alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """类内 kNN 密度半径：在 P 空间度量样本偏离度，得到 ``d_img`` 与 ``r_k``。

    算法（每个类别 c 独立）
    --------------------
    1. 取类 c 内所有样本的 P 向量（概率单纯形上的点）；
    2. 对每个样本 i，在类内找 k_c 个最近邻（距离可用 JS / Hellinger）；
    3. ``r_k[i]`` = 到第 k_c 近邻的距离（k-th neighbor radius）；
    4. ``d_img[i]`` = 类内所有 j 到 i 的距离的某种聚合（实现见各 backend），
       反映 i 在概念分配空间中的「稀疏/偏离」程度。

    k 的选择（论文规则）
    -------------------
    ``k_c = ceil(k_frac * n_c)``，再 clamp 到 ``[k_min, n_c - 1]``。

    返回值
    ------
    out_d : [N]  密度偏离度（越大 → 越可能是小簇/离群）
    out_r : [N]  每样本 k-th 近邻半径（诊断用，写入 r_k.npy）

    后端
    ----
    - ``exact_js``：CPU 精确 Jensen-Shannon 距离（默认，可复现）
    - ``torch_hellinger_gpu`` / ``faiss_hellinger_gpu``：GPU 加速变体
    """
    backend_key = str(backend).strip().lower()
    x = np.asarray(P, dtype=np.float64).clip(min=eps, max=1.0)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    out_d = np.zeros(x.shape[0], dtype=np.float64)
    out_r = np.zeros(x.shape[0], dtype=np.float64)

    def _compute_k_c(n_c: int) -> int:
        """论文线性 k 规则：ceil(k_frac * n_c)，夹在 [k_min, n_c-1]。"""
        k = int(np.ceil(float(k_frac) * int(n_c)))
        return min(max(int(k_min), k), int(n_c) - 1)

    if backend_key == "torch_hellinger_gpu":
        # GPU 分块 Hellinger 距离，按类独立计算 k-th 近邻半径
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p_sqrt = np.sqrt(np.clip(np.asarray(P, dtype=np.float32), 1.0e-12, 1.0))
        chunk_size = max(1, int(chunk))
        det_enabled = bool(torch.are_deterministic_algorithms_enabled())
        warn_only = bool(torch.is_deterministic_algorithms_warn_only_enabled()) if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled") else False
        if device.type == "cuda" and det_enabled:
            torch.use_deterministic_algorithms(False)
        try:
            for c in tqdm(np.unique(y).tolist(), desc="concept knn radius", dynamic_ncols=True, leave=False):
                idx = np.where(y == int(c))[0]
                n_c = int(idx.shape[0])
                if n_c <= 1:
                    continue
                k_c = _compute_k_c(int(n_c))
                pc = torch.from_numpy(np.ascontiguousarray(p_sqrt[idx])).to(device=device, dtype=torch.float32, non_blocking=True)
                pc_sq = (pc * pc).sum(dim=1)
                d_c = torch.empty(n_c, device=device, dtype=torch.float32)
                kth_k = min(int(k_c) + 1, n_c)
                for start in range(0, n_c, chunk_size):
                    end = min(start + chunk_size, n_c)
                    q = pc[start:end]
                    q_sq = pc_sq[start:end]
                    dist2 = q_sq[:, None] + pc_sq[None, :] - 2.0 * (q @ pc.T)
                    dist2 = dist2.clamp_(min=0.0)
                    kth_val, _ = torch.kthvalue(dist2, kth_k, dim=1)
                    d_c[start:end] = kth_val.sqrt_()
                r_k = d_c.clamp_(min=float(eps)).detach().cpu().numpy().astype(np.float64, copy=False)
                out_r[idx] = r_k
                out_d[idx] = np.log(r_k.clip(min=eps))
                del pc, pc_sq, d_c
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            if device.type == "cuda" and det_enabled:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
        return out_d.astype(np.float64), out_r.astype(np.float64)

    if backend_key == "faiss_hellinger_gpu":
        try:
            import faiss
        except Exception as exc:
            raise RuntimeError("FAISS GPU density backend requested but faiss is unavailable.") from exc

        if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "GpuIndexFlatL2"):
            raise RuntimeError("Current faiss build does not expose GPU flat L2 index.")

        for c in tqdm(np.unique(y).tolist(), desc="concept knn radius", dynamic_ncols=True, leave=False):
            idx = np.where(y == int(c))[0]
            if idx.shape[0] <= 1:
                continue
            cls_x = np.sqrt(np.clip(x[idx], eps, 1.0)).astype(np.float32, copy=False)
            cls_x = np.ascontiguousarray(cls_x)
            k_c = _compute_k_c(int(idx.shape[0]))
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, int(cls_x.shape[1]))
            index.add(cls_x)
            D, _ = index.search(cls_x, int(k_c) + 1)
            r_k = np.sqrt(np.clip(D[:, int(k_c)], a_min=0.0, a_max=None)).astype(np.float64, copy=False)
            out_r[idx] = r_k
            out_d[idx] = np.log(r_k.clip(min=eps))
        return out_d.astype(np.float64), out_r.astype(np.float64)

    chunk = 128
    dim = int(x.shape[1])
    log_x = np.log(x)

    # exact_js：精确 Jensen-Shannon 散度，CPU 分块（默认后端，确定性最好）
    for c in tqdm(np.unique(y).tolist(), desc="concept knn radius", dynamic_ncols=True, leave=False):
        idx = np.where(y == int(c))[0]
        if idx.shape[0] <= 1:
            continue
        cls_x = x[idx]
        cls_log_x = log_x[idx]
        k_c = _compute_k_c(int(idx.shape[0]))
        kth = np.zeros(idx.shape[0], dtype=np.float64)
        for start in range(0, int(idx.shape[0]), chunk):
            end = min(start + chunk, int(idx.shape[0]))
            q = cls_x[start:end]
            q_log = cls_log_x[start:end]
            mix = 0.5 * (q[:, None, :] + cls_x[None, :, :])
            log_mix = np.log(mix.clip(min=eps))
            js_left = 0.5 * np.sum(q[:, None, :] * (q_log[:, None, :] - log_mix), axis=-1)
            js_right = 0.5 * np.sum(cls_x[None, :, :] * (cls_log_x[None, :, :] - log_mix), axis=-1)
            js = js_left + js_right
            row_ids = np.arange(start, end)
            js[np.arange(end - start), row_ids] = np.inf
            kth[start:end] = np.partition(js, k_c - 1, axis=1)[:, k_c - 1]
        out_r[idx] = kth
        out_d[idx] = float(dim) * np.log(kth.clip(min=eps))
    return out_d.astype(np.float64), out_r.astype(np.float64)




def main() -> None:
    """Run the reviewed pi-consensus chain and write only Stage-II inputs.

    The score is ``min(z(probe_nll), z(density(P)))`` within each class.  The
    class labels are used only while deriving this annotation-free score; they
    are never written to the artifact directory.
    """
    args = _parse_args()
    output_dir = _output_dir(args)
    protected = (output_dir / "P.npy", output_dir / "consscore.npy")
    if not args.overwrite_existing and any(path.exists() for path in protected):
        raise FileExistsError(f"Refusing to overwrite existing concept artifacts in {output_dir}")

    checkpoint = torch.load(str(args.ckpt), map_location="cpu")
    config = checkpoint.get("cfg") or load_cfg(str(args.config_name), args.overrides)
    config.setdefault("data", {})["sample"] = "simple"
    set_seed(int(args.pipeline_random_state))
    device = get_device(config)
    _log(f"loading Stage-I checkpoint: {args.ckpt}")
    model = DinoSlotStage1(_build_model_cfg(config)).to(device)
    mismatch = model.load_state_dict(checkpoint["model"], strict=True)
    if mismatch.missing_keys or mismatch.unexpected_keys:
        raise RuntimeError(f"Stage-I checkpoint mismatch: {mismatch}")

    prototype = _collect(model, build_dataloader(config, split=str(args.prototype_split), is_train=False), device, max_samples=int(args.max_prototype_samples), desc="concept dictionary")
    evaluation = _collect(model, build_dataloader(config, split=str(args.eval_split), is_train=False), device, max_samples=int(args.max_eval_samples), desc="concept assignments")
    _log(f"collected dictionary={len(prototype['labels'])} assignments={len(evaluation['labels'])}")

    dictionary = _global_concept_dictionary(prototype["slots"], num_prototypes=int(args.num_prototypes), backend=str(args.concept_dict_backend), random_state=int(args.pipeline_random_state))
    assignments = _global_prototype_distribution(evaluation["slots"], dictionary, temperature=float(args.temperature))
    density, _ = _knn_radius_density_from_p(assignments, evaluation["labels"].numpy(), k_frac=float(args.density_k_frac), k_min=int(args.density_k_min), backend=str(args.density_backend), chunk=int(args.density_chunk))
    score = np.minimum(_per_class_zscore(evaluation["probe_nll"].numpy(), evaluation["labels"].numpy()), _per_class_zscore(density, evaluation["labels"].numpy())).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "P.npy", assignments.astype(np.float32))
    np.save(output_dir / "consscore.npy", score)
    _log(f"wrote P.npy {assignments.shape} and consscore.npy {score.shape} -> {output_dir}")


if __name__ == "__main__":
    main()
