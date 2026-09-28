# CAFIL

> 中文在前，English version follows.

## 中文

CAFIL 是一个四阶段图像分类流程：先用冻结 DINO 与 Slot Attention 训练表征，再从 Stage I checkpoint 生成概念分配与一致性分数，最后训练并评估 CAFIL 图像分类器。训练主链不使用环境标签。

### 安装

先使用 [PyTorch selector](https://pytorch.org/get-started/locally/) 安装与本机 CPU/CUDA 匹配的 `torch` 和 `torchvision`，然后安装 CAFIL 的平台无关依赖：

```bash
pip install -r requirements.txt
```

CelebA 的正式 concept-inference 配置使用 GPU FAISS；仅在运行该配置时安装：

```bash
pip install -r requirements-faiss-gpu.txt
```

### 数据和输出

仓库不包含数据集、checkpoint、概念 artifact 或 DINOv2 权重。运行前设置以下三个绝对路径：

```bash
export CAFIL_DATA_ROOT=/absolute/path/to/datasets
export CAFIL_OUTPUT_ROOT=/absolute/path/to/cafil_outputs
export CAFIL_DINO_HOME=/absolute/path/to/torch_cache
```

`CAFIL_DINO_HOME` 必须包含本地 DINOv2 Torch Hub checkout 和相应 checkpoint；CAFIL 不会下载模型权重。每份正式 YAML 都通过这三个变量解析路径，变量缺失会立即报错。可从 `.env.example` 复制变量名。

### 快速开始

```bash
git clone https://github.com/Yudhpt/CAFIL.git
cd CAFIL
# 创建并激活 Python 环境，然后按本机平台安装 PyTorch。
pip install -r requirements.txt
cp .env.example .env.local
# 编辑 .env.local，然后加载到当前 shell。
set -a; source .env.local; set +a
```

在 shell 中设置以下四个变量；`.env.local` 仅供保存私有模板，CAFIL 不会自动读取它：

```bash
export CAFIL_DATA_ROOT=/absolute/path/to/datasets
export CAFIL_OUTPUT_ROOT=/absolute/path/to/cafil_outputs
export CAFIL_DINO_HOME=/absolute/path/to/torch_cache
export CAFIL_FORCE_CPU=0
```

DINOv2 必须预先放置在以下本地 Torch Hub 布局中。仓库在运行时不会下载代码或权重：

```text
<CAFIL_DINO_HOME>/
  hub/facebookresearch_dinov2_main/
  hub/checkpoints/dinov2_vitb14_pretrain.pth
```

### 数据目录

```text
<CAFIL_DATA_ROOT>/
  waterbirds/data/{train-00000-of-00001,validation-00000-of-00001,test-00000-of-00001}.parquet
  CelebA/{Img/img_align_celeba,Anno/list_attr_celeba.csv,Eval/list_eval_partition.txt}
  nico/NICO/multi_classification/{train,val,test}/
  metashift/{MetaShift-Cat-Dog-indoor-outdoor,metadata_metashift.csv}
```

下载 MetaShift 的四个源目录后，用以下命令生成 CSV：

```bash
python scripts/metashift/prepare.py --data-root "$CAFIL_DATA_ROOT"
```

Stage I 到 Stage II 的最小文件契约为：

```text
P.npy          # [N, K]，训练样本顺序上的概念分配
consscore.npy  # [N]，对应样本的一致性分数
sample_ids.npy # [N]，数据集相对路径的稳定 SHA-256 identity
labels.npy     # [N]，用于逐行核验的目标标签
stage1_artifacts_manifest.json  # 完整发布标记、shape、dtype、大小与 SHA-256
```

### 运行流程

每个数据集的输出均位于 `$CAFIL_OUTPUT_ROOT/cafil_best/<dataset>/`：Stage I checkpoint 为 `stage1/best_cafil_stage1.pth`，concept artifact 为 `concept/`，Stage II checkpoint 为 `stage2/`。


在仓库根目录执行。`<dataset>` 可为 `waterbirds`、`celeba`、`nico` 或 `metashift`。

```bash
# 1. 训练冻结 DINO 的 Slot Attention 表征
python train_stage1.py --config config/<dataset>/stage1.yaml

# 2. 从 Stage I checkpoint 生成带身份校验的四个数组与 complete manifest
python concept_infer.py --config config/<dataset>/concept_infer.yaml --ckpt /path/to/stage1.pth

# 3. 训练 CAFIL 分类器
python train_stage2.py --config config/<dataset>/stage2.yaml

# 4. 评估选出的 Stage II checkpoint
python inference.py --config config/<dataset>/stage2.yaml
```

设置 `STAGE1_CHECKPOINT` 后，也可运行 `scripts/run_pipeline.sh <dataset>`：

```bash
export STAGE1_CHECKPOINT="$CAFIL_OUTPUT_ROOT/cafil_best/waterbirds/stage1/best_cafil_stage1.pth"
bash scripts/run_pipeline.sh waterbirds
```

脚本会先训练 Stage I；若复用已有 Stage I checkpoint，请使用上方的四步手动命令。

### 选模协议

- `eval.primary_metric: wga`：验证时读取 group/context，仅以 worst-group accuracy 选模。
- `eval.primary_metric: mean` 且 `eval.annotation_free: true`：纯 annotation-free 模式；验证和推理不读取 group metadata，仅以 mean accuracy 选模。NICO Stage II YAML 是该模式的正式示例。

### 验证

```bash
pytest -q
CAFIL_FORCE_CPU=1 python train_stage2.py --config config/waterbirds/stage2.yaml --device cpu --dry-run
```

## English

CAFIL is a four-stage image-classification pipeline. It trains frozen-DINO Slot Attention representations, derives concept assignments and consensus scores from a Stage I checkpoint, and then trains and evaluates the CAFIL classifier. The training path does not consume environment labels.

### Installation

Use the [PyTorch selector](https://pytorch.org/get-started/locally/) to install a CPU/CUDA-compatible `torch` and `torchvision` build, then install CAFIL's platform-independent dependencies:

```bash
pip install -r requirements.txt
```

The formal CelebA concept-inference configuration uses GPU FAISS. Install it only when running that configuration:

```bash
pip install -r requirements-faiss-gpu.txt
```

### Data and Outputs

The repository does not bundle datasets, checkpoints, concept artifacts, or DINOv2 weights. Set these absolute paths before running:

```bash
export CAFIL_DATA_ROOT=/absolute/path/to/datasets
export CAFIL_OUTPUT_ROOT=/absolute/path/to/cafil_outputs
export CAFIL_DINO_HOME=/absolute/path/to/torch_cache
```

`CAFIL_DINO_HOME` must contain a local DINOv2 Torch Hub checkout and its checkpoint; CAFIL never downloads model weights. Every formal YAML resolves paths through these variables and fails immediately if one is unset.

### Quick Start

```bash
git clone https://github.com/Yudhpt/CAFIL.git
cd CAFIL
# Create and activate a Python environment, then install PyTorch for your platform.
pip install -r requirements.txt
cp .env.example .env.local
# Edit .env.local, then load it into the current shell.
set -a; source .env.local; set +a
```

Set the four variables in your shell. `.env.local` is a private template only; CAFIL does not load it automatically:

```bash
export CAFIL_DATA_ROOT=/absolute/path/to/datasets
export CAFIL_OUTPUT_ROOT=/absolute/path/to/cafil_outputs
export CAFIL_DINO_HOME=/absolute/path/to/torch_cache
export CAFIL_FORCE_CPU=0
```

DINOv2 must already use this local Torch Hub layout. The repository deliberately never downloads code or weights at runtime:

```text
<CAFIL_DINO_HOME>/
  hub/facebookresearch_dinov2_main/
  hub/checkpoints/dinov2_vitb14_pretrain.pth
```

### Data Layout

```text
<CAFIL_DATA_ROOT>/
  waterbirds/data/{train-00000-of-00001,validation-00000-of-00001,test-00000-of-00001}.parquet
  CelebA/{Img/img_align_celeba,Anno/list_attr_celeba.csv,Eval/list_eval_partition.txt}
  nico/NICO/multi_classification/{train,val,test}/
  metashift/{MetaShift-Cat-Dog-indoor-outdoor,metadata_metashift.csv}
```

After downloading the four MetaShift source folders, generate its CSV with:

```bash
python scripts/metashift/prepare.py --data-root "$CAFIL_DATA_ROOT"
```

The Stage I to Stage II contract is:

```text
P.npy          # [N, K] concept assignment distribution in training-sample order
consscore.npy  # [N] consensus score for the same samples
sample_ids.npy # [N] stable SHA-256 identities from dataset-relative paths
labels.npy     # [N] target labels for row-by-row validation
stage1_artifacts_manifest.json  # complete marker with shape, dtype, size, and SHA-256
```

### Workflow

Each dataset writes to `$CAFIL_OUTPUT_ROOT/cafil_best/<dataset>/`: the Stage I checkpoint is `stage1/best_cafil_stage1.pth`, concept artifacts are in `concept/`, and Stage II checkpoints are in `stage2/`.

Run from the repository root. Replace `<dataset>` with `waterbirds`, `celeba`, `nico`, or `metashift`.

```bash
python train_stage1.py --config config/<dataset>/stage1.yaml
python concept_infer.py --config config/<dataset>/concept_infer.yaml --ckpt /path/to/stage1.pth
python train_stage2.py --config config/<dataset>/stage2.yaml
python inference.py --config config/<dataset>/stage2.yaml
```

`scripts/run_pipeline.sh <dataset>` runs the same sequence after `STAGE1_CHECKPOINT` is set. For example:

```bash
export STAGE1_CHECKPOINT="$CAFIL_OUTPUT_ROOT/cafil_best/waterbirds/stage1/best_cafil_stage1.pth"
bash scripts/run_pipeline.sh waterbirds
```

The script intentionally trains Stage I first; use the manual commands when reusing an existing Stage I checkpoint.

### Selection Protocols

- `eval.primary_metric: wga` loads group/context metadata only for evaluation and selects by worst-group accuracy.
- `eval.primary_metric: mean` together with `eval.annotation_free: true` is the pure annotation-free protocol: validation and inference do not load group metadata and selection uses mean accuracy. The NICO Stage II YAML is the formal example.

### Layout

```text
config/     Dataset-specific Stage I, concept-inference, and Stage II YAML
data/       Dataset adapters, staged DataLoader construction, and Stage I artifact validation
models/     Stage II visual classifier
modules/    Frozen DINO adapter and Slot Attention
train/      Four pipeline implementations, objectives, evaluation, and checkpoint resolution
utils/      Configuration, runtime, Stage I, diagnostics, and transform helpers
scripts/    Portable workflow script and MetaShift metadata preparation
test/       Unit and contract tests
```

### Verification

```bash
pytest -q
CAFIL_FORCE_CPU=1 python train_stage2.py --config config/waterbirds/stage2.yaml --device cpu --dry-run
```
