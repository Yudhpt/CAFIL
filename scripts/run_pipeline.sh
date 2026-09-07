#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?usage: scripts/run_pipeline.sh <waterbirds|celeba|nico|metashift>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python}"
stage1_config="config/${dataset}/stage1.yaml"
concept_config="config/${dataset}/concept_infer.yaml"
stage2_config="config/${dataset}/stage2.yaml"

cd "$root"
for config in "$stage1_config" "$concept_config" "$stage2_config"; do
  [[ -f "$config" ]] || { echo "Unsupported dataset: $dataset" >&2; exit 2; }
done

"$python_bin" train_stage1.py --config "$stage1_config"
checkpoint="${STAGE1_CHECKPOINT:-}"
if [[ -z "$checkpoint" ]]; then
  echo "Set STAGE1_CHECKPOINT to the Stage-I checkpoint before concept inference." >&2
  exit 2
fi
"$python_bin" concept_infer.py --config "$concept_config" --ckpt "$checkpoint"
"$python_bin" train_stage2.py --config "$stage2_config"
"$python_bin" inference.py --config "$stage2_config"
