#!/usr/bin/env bash
# run_kaldi_to_msswift.sh
#
# Convert Aishell train / dev / test Kaldi data dirs to ms-swift JSONL.

set -euo pipefail

# ─── edit these if needed ────────────────────────────────────────────
PY=local/prepare_aishell.py            # path to the Python converter
SYSTEM_MSG="You are a helpful ASR system."
QUERY_PREFIX="请听下面的语音并给出文字："
QUERY_SUFFIX=""
OUT_DIR="data"
# ─────────────────────────────────────────────────────────────────────

declare -A DATA_DIRS=(
  [train]="/home/jinzr/nfs/projects/icefall/egs/aishell/ASR/data/manifests/aishell_train"
  [dev]="/home/jinzr/nfs/projects/icefall/egs/aishell/ASR/data/manifests/aishell_dev"
  [test]="/home/jinzr/nfs/projects/icefall/egs/aishell/ASR/data/manifests/aishell_test"
)

mkdir -p "$OUT_DIR"

for split in train dev test; do
  data_dir="${DATA_DIRS[$split]}"
  out_file="${OUT_DIR}/aishell_${split}.jsonl"

  echo "⇒ Converting ${split} split"
  python "$PY" \
      --data-dir "$data_dir" \
      --output   "$out_file" \
      --system   "$SYSTEM_MSG" \
      --query-prefix "$QUERY_PREFIX" \
      --query-suffix "$QUERY_SUFFIX"
done

echo "✓ JSONL files are in $OUT_DIR"