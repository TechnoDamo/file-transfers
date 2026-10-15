#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: GPU_ID=<free-index> OLMO_DATA_ROOT=<data-root> $0 <baseline|hybrid> <run-dir> [extra OLMo overrides...]" >&2
  exit 2
fi

arm=$1
run_dir=$2
shift 2
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "$script_dir/.." && pwd)"
case "$arm" in
  baseline) config="$experiment_dir/configs/baseline-oracle-160m.yaml" ;;
  hybrid) config="$experiment_dir/configs/hybrid-fo-head-embed-oracle-body-160m.yaml" ;;
  *) echo "Unknown arm '$arm'; choose baseline or hybrid" >&2; exit 2 ;;
esac

: "${GPU_ID:?Set GPU_ID to an available GPU index after checking the fleet}"
: "${OLMO_DATA_ROOT:?Set OLMO_DATA_ROOT to the folder containing dcml/ and wikitext/}"
: "${WANDB_BASE_URL:=http://127.0.0.1:28080}"
export WANDB_BASE_URL
export WANDB_MODE=online
if [[ "$run_dir" != /* ]]; then
  echo "Use an absolute run directory so outputs stay outside the source tree." >&2
  exit 2
fi
mkdir -p "$run_dir"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OLMO_DATA_ROOT
export OLMO_NO_DISTRIBUTED=1
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT

for required_path in "$OLMO_DATA_ROOT/dcml/subset-6B" "$OLMO_DATA_ROOT/wikitext/validation"; do
  if [[ ! -d "$required_path" ]]; then
    echo "Required dataset directory is missing: $required_path" >&2
    exit 2
  fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the single-GPU launcher." >&2
  exit 2
fi
nvidia-smi -i "$GPU_ID" --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv \
  | tee "$run_dir/preflight.log"
python -c 'import torch, wandb; print("PyTorch", torch.__version__, "W&B", wandb.__version__, "CUDA available:", torch.cuda.is_available(), "visible devices:", torch.cuda.device_count()); assert torch.cuda.is_available() and torch.cuda.device_count() == 1' \
  2>&1 | tee -a "$run_dir/preflight.log"
python - "$WANDB_BASE_URL" <<'PY' 2>&1 | tee -a "$run_dir/preflight.log"
import sys
import urllib.request

url = sys.argv[1].rstrip("/") + "/"
with urllib.request.urlopen(url, timeout=5) as response:
    print("W&B endpoint:", url, "HTTP", response.status)
    if response.status >= 400:
        raise SystemExit("W&B endpoint is not healthy")
PY

cd "$experiment_dir/source/code/OLMo"
python scripts/train.py "$config" \
  "--data_root=$OLMO_DATA_ROOT" \
  "--save_folder=$run_dir" \
  "--distributed_strategy=single" \
  --run_name="$arm-160m" \
  "$@" 2>&1 | tee "$run_dir/train.log"
