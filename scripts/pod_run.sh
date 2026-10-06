#!/usr/bin/env bash
# Run an experiment queue on a RunPod GPU pod (CUDA 13 host). Environment:
#   EXPERIMENTS  queue file, e.g. experiments/m8_lr_schedule.txt   PARALLEL  concurrent jobs
# Results are printed as RESULT lines; the pod sleeps afterwards so the logs stay readable.
set -ex
cd "$(dirname "$0")/.."
DRV=$(nvidia-smi | grep -o 'CUDA Version: [0-9]*' | grep -o '[0-9]*$')
if [ "$DRV" -lt 13 ]; then echo "HOST_CUDA_TOO_OLD $DRV"; exit 1; fi
pip install -q uv
mkdir -p data/Criteo_x4
( cd data && curl -sL -o z.zip 'https://huggingface.co/datasets/reczoo/Criteo_x4/resolve/main/Criteo_x4.zip?download=true' \
  && unzip -q z.zip -d Criteo_x4 && rm z.zip && echo DATA_READY ) &
uv sync --frozen
wait
nproc; free -g | head -2
uv run python -m bars_dcn.bench.queue "${EXPERIMENTS}" --parallel "${PARALLEL:-3}" --device auto
echo ALL_DONE
