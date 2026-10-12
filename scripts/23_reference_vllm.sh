#!/bin/bash
source /scratch/ybicakci/projects/elevate_amsterdam/env.sh
for t in active_mobility_audit_v2 active_mobility_audit_v1; do
  $PY_VLM scripts/21_audit.py data/derived/pilot_front200.txt outputs/wp3/reference_pilot_front200_qwen38_27b_fp8_${t##*_}.parquet \
      --model Qwen/Qwen3.8-27B-FP8 --template $t --tp 2 --util 0.85 --max-num-seqs 64
done
echo REF_DONE
