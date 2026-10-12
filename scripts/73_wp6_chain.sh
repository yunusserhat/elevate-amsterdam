#!/bin/bash
# Steps 7.2: confirm the WP6 candidate pool with the 27B reference, then make the edits.
source /scratch/ybicakci/projects/elevate_amsterdam/env.sh
LOG=logs/73_wp6_chain.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
say "START"
OUT=outputs/wp6/reference_pool_v2
$PY_VLM scripts/21_audit.py data/derived/wp6_pool_views.txt $OUT.parquet \
    --model Qwen/Qwen3.8-27B-FP8 --template active_mobility_audit_v2 --tp 2 --util 0.85 --max-num-seqs 64 > logs/73_ref_pool.log 2>&1
$PY_VLM - <<'PY' >> $LOG 2>&1
import pandas as pd
r = pd.read_parquet("outputs/wp6/reference_pool_v2.parquet")
bad = r[r.parse_error.astype(bool)].image_path
open("data/derived/wp6_pool_failed.txt", "w").write("".join(p.split("elevate_amsterdam/")[1] + "\n" for p in bad))
print(f"reference pool: {len(r)} records, {len(bad)} parse failures")
PY
if [ -s data/derived/wp6_pool_failed.txt ]; then
  $PY_VLM scripts/21_audit.py data/derived/wp6_pool_failed.txt ${OUT}_rp105.parquet \
      --model Qwen/Qwen3.8-27B-FP8 --template active_mobility_audit_v2 --tp 2 --util 0.85 --max-num-seqs 64 \
      --repetition-penalty 1.05 > logs/73_ref_pool_rp105.log 2>&1
fi
$PY_VLM scripts/72_wp6_confirm.py >> $LOG 2>&1 && say "confirm done" || { say "confirm FAILED"; exit 1; }
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY_IMAGE scripts/71_wp6_edit.py --width 1024 --height 768 > logs/71_wp6_edit.log 2>&1
say "edits: $(grep -c . outputs/wp6/edits_log.csv) log lines; $(tail -1 logs/71_wp6_edit.log)"
say "CHAIN_DONE"
