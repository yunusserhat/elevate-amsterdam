#!/bin/bash
# Step 7.3: after 73_wp6_chain.sh, audit the edited views with 4B and 27B and measure changes.
source /scratch/ybicakci/projects/elevate_amsterdam/env.sh
LOG=logs/75_wp6_verify.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -q CHAIN_DONE logs/73_wp6_chain.log 2>/dev/null; do grep -q FAILED logs/73_wp6_chain.log && { say "chain failed"; exit 1; }; sleep 30; done
say "START"
$PY_VLM scripts/74_wp6_verify.py prepare >> $LOG 2>&1
$PY_VLM scripts/21_audit.py data/derived/wp6_edits_768.txt outputs/wp6/audit_edits_qwen35_4b_v2.parquet \
    --template active_mobility_audit_v2 > logs/75_audit_edits_4b.log 2>&1 && say "4B audit done"
OUT=outputs/wp6/reference_edits_qwen38_27b_v2
$PY_VLM scripts/21_audit.py data/derived/wp6_edits_768.txt $OUT.parquet \
    --model Qwen/Qwen3.8-27B-FP8 --template active_mobility_audit_v2 --tp 2 --util 0.85 --max-num-seqs 64 > logs/75_ref_edits.log 2>&1 && say "27B audit done"
$PY_VLM - <<'PY' >> $LOG 2>&1
import pandas as pd
r = pd.read_parquet("outputs/wp6/reference_edits_qwen38_27b_v2.parquet")
bad = r[r.parse_error.astype(bool)].image_path
open("data/derived/wp6_edits_failed.txt", "w").write("".join(p.split("elevate_amsterdam/")[1] + "\n" for p in bad))
print(f"27B on edits: {len(r)} records, {len(bad)} parse failures")
PY
if [ -s data/derived/wp6_edits_failed.txt ]; then
  $PY_VLM scripts/21_audit.py data/derived/wp6_edits_failed.txt ${OUT}_rp105.parquet \
      --model Qwen/Qwen3.8-27B-FP8 --template active_mobility_audit_v2 --tp 2 --util 0.85 --max-num-seqs 64 \
      --repetition-penalty 1.05 > logs/75_ref_edits_rp105.log 2>&1
fi
$PY_VLM scripts/74_wp6_verify.py measure >> $LOG 2>&1 && say "VERIFY_DONE" || say "verify FAILED"
