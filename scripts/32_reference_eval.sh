#!/bin/bash
# Step 2.7 (a): reference audit (Qwen3.8-27B-FP8, both GPUs) on the 500 evaluation views, then one
# re-run with repetition_penalty 1.05 for records whose JSON-schema output looped on whitespace.
source /scratch/ybicakci/projects/elevate_amsterdam/env.sh
OUT=outputs/wp3/reference_eval500_qwen38_27b_fp8_v2
$PY_VLM scripts/21_audit.py data/derived/eval_500.txt $OUT.parquet \
    --model Qwen/Qwen3.8-27B-FP8 --template active_mobility_audit_v2 --tp 2 --util 0.85 --max-num-seqs 64
$PY_VLM - <<'EOF'
import pandas as pd
r = pd.read_parquet("outputs/wp3/reference_eval500_qwen38_27b_fp8_v2.parquet")
bad = r[r.parse_error.astype(bool)].image_path
open("data/derived/eval_500_ref_failed.txt", "w").write("".join(p.split("elevate_amsterdam/")[1] + "\n" for p in bad))
print(f"first pass: {len(r)} records, {len(bad)} parse failures")
EOF
if [ -s data/derived/eval_500_ref_failed.txt ]; then
  $PY_VLM scripts/21_audit.py data/derived/eval_500_ref_failed.txt ${OUT}_rp105.parquet \
      --model Qwen/Qwen3.8-27B-FP8 --template active_mobility_audit_v2 --tp 2 --util 0.85 --max-num-seqs 64 \
      --repetition-penalty 1.05
fi
echo EVALREF_DONE
