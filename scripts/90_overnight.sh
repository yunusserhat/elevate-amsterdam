#!/bin/bash
# Overnight chain (2026-10-10 -> 11), runs unattended on worf. Each step logs to logs/;
# a step that fails is recorded in logs/90_overnight.log and the chain moves on where it can.
source /scratch/ybicakci/projects/elevate_amsterdam/env.sh
LOG=logs/90_overnight.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }

say "START overnight chain"
# 1. wait for view extraction and the 27B reference runs
until grep -q "status\|ok" logs/15_extract_views.log 2>/dev/null && ! pgrep -f "[1]5_extract_views" > /dev/null; do sleep 60; done
say "view extraction finished: $(tail -1 logs/15_extract_views.log)"
until grep -qE "REF_DONE|Traceback" logs/23_reference_vllm.log 2>/dev/null; do sleep 60; done
say "reference runs finished: $(grep -c REF_DONE logs/23_reference_vllm.log) REF_DONE marker(s)"

# 2. pilot comparison (4B v1 / 4B v2 / 27B v1 against 27B v2)
$PY_VLM scripts/24_compare_pilot.py > logs/24_compare_pilot.log 2>&1 && say "pilot comparison written" || say "pilot comparison FAILED"

# 3. full view list (views whose four files were written)
$PY_VLM - <<'EOF' >> $LOG 2>&1
import pandas as pd
from pathlib import Path
proj = Path("/scratch/ybicakci/projects/elevate_amsterdam")
m = pd.read_parquet(proj / "data/derived/views_manifest.parquet")
ok = m[m.status.isin(["ok", "exists"])]
full = ok.groupby("pano_id").filter(lambda g: len(g) == 4)
lines = [str(Path(p).relative_to(proj)) for p in full.sort_values(["pano_id", "view"]).path]
(proj / "data/derived/views_all.txt").write_text("\n".join(lines) + "\n")
print(f"views_all.txt: {len(lines)} views of {full.pano_id.nunique()} panoramas; failed: {int((m.status == 'download_failed').sum())}")
EOF

# 4. full audit, template v2, one shard per GPU (about 3.5 h)
say "full audit START"
(CUDA_VISIBLE_DEVICES=0 $PY_VLM scripts/21_audit.py data/derived/views_all.txt outputs/wp3/audit_full_qwen35_4b_v2_shard0.parquet --shard 0/2 --template active_mobility_audit_v2 > logs/21_full_shard0.log 2>&1) &
(CUDA_VISIBLE_DEVICES=1 $PY_VLM scripts/21_audit.py data/derived/views_all.txt outputs/wp3/audit_full_qwen35_4b_v2_shard1.parquet --shard 1/2 --template active_mobility_audit_v2 > logs/21_full_shard1.log 2>&1) &
wait
say "full audit END: $(ls outputs/wp3/audit_full_qwen35_4b_v2_shard*.summary.json 2>/dev/null | wc -l) shard summaries"

# 5. segmentation, SegFormer-B5 Cityscapes, one shard per GPU
say "segmentation START"
(CUDA_VISIBLE_DEVICES=0 $PY_VLM scripts/25_segment.py data/derived/views_all.txt outputs/wp3/segmentation/segformer_b5 --shard 0/2 > logs/25_seg_shard0.log 2>&1) &
(CUDA_VISIBLE_DEVICES=1 $PY_VLM scripts/25_segment.py data/derived/views_all.txt outputs/wp3/segmentation/segformer_b5 --shard 1/2 > logs/25_seg_shard1.log 2>&1) &
wait
say "segmentation END: $(grep -h SEG_SHARD logs/25_seg_shard*.log | tr '\n' ' ')"
say "OVERNIGHT_DONE"
