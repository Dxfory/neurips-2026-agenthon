#!/usr/bin/env bash
# Rebuild the candidate code tree from a pinned team-repo commit using an explicit allowlist.
# Local only. Never pushes. Usage: scripts/export_candidate.sh [SRC_REPO] [COMMIT]
set -euo pipefail
SRC=${1:-/workspace/Agenthon-2026-yiping}
COMMIT=${2:-65a7ad33}
OUT=$(cd "$(dirname "$0")/.." && pwd)/tracks
rm -rf "$OUT/t1-coding/src" "$OUT/t2-forecasting/src" "$OUT/t4-explainability/src"
mkdir -p "$OUT/t1-coding/src" "$OUT/t2-forecasting/src" "$OUT/t4-explainability/src" "$OUT/t3-simulation"
x() { dest=$1; shift; git -C "$SRC" archive --format=tar "$COMMIT" -- "$@" | tar -x -C "$dest"; }
# T1: Final-line Python solver agent + runtime + build, no official task copies/evidence
x "$OUT/t1-coding/src" T1/README.md T1/PIPELINE.md T1/TOOL_USE_CONTRACT.md T1/CODE_OUTPUT_CONTRACT.md \
  T1/t1_phase1 T1/qfruntime T1/qfkit T1/scripts T1/build T1/Dockerfile T1/Dockerfile.amd64 T1/requirements-container.txt
# T2: final v2.7.7 numeric forecaster only (no contracts/research/backtest data)
x "$OUT/t2-forecasting/src" T2/README.md T2/FORECAST_PIPELINE.md T2/TASK_CONTRACT.md T2/forecast-v2_7_7
# T4: current v21.14 source tree (agent_v21_11) + README; no calib_data/error_book/sub
x "$OUT/t4-explainability/src" T4/README.md T4/agent_v21_11
echo "exported from $SRC@$COMMIT"; du -sh "$OUT"/*
# scrub links to the private team repo
rg -l 'github.com/Yiping-Yin/Agenthon-2026' "$OUT" 2>/dev/null | xargs -r sed -i 's#https://github.com/Yiping-Yin/Agenthon-2026#<team-private-repo>#g'
