#!/bin/bash
# Pod-side runner for the Political-Compass follow-ups, one model at a time.
#   bash scripts/pct/run_pct.sh <model_id> <tag> <group>
# group A: stated-opinion logit readout + the 62 PCT propositions through the conjoint
# group B: group A plus the AGI/implant/paraphrase/office variants (scripts/pct/run_variants.sh)
# Prints PCT_DONE when the stated and PCT runs both finished (variants are best-effort).
set -u
MODEL=$1; TAG=$2; GROUP=${3:-A}
STEPS=${STEPS:-stated,pct,decode}   # comma-separated subset of stated,pct,decode
PCT_DRAWS=${PCT_DRAWS:-8}           # name draws per (background, ordering) for the PCT conjoint
cd /workspace/election
PY=${PY:-/workspace/venv/bin/python}
export HF_TOKEN=$(cat /root/.hf_token) HUGGING_FACE_HUB_TOKEN=$(cat /root/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/root/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p experiments/pct logs
ok=1

if [[ ",$STEPS," == *",stated,"* ]]; then
echo "=== stated $TAG $(date -u +%T)"
$PY -u scripts/pct/stated_logit.py --model "$MODEL" --out experiments/pct/stated_${TAG}.jsonl --pct data/pct_issues.json > logs/stated_${TAG}.log 2>&1
grep -q SWEEP_DONE logs/stated_${TAG}.log || { echo "stated FAILED"; tail -5 logs/stated_${TAG}.log; ok=0; }
fi

if [[ ",$STEPS," == *",pct,"* ]]; then
echo "=== pct-conjoint $TAG $(date -u +%T)"
$PY -u scripts/logit_sweep.py --model "$MODEL" --issues data/pct_issues.json --out experiments/pct/pct_${TAG}.jsonl \
    --sweep_draws $PCT_DRAWS --legacy_draws 0 --backgrounds 8 > logs/pct_${TAG}.log 2>&1
grep -q SWEEP_DONE logs/pct_${TAG}.log || { echo "pct-conjoint FAILED"; tail -5 logs/pct_${TAG}.log; ok=0; }
fi

if [[ ",$STEPS," == *",decode,"* ]]; then
echo "=== decode-check $TAG $(date -u +%T)"
$PY -u scripts/pct/decode_check.py --model "$MODEL" --out experiments/pct/decode_${TAG}.jsonl > logs/decode_${TAG}.log 2>&1
grep -q SWEEP_DONE logs/decode_${TAG}.log || { echo "decode-check FAILED"; tail -5 logs/decode_${TAG}.log; ok=0; }
fi

if [ "$GROUP" = "B" ]; then
  echo "=== variants $TAG $(date -u +%T)"
  MODEL="$MODEL" TAG="$TAG" PY="$PY" bash scripts/pct/run_variants.sh all > logs/variants_${TAG}.log 2>&1 || { echo "variants FAILED (best-effort)"; tail -5 logs/variants_${TAG}.log; }
fi

rm -rf /root/.cache/huggingface/hub /root/.cache/huggingface/xet
[ "$ok" = 1 ] && echo PCT_DONE || echo PCT_PARTIAL
