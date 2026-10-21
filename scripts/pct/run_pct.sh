#!/bin/bash
# Pod-side runner for the Political-Compass follow-ups, one model at a time.
#   bash scripts/pct/run_pct.sh <model_id> <tag> <group>
# group A: stated-opinion logit readout + the 62 PCT propositions through the conjoint
# group B: group A plus the AGI/implant/paraphrase/office variants (scripts/pct/run_variants.sh)
# Prints PCT_DONE when the stated and PCT runs both finished (variants are best-effort).
set -u
MODEL=$1; TAG=$2; GROUP=${3:-A}
STEPS=${STEPS:-stated,pct,decode}   # comma-separated subset of stated,pct,decode,sdecode,reword
PCT_DRAWS=${PCT_DRAWS:-8}           # name draws per (background, ordering) for the PCT conjoint
DECODE_DRAWS=${DECODE_DRAWS:-4}     # name draws for the decode-vs-logit check (512 trials at 4)
for s in ${STEPS//,/ }; do case $s in stated|pct|decode|sdecode|reword|issues) ;; *) echo "unknown step '$s' in STEPS=$STEPS"; exit 1;; esac; done
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

if [[ ",$STEPS," == *",reword,"* ]]; then
echo "=== reword $TAG $(date -u +%T)"
$PY -u scripts/logit_sweep.py --model "$MODEL" --issues data/reword_issues.json --out experiments/pct/reword_${TAG}.jsonl \
    --flips ${REWORD_FLIPS} --sweep_draws ${REWORD_DRAWS:-6} --legacy_draws 0 --backgrounds 8 > logs/reword_${TAG}.log 2>&1
grep -q SWEEP_DONE logs/reword_${TAG}.log || { echo "reword FAILED"; tail -5 logs/reword_${TAG}.log; ok=0; }
fi

if [[ ",$STEPS," == *",issues,"* ]]; then   # generic: run ISSUES_FILE (flips ISSUES_FLIPS) -> experiments/pct/${ISSUES_PREFIX}_<tag>.jsonl
echo "=== issues($ISSUES_PREFIX) $TAG $(date -u +%T)"
$PY -u scripts/logit_sweep.py --model "$MODEL" --issues ${ISSUES_FILE} --out experiments/pct/${ISSUES_PREFIX}_${TAG}.jsonl \
    --flips ${ISSUES_FLIPS:-all} --sweep_draws ${ISSUES_DRAWS:-6} --legacy_draws 0 --backgrounds 8 > logs/${ISSUES_PREFIX}_${TAG}.log 2>&1
grep -q SWEEP_DONE logs/${ISSUES_PREFIX}_${TAG}.log || { echo "issues($ISSUES_PREFIX) FAILED"; tail -5 logs/${ISSUES_PREFIX}_${TAG}.log; ok=0; }
fi

if [[ ",$STEPS," == *",sdecode,"* ]]; then
echo "=== stated-decode $TAG $(date -u +%T)"
$PY -u scripts/pct/stated_decode.py --model "$MODEL" --out experiments/pct/sdecode_${TAG}.jsonl > logs/sdecode_${TAG}.log 2>&1
grep -q SWEEP_DONE logs/sdecode_${TAG}.log || { echo "stated-decode FAILED"; tail -5 logs/sdecode_${TAG}.log; ok=0; }
fi

if [[ ",$STEPS," == *",decode,"* ]]; then
echo "=== decode-check $TAG $(date -u +%T)"
$PY -u scripts/pct/decode_check.py --model "$MODEL" --out experiments/pct/decode_${TAG}.jsonl --draws ${DECODE_DRAWS:-4} > logs/decode_${TAG}.log 2>&1
grep -q SWEEP_DONE logs/decode_${TAG}.log || { echo "decode-check FAILED"; tail -5 logs/decode_${TAG}.log; ok=0; }
fi

if [ "$GROUP" = "B" ]; then
  echo "=== variants $TAG $(date -u +%T)"
  MODEL="$MODEL" TAG="$TAG" PY="$PY" bash scripts/pct/run_variants.sh all > logs/variants_${TAG}.log 2>&1 || { echo "variants FAILED (best-effort)"; tail -5 logs/variants_${TAG}.log; }
fi

rm -rf /root/.cache/huggingface/hub /root/.cache/huggingface/xet
[ "$ok" = 1 ] && echo PCT_DONE || echo PCT_PARTIAL
