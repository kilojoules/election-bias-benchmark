# Runbook: reproducing the run on a single 80GB GPU

Everything below was run on one RunPod A100 80GB (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`,
200GB container disk). Any 80GB card works. Gated models (Gemma 3, Llama 3.1) need a Hugging Face token
with the licenses accepted.

```bash
# 0. on the pod
pip install -r requirements.txt
export HF_TOKEN=...   # also HUGGING_FACE_HUB_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=0

# 1. regenerate the factorial (only needed if you edit data/stances.json)
python3 scripts/generate_candidates.py

# 2. generate target responses (resumable; --prompts limits to a prompt subset; --free_cache deletes each model's weights after use)
python3 -u scripts/run_targets.py \
  --models meta-llama/Llama-3.1-8B-Instruct,google/gemma-3-1b-it,google/gemma-3-4b-it,google/gemma-3-12b-it,google/gemma-3-27b-it \
  --n 10 --out experiments/responses.jsonl --batch 16 --free_cache

# 3. grade with the blind judge (Llama 3.1 70B, pre-quantized nf4; resumable per call; --models sets priority order)
python3 -u scripts/judge.py --responses experiments/responses.jsonl --out experiments/grades.jsonl --batch 64

# 4. analyze locally
python3 scripts/analyze.py --responses experiments/responses.jsonl --grades experiments/grades.jsonl
```

Notes
- Gemma 3 must be loaded with eager attention; SDPA produces NaNs under left-padded sampling. The runner does this.
- Gemma 3's chat template accepts a system role. The runner detects templates that do not and folds the system text into the user turn.
- The judge is the bottleneck: the bitsandbytes 4-bit path runs at roughly 2.3 s per call at batch 64. The judge skips the
  AI-axis call when a response contains no AI-related token (it can only return not_mentioned) and records that as
  `judge: prefilter:no_ai_keywords`.
- `data/rubric.json` has a `version`. Grade lines record it; the analysis ignores lines from other versions, so a rubric
  edit followed by a re-run regrades only what changed.
- Grades are one line per (model, job_key, call) with call in {property_tax, agi_limitations, response}.
