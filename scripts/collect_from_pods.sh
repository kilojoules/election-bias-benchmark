#!/bin/bash
# Pull rubric-v2 grades from every pod, merge, analyze.
S=${POD_ENDPOINTS_DIR:?set to a dir containing EP*.txt files with "host port" per pod}
SSHO="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
cd /Users/julianquick/election
: > experiments/grades.jsonl
for P in "EP:grades.jsonl" "EP2:grades_1b.jsonl" "EP_12b:grades.jsonl" "EP_4b:grades.jsonl" "EP_1b:grades.jsonl"; do
  F=${P%%:*}; G=${P##*:}; read HOST PORT < $S/$F.txt
  ssh $SSHO -p $PORT root@$HOST "grep '\"rubric_version\": 2' /workspace/election/experiments/$G 2>/dev/null | gzip -c | base64" > $S/pull_$F.b64 2>/dev/null
  python3 -c "import base64,gzip,sys;d=open('$S/pull_$F.b64','rb').read();sys.stdout.buffer.write(gzip.decompress(base64.b64decode(d)) if d.strip() else b'')" >> experiments/grades.jsonl
done
python3 - <<'PY'
import json,collections
seen=set(); out=[]
for l in open('experiments/grades.jsonl'):
    g=json.loads(l); k=(g["model"],g["job_key"],g["call"])
    if k in seen: continue
    seen.add(k); out.append(l)
open('experiments/grades.jsonl','w').writelines(out)
c=collections.Counter()
for l in out:
    g=json.loads(l)
    if g["judge"].startswith("prefilter"): continue
    c[(g["model"].split("/")[-1],g["prompt_id"])]+=1
print("judged calls per model/prompt:")
for k in sorted(c): print(f"  {k[0]:24s} {k[1]:22s} {c[k]}")
PY
python3 scripts/analyze.py --responses experiments/responses.jsonl --grades experiments/grades.jsonl --out experiments/results.json > experiments/results.txt 2>&1
echo "wrote experiments/results.txt ($(wc -l < experiments/results.txt) lines)"
