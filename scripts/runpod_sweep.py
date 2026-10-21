"""Drive one RunPod pod through a list of models for the logit sweep, then destroy it.

Runs locally (stdlib only). Creates the pod with runpodctl (super_lab key from ~/.super_lab_run.pod),
pushes scripts/ and data/, installs deps, launches a setsid driver that runs scripts/logit_sweep.py
per model, polls, pulls each finished model's results into experiments/sweep/, and terminates the
pod, verifying the termination through the GraphQL API. Every pod is named julian-election-sweep-<pod>
with OWNER=julian and an auto-terminate deadline as a backstop.

Usage: caffeinate -i python3 scripts/runpod_sweep.py --pod s1 [--models a,b] [--attach POD_ID [--setup]] [--keep]
"""
import argparse
import base64
import datetime as dt
import gzip
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROSTER = json.loads((ROOT / "scripts" / "sweep_models.json").read_text())
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
SSHO = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30", "-o", "BatchMode=yes"]
POD_PREFIX = "julian-election-sweep-"
PCT_MODE = False   # set from the pod spec ("driver": "pct"): results live under experiments/pct/
PIP = 'pip install -q --timeout 60 --retries 8 "transformers>=5.20" "accelerate>=1.10" "huggingface_hub>=0.30" sentencepiece protobuf tiktoken'
PIP_OPTIONAL = "blobfile kernels mistral-common"


def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def api_key():
    return re.search(r"rpa_[A-Za-z0-9]+", (Path.home() / ".super_lab_run.pod").read_text()).group(0)


def gql(query):
    req = urllib.request.Request(f"https://api.runpod.io/graphql?api_key={api_key()}",
                                 data=json.dumps({"query": query}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def my_pods():
    return gql("query{ myself{ pods{ id name desiredStatus costPerHr } } }")["data"]["myself"]["pods"]


def pod_endpoint(pid):
    d = gql('query{ pod(input:{podId:"%s"}){ desiredStatus runtime{ uptimeInSeconds ports{ ip isIpPublic publicPort type } } } }' % pid)
    pod = d["data"]["pod"]
    if pod is None:
        return "gone", None
    rt = pod.get("runtime") or {}
    ports = [p for p in (rt.get("ports") or []) if p["type"] == "tcp" and p["isIpPublic"]]
    return pod["desiredStatus"], (ports[0]["ip"], ports[0]["publicPort"]) if ports else None, bool(rt.get("ports"))


def run(cmd, timeout=600, check=False, input_=None):
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout, input=input_,
                       **({} if input_ is not None else {"stdin": subprocess.DEVNULL}))
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])}... failed: {p.stderr[-800:]}")
    return p


class Pod:
    def __init__(self, name, host=None, port=None):
        self.name, self.host, self.port = name, host, port

    def ssh(self, script, timeout=600):
        return run(["ssh", *SSHO, "-p", str(self.port), f"root@{self.host}", script], timeout=timeout)

    def scp_to(self, local, remote, timeout=600):
        return run(["scp", *SSHO, "-P", str(self.port), str(local), f"root@{self.host}:{remote}"], timeout=timeout)

    def pull_file(self, remote, local):
        """gzip+base64 over the ssh channel (scp egress can stall on loaded pods)."""
        p = self.ssh(f"[ -f {remote} ] && gzip -c {remote} | base64 -w0", timeout=1800)
        if p.returncode != 0 or not p.stdout.strip():
            return False
        Path(local).write_bytes(gzip.decompress(base64.b64decode(p.stdout.strip())))
        return True


def create_pod_gql(spec, name, hours):
    """GraphQL deploy that insists on a host with a public IP (runpodctl cannot filter for one; some secure-cloud
    hosts map only a proxied HTTP port and are unreachable over ssh)."""
    terminate_at = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    env = ", ".join('{key:"%s", value:"%s"}' % kv for kv in (("OWNER", "julian"), ("PROJECT", "election"), ("JOB", name)))
    q = ('mutation { podFindAndDeployOnDemand(input: { cloudType: SECURE, gpuCount: %d, volumeInGb: 20, containerDiskInGb: %d, '
         'minVcpuCount: 2, minMemoryInGb: 15, gpuTypeId: "%s", name: "%s", imageName: "%s", dockerArgs: "", ports: "22/tcp", '
         'volumeMountPath: "/workspace", env: [%s], supportPublicIp: true, startSsh: true, terminateAfter: "%s" }) { id } }'
         % (spec["gpu_count"], spec["disk_gb"], spec["gpu"], name, IMAGE, env, terminate_at))
    log(f"creating {name} via GraphQL (public IP required): {spec['gpu_count']}x {spec['gpu']}, auto-terminate {terminate_at}")
    d = gql(q)
    if d.get("errors") and "terminateAfter" in json.dumps(d["errors"]):
        log("API rejected terminateAfter; retrying without it (the monitor deadline is the only backstop)")
        d = gql(q.replace(', terminateAfter: "%s"' % terminate_at, ""))
    pod = ((d.get("data") or {}).get("podFindAndDeployOnDemand") or {})
    if not pod.get("id"):
        raise RuntimeError(f"GraphQL pod create failed: {json.dumps(d)[-1200:]}")
    log(f"pod id {pod['id']}")
    return pod["id"]


def create_pod(spec, name, hours):
    if spec.get("public_ip"):
        return create_pod_gql(spec, name, hours)
    env = json.dumps({"OWNER": "julian", "PROJECT": "election", "JOB": name})
    terminate_at = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cmd = ["runpodctl", "pod", "create", "--name", name, "--gpu-id", spec["gpu"], "--gpu-count", str(spec["gpu_count"]),
           "--image", IMAGE, "--container-disk-in-gb", str(spec["disk_gb"]), "--volume-in-gb", "20",
           "--env", env, "--ports", "22/tcp", "--cloud-type", "SECURE", "--terminate-after", terminate_at, "-o", "json"]
    env_ = dict(os.environ, RUNPOD_API_KEY=api_key())
    log(f"creating {name}: {spec['gpu_count']}x {spec['gpu']}, {spec['disk_gb']}GB disk, auto-terminate {terminate_at}")
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env_)
    out = p.stdout + p.stderr
    m = re.search(r'"id"\s*:\s*"([a-z0-9]{10,})"', out) or re.search(r"\b([a-z0-9]{14})\b", out)
    if p.returncode != 0 or not m:
        raise RuntimeError(f"pod create failed (rc={p.returncode}): {out[-1500:]}")
    pid = m.group(1)
    log(f"pod id {pid}")
    return pid


class NoSshPort(RuntimeError):
    """The pod booted on a host that maps no public TCP port, so it can never be reached over ssh."""


def wait_ssh(pid, pod, minutes=20, no_tcp_minutes=5):
    deadline = time.time() + 60 * minutes
    booted_without_tcp = None
    while time.time() < deadline:
        status, ep, has_runtime = pod_endpoint(pid)
        if status == "gone":
            raise RuntimeError("pod vanished while waiting for ssh")
        if ep:
            pod.host, pod.port = ep
            if pod.ssh("echo ok", timeout=40).stdout.strip() == "ok":
                log(f"ssh up at {pod.host}:{pod.port}")
                return
        elif has_runtime:   # the container is up and has mapped ports, none of them public TCP
            booted_without_tcp = booted_without_tcp or time.time()
            if time.time() - booted_without_tcp > 60 * no_tcp_minutes:
                raise NoSshPort(f"pod {pid} has been running {no_tcp_minutes} min with no public TCP port")
        time.sleep(15)
    raise RuntimeError("no ssh endpoint within the wait window")


def pod_name(pid):
    d = gql('query{ pod(input:{podId:"%s"}){ name } }' % pid)
    pod = (d.get("data") or {}).get("pod")
    return pod["name"] if pod else None


def assert_mine(pid):
    """Shared account: refuse to touch any pod that is not one of ours."""
    name = pod_name(pid)
    if name is None:
        return None
    if not name.startswith(POD_PREFIX):
        raise SystemExit(f"refusing to act on pod {pid} named {name!r}: not a {POD_PREFIX}* pod")
    return name


def terminate(pid):
    if assert_mine(pid) is None:
        log(f"pod {pid} already gone")
        return True
    env_ = dict(os.environ, RUNPOD_API_KEY=api_key())
    for attempt in range(6):
        subprocess.run(["runpodctl", "pod", "delete", pid], capture_output=True, text=True, timeout=120, env=env_)
        time.sleep(5)
        ids = [p["id"] for p in my_pods()]
        if pid not in ids:
            log(f"pod {pid} terminated (verified via API); remaining pods: {ids}")
            return True
        log(f"terminate attempt {attempt+1}: pod still listed")
        if attempt >= 2:
            gql('mutation{ podTerminate(input:{podId:"%s"}) }' % pid)
            time.sleep(8)
    log(f"WARNING: could not verify termination of {pid}; check the dashboard")
    return False


def setup(pod, models, spec, pod_name):
    tags = [next((x["tag"] for x in ROSTER["models"] if x["id"] == m), m.split("/")[-1]) for m in models]
    for t in tags:   # stale local markers would make pull_results skip or mis-pull this run's results
        for ext in (".DONE", ".FAIL"):
            (ROOT / "experiments" / "sweep" / f"{t}{ext}").unlink(missing_ok=True)
    # Stop any previous driver and wait until its scoring process has actually released the GPU.
    # ([h]-style patterns so pkill/pgrep never match this ssh command's own shell.)
    pod.ssh("pkill -9 -f 'bash /workspace/run.s[h]'; pkill -9 -f 'run_pc[t].sh'; pkill -9 -f 'run_variant[s].sh'; "
            "P='logit_swee[p].py|stated_logi[t].py|decode_chec[k].py'; pkill -f \"$P\"; "
            "for i in $(seq 1 40); do pgrep -f \"$P\" >/dev/null || break; sleep 2; [ $i -ge 10 ] && pkill -9 -f \"$P\"; done; "
            "for i in $(seq 1 30); do [ \"$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)\" -lt 2000 ] && break; sleep 2; done; "
            "pgrep -f \"$P\" >/dev/null && echo STILL_RUNNING; "
            "rm -rf /root/.cache/huggingface/hub /root/.cache/huggingface/xet; "   # a killed run never reached its own purge
            + "rm -f " + " ".join(f"/workspace/election/experiments/sweep/{t}.DONE /workspace/election/experiments/sweep/{t}.FAIL" for t in tags) + "; "
            "mkdir -p /workspace/election/experiments/sweep /workspace/election/logs; "
            "rm -f /workspace/election/experiments/sweep/ALL_DONE", timeout=180)   # a stale marker would end the new run at its first poll
    tar = subprocess.run(["tar", "czf", "-", "-C", str(ROOT), "scripts", "data", "requirements.txt"], capture_output=True).stdout
    p = subprocess.run(["ssh", *SSHO, "-p", str(pod.port), f"root@{pod.host}", "tar xzf - --no-same-owner --no-same-permissions -C /workspace/election"],
                       input=tar, capture_output=True, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"push failed: {p.stderr[-500:]}")
    pod.scp_to(Path.home() / ".hf_token", "/root/.hf_token")
    r = pod.ssh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h / | tail -1; "
                "for p in python3 /venv/bin/python /workspace/venv/bin/python /opt/conda/bin/python /usr/bin/python3; do "
                "$p -c 'import torch,sys;print(\"TORCH\", sys.executable, torch.__version__)' 2>/dev/null && break; done", timeout=120)
    log("pod: " + " | ".join(l.strip() for l in r.stdout.splitlines()))
    m = re.search(r"TORCH (\S+) (\S+)", r.stdout)
    if not m:
        raise RuntimeError("no python with torch found on the pod")
    sys_py = m.group(1)
    py = "/workspace/venv/bin/python"
    r = pod.ssh(f"export HF_HUB_ENABLE_HF_TRANSFER=0; {sys_py} -m venv --system-site-packages /workspace/venv > /root/pip.log 2>&1 && "
                f"{py} -m {PIP} >> /root/pip.log 2>&1 && "
                f"({py} -m pip install -q {PIP_OPTIONAL} >> /root/pip.log 2>&1 || echo OPTIONAL_PIP_FAILED) && "
                f"{py} -c 'import transformers,torch,accelerate;print(transformers.__version__, torch.__version__, torch.cuda.is_available())' || {{ echo PIP_FAIL; tail -5 /root/pip.log; }}", timeout=1500)
    log("pip: " + r.stdout.strip().replace("\n", " | "))
    if "PIP_FAIL" in r.stdout:
        wheels = os.environ.get("ELECTION_WHEELS")   # offline fallback: a local dir of linux/cp312 wheels for PIP's packages
        if wheels and Path(wheels).is_dir():
            log(f"PyPI unreachable from the pod; pushing wheels from {wheels} and installing offline")
            tar = subprocess.run(["tar", "czf", "-", "-C", wheels, "."], capture_output=True)
            subprocess.run(["ssh", *SSHO, "-p", str(pod.port), f"root@{pod.host}", "mkdir -p /workspace/wheels && tar xzf - -C /workspace/wheels"],
                           input=tar.stdout, capture_output=True, timeout=900)
            pkgs = PIP.split("pip install -q", 1)[1].replace("--timeout 60 --retries 8", "")
            r = pod.ssh(f"{py} -m pip install -q --no-index --find-links /workspace/wheels {pkgs} >> /root/pip.log 2>&1; "
                        f"{py} -c 'import transformers,torch,accelerate;print(transformers.__version__, torch.__version__, torch.cuda.is_available())' || {{ echo PIP_FAIL; tail -5 /root/pip.log; }}", timeout=900)
            log("pip (offline): " + r.stdout.strip().replace("\n", " | "))
        if "PIP_FAIL" in r.stdout:
            raise RuntimeError("pip install failed")
    lines = ["#!/bin/bash", "cd /workspace/election",
             "export HF_TOKEN=$(cat /root/.hf_token) HUGGING_FACE_HUB_TOKEN=$(cat /root/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=0",
             "export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=/root/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", f"PY={py}",
             "echo $$ > /workspace/driver.pid"]
    for m in models:
        tag = next((x["tag"] for x in ROSTER["models"] if x["id"] == m), m.split("/")[-1])
        lines.append(f"echo \"=== START {tag} $(date -u +%H:%M:%S)\" >> logs/driver.log")
        if spec.get("driver") == "pct":
            group = "B" if m in spec.get("variant_models", []) else "A"
            steps = spec.get("model_steps", {}).get(tag, spec.get("steps", "stated,pct,decode"))
            envs = (f"STEPS={steps} PCT_DRAWS={spec.get('pct_draws', 8)} DECODE_DRAWS={spec.get('decode_draws', 4)} "
                    f"REWORD_DRAWS={spec.get('reword_draws', 6)} REWORD_FLIPS={spec.get('reword_flips', 'all')} "
                    f"ISSUES_FILE={spec.get('issues_file', 'data/issues.json')} ISSUES_PREFIX={spec.get('issues_prefix', 'issues')} "
                    f"ISSUES_FLIPS={spec.get('issues_flips', 'all')} ISSUES_DRAWS={spec.get('issues_draws', 6)} ISSUES_CONDITIONS={spec.get('issues_conditions', '')}")
            lines.append(f"{envs} PY=$PY bash scripts/pct/run_pct.sh {m} {tag} {group} > logs/{tag}.log 2>&1")
            marker = "PCT_DONE"   # a PCT_PARTIAL (a failed step) falls through to the FAIL branch and is retried once
        else:
            lines.append(f"$PY -u scripts/logit_sweep.py --model {m} --out experiments/sweep/{tag}.jsonl "
                         f"--sweep_draws {spec['sweep_draws']} --legacy_draws {spec['legacy_draws']} --free_cache > logs/{tag}.log 2>&1")
            marker = "SWEEP_DONE"
        lines.append(f"if grep -q '{marker}' logs/{tag}.log; then touch experiments/sweep/{tag}.DONE; echo \"=== DONE {tag} $(date -u +%H:%M:%S)\" >> logs/driver.log; "
                     f"else touch experiments/sweep/{tag}.FAIL; echo \"=== FAIL {tag} $(date -u +%H:%M:%S)\" >> logs/driver.log; "
                     f"rm -rf /root/.cache/huggingface/hub /root/.cache/huggingface/xet; fi")
    lines.append("touch experiments/sweep/ALL_DONE")
    script = "\n".join(lines) + "\n"
    p = subprocess.run(["ssh", *SSHO, "-p", str(pod.port), f"root@{pod.host}", "cat > /workspace/run.sh && chmod +x /workspace/run.sh"],
                       input=script, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"could not write run.sh: {p.stderr[-300:]}")
    # Close the remote shell's stdio before forking, so nothing keeps the ssh channel open.
    r = pod.ssh("cd /workspace; echo launching; exec >/dev/null 2>&1 </dev/null; setsid nohup bash /workspace/run.sh </dev/null >/workspace/run.out 2>&1 &", timeout=60)
    time.sleep(5)
    r = pod.ssh("pgrep -x -f 'bash /workspace/run.sh' >/dev/null && echo launched || { echo NOT_RUNNING; cat /workspace/run.out; }", timeout=60)
    if "launched" not in r.stdout:
        raise RuntimeError(f"driver did not start: {r.stdout} {r.stderr[-300:]}")
    log(f"driver: {r.stdout.strip()} ({len(models)} models: {', '.join(m.split('/')[-1] for m in models)})")


def pull_partial(pod, tags, local_dir):
    """Copy whatever in-progress jsonl/meta exist (resumable); used periodically and on every exit path."""
    r = pod.ssh("cd /workspace/election/experiments/sweep && ls *.jsonl 2>/dev/null", timeout=60)
    for f in r.stdout.split():
        tag = f[:-6]
        if tag in tags and not (local_dir / f"{tag}.DONE").exists():
            if pod.pull_file(f"/workspace/election/experiments/sweep/{f}", local_dir / "partial" / f):
                pod.pull_file(f"/workspace/election/experiments/sweep/{tag}.meta.json", local_dir / "partial" / f"{tag}.meta.json")
                n = sum(1 for _ in open(local_dir / "partial" / f))
                log(f"partial pull {tag}: {n} trials")


def pull_results(pod, tags, local_dir, pulled):
    """pulled: dict tag -> state already pulled; a FAIL that later turns DONE (rerun) is pulled again."""
    for tag in tags:
        if pulled.get(tag) == "DONE":
            continue
        r = pod.ssh(f"ls /workspace/election/experiments/sweep/{tag}.DONE /workspace/election/experiments/sweep/{tag}.FAIL 2>/dev/null", timeout=60)
        if ".DONE" in r.stdout and pulled.get(tag) == "FAIL":
            (local_dir / f"{tag}.FAIL").unlink(missing_ok=True)
        if (".DONE" in r.stdout and pulled.get(tag) != "DONE") or (".FAIL" in r.stdout and tag not in pulled):
            if PCT_MODE:
                pdir = ROOT / "experiments" / "pct"
                (pdir / "logs").mkdir(parents=True, exist_ok=True)
                files = pod.ssh(f"cd /workspace/election/experiments/pct && ls *_{tag}.jsonl *_{tag}.jsonl.meta.json {tag}_*.jsonl {tag}_*.jsonl.meta.json 2>/dev/null; "
                                f"cd /workspace/election/logs && ls *_{tag}.log 2>/dev/null | sed 's|^|logs/|'", timeout=60).stdout.split()
                n = 0
                for f in files:
                    dest = pdir / "logs" / Path(f).name if f.startswith("logs/") else pdir / f
                    src = f"/workspace/election/{f}" if f.startswith("logs/") else f"/workspace/election/experiments/pct/{f}"
                    if pod.pull_file(src, dest) and f.endswith(".jsonl"):
                        n += sum(1 for _ in open(dest))
                ok = n > 0
                pod.pull_file(f"/workspace/election/logs/{tag}.log", local_dir / "logs" / f"pct-{tag}.log")
            else:
                ok = pod.pull_file(f"/workspace/election/experiments/sweep/{tag}.jsonl", local_dir / f"{tag}.jsonl")
                pod.pull_file(f"/workspace/election/experiments/sweep/{tag}.meta.json", local_dir / f"{tag}.meta.json")
                pod.pull_file(f"/workspace/election/logs/{tag}.log", local_dir / "logs" / f"{tag}.log")
                n = sum(1 for _ in open(local_dir / f"{tag}.jsonl")) if ok and (local_dir / f"{tag}.jsonl").exists() else 0
            state = "DONE" if ".DONE" in r.stdout else "FAIL"
            (local_dir / f"{tag}.{state}").write_text("")
            log(f"{state} {tag}: pulled {n} trials" + ("" if ok else " (no jsonl)"))
            pulled[tag] = state
            if state == "FAIL":
                tail = pod.ssh(f"tail -c 1500 /workspace/election/logs/{tag}.log", timeout=60).stdout
                log(f"  failure tail: {tail.strip()[-1200:]}")


def monitor(pod, pid, tags, local_dir, max_hours, created_at):
    pulled = {}
    deadline = created_at + 3600 * (max_hours - 0.4)   # fire before the pod's own auto-terminate
    (local_dir / "partial").mkdir(exist_ok=True)
    fails = 0
    polls = 0
    last_status = ""
    while time.time() < deadline:
        polls += 1
        try:
            r = pod.ssh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | paste -sd ';' -; "
                        "tail -1 /workspace/election/logs/driver.log 2>/dev/null; "
                        "f=$(ls -t /workspace/election/logs/*.log 2>/dev/null | grep -v driver | head -1); [ -n \"$f\" ] && tail -1 $f; "
                        "ls /workspace/election/experiments/sweep/ALL_DONE 2>/dev/null; "
                        "pgrep -x -f 'bash /workspace/run.sh' >/dev/null || echo DRIVER_NOT_RUNNING; df -h / | tail -1 | awk '{print \"disk \" $3 \"/\" $2}'; echo POLL_OK", timeout=90)
            if "POLL_OK" not in r.stdout:
                raise RuntimeError(r.stderr[-200:] or f"rc={r.returncode}")
            if "DRIVER_NOT_RUNNING" in r.stdout and "ALL_DONE" not in r.stdout:
                log("driver process is gone without ALL_DONE; pulling what exists and stopping")
                pull_results(pod, tags, local_dir, pulled)
                tail = pod.ssh("tail -c 1200 /workspace/run.out; tail -3 /workspace/election/logs/driver.log", timeout=60).stdout
                log(f"run.out tail: {tail.strip()[-1200:]}")
                return False
            fails = 0
            status = " | ".join(l.strip() for l in r.stdout.splitlines() if l.strip())
            if status != last_status:
                log(status)
                last_status = status
            pull_results(pod, tags, local_dir, pulled)
            if polls % 5 == 0:
                pull_partial(pod, tags, local_dir)
            if "ALL_DONE" in r.stdout:
                log("all models finished")
                return True
        except Exception as e:
            fails += 1
            log(f"poll failed ({fails}): {str(e)[:200]}")
            if fails >= 8:
                st, ep, _ = pod_endpoint(pid)
                log(f"pod status via API: {st} {ep}")
                if st == "gone":
                    return False
                if ep and ep != (pod.host, pod.port):
                    pod.host, pod.port = ep
        time.sleep(120)
    log("max hours reached; pulling partial results")
    try:
        pull_results(pod, tags, local_dir, pulled)
        pull_partial(pod, tags, local_dir)
    except Exception as e:
        log(f"final pull failed: {e}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", required=True, choices=list(ROSTER["pods"]))
    ap.add_argument("--models", default=None, help="override the pod's model list")
    ap.add_argument("--attach", default=None, help="existing pod id to reuse (skips create+setup unless --setup)")
    ap.add_argument("--setup", action="store_true", help="with --attach: push files and start the driver")
    ap.add_argument("--keep", action="store_true", help="do not terminate the pod at the end")
    ap.add_argument("--no_retry", action="store_true", help="do not re-run FAILed models once at the end")
    args = ap.parse_args()
    spec = dict(ROSTER["pods"][args.pod])
    global PCT_MODE
    PCT_MODE = spec.get("driver") == "pct"
    models = args.models.split(",") if args.models else spec["models"]
    tags = [next((x["tag"] for x in ROSTER["models"] if x["id"] == m), m.split("/")[-1]) for m in models]
    name = f"julian-election-sweep-{args.pod}"
    local_dir = ROOT / "experiments" / "sweep"
    (local_dir / "logs").mkdir(parents=True, exist_ok=True)
    pod = Pod(name)
    created_at = time.time()
    if args.attach:
        pid = args.attach
        pname = assert_mine(pid)
        if pname != name:
            raise SystemExit(f"pod {pid} is named {pname!r}, expected {name!r}")
        d = gql('query{ pod(input:{podId:"%s"}){ runtime{ uptimeInSeconds } } }' % pid)
        up = ((d.get("data") or {}).get("pod") or {}).get("runtime") or {}
        created_at = time.time() - float(up.get("uptimeInSeconds") or 0)
        wait_ssh(pid, pod, minutes=5)
        if args.setup:
            setup(pod, models, spec, name)
    else:
        existing = [p for p in my_pods() if p["name"] == name]
        if existing:
            raise SystemExit(f"a pod named {name} already exists: {existing}; use --attach")
        for attempt in range(3):   # a host that maps no public TCP port is unusable: destroy it and buy another
            pid = create_pod(spec, name, spec["hours"])
            try:
                wait_ssh(pid, pod)
                setup(pod, models, spec, name)
                break
            except NoSshPort as e:
                log(f"{e}; terminating and recreating ({attempt + 1}/3)")
                terminate(pid)
                if attempt == 2:
                    raise
            except Exception as e:
                log(f"setup failed: {e}")
                terminate(pid)
                raise
    ok = monitor(pod, pid, tags, local_dir, spec["hours"], created_at)
    if ok and not args.no_retry:
        failed = [m for m, t in zip(models, tags) if (local_dir / f"{t}.FAIL").exists() and not (local_dir / f"{t}.DONE").exists()]
        if failed:
            log(f"retrying {len(failed)} failed model(s) once: {', '.join(failed)}")
            for t in [t for m, t in zip(models, tags) if m in failed]:
                (local_dir / f"{t}.FAIL").unlink(missing_ok=True)
            pod.ssh("cd /workspace/election/experiments/sweep && rm -f ALL_DONE " + " ".join(f"{t}.FAIL" for m, t in zip(models, tags) if m in failed), timeout=60)
            setup(pod, failed, spec, name)
            ok = monitor(pod, pid, [t for m, t in zip(models, tags) if m in failed], local_dir, spec["hours"], created_at)
    if not args.keep:
        terminate(pid)
    else:
        log(f"keeping pod {pid} ({pod.host}:{pod.port})")
    log("finished" if ok else "finished with problems")


if __name__ == "__main__":
    main()
