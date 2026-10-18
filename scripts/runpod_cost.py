"""Spend on the sweep pods: current pods from the API plus the per-pod log history.

Usage: python3 scripts/runpod_cost.py
"""
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RATE = {"s1": 2.49, "s2": 2.49, "l": 4.98, "xl": 7.47}   # $/hr actually billed (secure cloud, Oct 2026; the API "lowestPrice" of $1.69 was not what we got)


def api_key():
    return re.search(r"rpa_[A-Za-z0-9]+", (Path.home() / ".super_lab_run.pod").read_text()).group(0)


def gql(query):
    req = urllib.request.Request(f"https://api.runpod.io/graphql?api_key={api_key()}",
                                 data=json.dumps({"query": query}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    pods = gql("query{ myself{ pods{ id name desiredStatus costPerHr runtime{ uptimeInSeconds } } } }")["data"]["myself"]["pods"]
    total = 0.0
    print("running sweep pods:")
    for p in pods:
        if not p["name"].startswith("julian-election-sweep-"):
            continue
        up = ((p.get("runtime") or {}).get("uptimeInSeconds") or 0) / 3600
        cost = up * p["costPerHr"]
        total += cost
        print(f"  {p['name']:28s} {p['id']}  {p['desiredStatus']:8s} ${p['costPerHr']:.2f}/h  up {up:.2f} h  = ${cost:.2f}")
    # earlier (terminated) attempts from the local logs: created -> terminated
    hist = 0.0
    for f in sorted((ROOT / "experiments" / "sweep" / "logs").glob("pod-*.log")):
        pod = f.name.split("-")[1].split(".")[0]
        txt = f.read_text()
        for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\] creating.*?\n(?:.*\n)*?\[(\d\d:\d\d:\d\d)\] pod \w+ terminated", txt):
            h0, h1 = [sum(int(x) * k for x, k in zip(t.split(":"), (3600, 60, 1))) for t in m.groups()]
            hist += max(0, h1 - h0) / 3600 * RATE.get(pod, 1.69)
    print(f"terminated attempts (from logs): ${hist:.2f}")
    print(f"running total so far: ${total + hist:.2f}")


if __name__ == "__main__":
    main()
