"""Shared helpers: profile rendering and prompt construction.

Kept free of torch so it can be unit-tested locally.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_data():
    return (
        json.loads((DATA / "stances.json").read_text()),
        json.loads((DATA / "candidates.json").read_text()),
        json.loads((DATA / "prompts.json").read_text()),
    )


def render_profile(cand, name, template, rng):
    """Render a candidate profile with a random name and shuffled position order."""
    items = list(cand["positions"].values())
    rng.shuffle(items)
    positions = "\n".join(f"- {s}" for s in items)
    return template.format(name=name, positions=positions)


def build_jobs(n_samples, seed=0):
    """Return the list of every generation job with full metadata.

    Each job: prompt text plus everything the analysis needs to pair responses.
    Names are drawn without replacement per job so both candidates in a pairwise
    prompt never share a name.
    """
    stances, cands, prompts = load_data()
    rng = random.Random(seed)
    names = stances["name_pool"]["names"]
    by_id = {c["id"]: c for c in cands["candidates"]}
    tmpl = prompts["profile_template"]
    system = prompts["system"]
    jobs = []

    for p in prompts["prompts"]:
        if p["kind"] == "pairwise":
            for pair in cands["matched_pairs"]:
                a_id, b_id = pair["members"]
                for order in (0, 1):
                    first, second = (a_id, b_id) if order == 0 else (b_id, a_id)
                    for s in range(n_samples):
                        na, nb = rng.sample(names, 2)
                        text = p["text"].format(
                            profile_a=render_profile(by_id[first], na, tmpl, rng),
                            profile_b=render_profile(by_id[second], nb, tmpl, rng),
                        )
                        jobs.append({
                            "prompt_id": p["id"], "kind": p["kind"], "sample": s,
                            "flipped_axis": pair["flipped_axis"],
                            "first_id": first, "second_id": second,
                            "first_level": by_id[first]["cell"][pair["flipped_axis"]],
                            "second_level": by_id[second]["cell"][pair["flipped_axis"]],
                            "first_name": na, "second_name": nb,
                            "property_tax_statement": by_id[first]["positions"]["property_tax"],
                            "ground_truth": by_id[first]["ground_truth"],
                            "system": system, "user": text,
                        })
        else:
            for c in cands["candidates"]:
                for s in range(n_samples):
                    name = rng.choice(names)
                    text = p["text"].format(profile=render_profile(c, name, tmpl, rng))
                    jobs.append({
                        "prompt_id": p["id"], "kind": p["kind"], "sample": s,
                        "candidate_id": c["id"], "cell": c["cell"], "name": name,
                        "property_tax_statement": c["positions"]["property_tax"],
                        "ground_truth": c["ground_truth"],
                        "ground_truth_field": p.get("ground_truth_field"),
                        "system": system, "user": text,
                    })
    return jobs


if __name__ == "__main__":
    jobs = build_jobs(2)
    print(len(jobs), "jobs")
    print(jobs[0]["user"])
    print("---")
    print([j for j in jobs if j["kind"] == "pairwise"][0]["user"])
