#!/usr/bin/env python3
"""Validate data/pct_issues.json: schema, level presence, report validity,
word-count balance between the agree/disagree statements, verbatim copies of
the four original axes from data/issues.json, and a printable summary table."""

import json
import re
import sys

REPO = "/Users/julianquick/election"
PCT_PATH = f"{REPO}/data/pct_issues.json"
ISSUES_PATH = f"{REPO}/data/issues.json"

DOMAINS = {"country_world", "economy", "social_values", "wider_society", "religion", "sex"}
DOMAIN_RANGE = {
    "country_world": range(1, 8),
    "economy": range(8, 22),
    "social_values": range(22, 40),
    "wider_society": range(40, 52),
    "religion": range(52, 57),
    "sex": range(57, 63),
}
COPIED_AXES = ["property_tax", "agi_limitations", "abortion", "state_surveillance"]
BALANCE_LIMIT = 6

problems = []


def wordcount(s):
    return len(re.findall(r"[\w'$-]+", s))


def main():
    with open(PCT_PATH) as f:
        pct = json.load(f)
    with open(ISSUES_PATH) as f:
        issues = json.load(f)

    # top-level keys mirror data/issues.json
    for key in ("description", "focal_axes", "n_fillers", "question", "axes"):
        if key not in pct:
            problems.append(f"missing top-level key: {key}")
    if pct.get("question") != issues["question"]:
        problems.append("top-level 'question' differs from data/issues.json")
    if pct.get("n_fillers") != issues["n_fillers"]:
        problems.append("top-level 'n_fillers' differs from data/issues.json")

    axes = pct["axes"]

    # exactly the 62 pct axes plus the four copied originals
    pct_keys = [k for k in axes if k.startswith("pct_")]
    ids = sorted(int(k.split("_")[1]) for k in pct_keys)
    if ids != list(range(1, 63)):
        problems.append(f"pct ids are not exactly 1..62: missing {sorted(set(range(1,63)) - set(ids))}, extra {sorted(set(ids) - set(range(1,63)))}")
    if set(k for k in axes if not k.startswith("pct_")) != set(COPIED_AXES):
        problems.append("non-pct axes are not exactly the four originals")

    # the four copied axes match data/issues.json exactly
    for key in COPIED_AXES:
        if axes.get(key) != issues["axes"].get(key):
            problems.append(f"copied axis {key} differs from data/issues.json")

    imbalance = []
    rows = []
    for k in sorted(pct_keys, key=lambda k: int(k.split("_")[1])):
        ax = axes[k]
        i = int(k.split("_")[1])

        for field in ("topic", "focal_phrase", "proposition", "pct_domain", "pct_axis",
                      "agree_sign", "sign_note", "levels", "report", "poles"):
            if field not in ax:
                problems.append(f"{k}: missing field {field}")

        levels = ax.get("levels", {})
        for lvl in ("agree", "disagree"):
            if lvl not in levels:
                problems.append(f"{k}: missing level {lvl}")
                continue
            stmt = levels[lvl]
            if not (stmt.startswith("Supports ") or stmt.startswith("Opposes ")
                    or stmt.startswith("Believes ")):
                problems.append(f"{k}/{lvl}: does not start with Supports/Opposes/Believes")
            if not re.search(r"[.!?]$", stmt) or stmt.count(".") > 1:
                problems.append(f"{k}/{lvl}: not a single sentence ending in one period")

        report = ax.get("report")
        if report not in levels:
            problems.append(f"{k}: report level '{report}' not in levels")

        poles = ax.get("poles", {})
        if set(poles) != set(levels):
            problems.append(f"{k}: poles keys {sorted(poles)} != levels keys {sorted(levels)}")
        for pole, label in poles.items():
            n = wordcount(label)
            if not 2 <= n <= 5:
                problems.append(f"{k}: pole '{pole}' label has {n} words (want 3-5)")

        if ax.get("pct_domain") not in DOMAINS:
            problems.append(f"{k}: bad pct_domain {ax.get('pct_domain')!r}")
        elif i in DOMAIN_RANGE[ax["pct_domain"]] is False:
            problems.append(f"{k}: id {i} outside domain range for {ax['pct_domain']}")
        if ax.get("pct_axis") not in ("economic", "social"):
            problems.append(f"{k}: bad pct_axis {ax.get('pct_axis')!r}")
        expected_axis = "economic" if ax.get("pct_domain") == "economy" else "social"
        if ax.get("pct_axis") != expected_axis:
            problems.append(f"{k}: pct_axis {ax.get('pct_axis')} != expected {expected_axis} for domain {ax.get('pct_domain')}")
        if ax.get("agree_sign") not in (1, -1):
            problems.append(f"{k}: bad agree_sign {ax.get('agree_sign')!r}")

        # proposition must appear verbatim in the entry
        if ax.get("proposition") and ax["proposition"] not in PROPOSITIONS.get(i, ""):
            problems.append(f"{k}: proposition text does not match the PCT list for id {i}")

        if "agree" in levels and "disagree" in levels:
            na, nd = wordcount(levels["agree"]), wordcount(levels["disagree"])
            if abs(na - nd) > BALANCE_LIMIT:
                imbalance.append((k, na, nd))
                problems.append(f"{k}: word counts differ by {abs(na-nd)} ({na} vs {nd})")

        rows.append((i, k, ax.get("pct_axis"), ax.get("agree_sign"),
                     ax.get("levels", {}).get("agree", ""), ax.get("levels", {}).get("disagree", "")))

    print(f"axes: {len(axes)} total, {len(pct_keys)} pct + {len(COPIED_AXES)} copied originals")
    if imbalance:
        print(f"\nword-count imbalance (> {BALANCE_LIMIT} words): {len(imbalance)}")
        for k, na, nd in imbalance:
            print(f"  {k}: agree {na} vs disagree {nd}")
    else:
        print(f"word-count imbalance (> {BALANCE_LIMIT} words): none")

    diffs = [abs(wordcount(r[4]) - wordcount(r[5])) for r in rows]
    print(f"word-count |agree - disagree|: max {max(diffs)}, mean {sum(diffs)/len(diffs):.2f}")

    print(f"\n{'id':>3} {'key':<8} {'axis':<8} {'sign':>4}  agree / disagree")
    for i, k, axis, sign, a, d in rows:
        print(f"{i:>3} {k:<8} {axis:<8} {sign:>+4}  {a} | {d}")

    if problems:
        print(f"\nFAILURES ({len(problems)}):")
        for p in problems:
            print("  -", p)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


# verbatim PCT proposition list, used to check the 'proposition' fields
PROPOSITIONS = {
    1: "If economic globalisation is inevitable, it should primarily serve humanity rather than the interests of trans-national corporations.",
    2: "I'd always support my country, whether it was right or wrong.",
    3: "No one chooses their country of birth, so it's foolish to be proud of it.",
    4: "Our race has many superior qualities, compared with other races.",
    5: "The enemy of my enemy is my friend.",
    6: "Military action that defies international law is sometimes justified.",
    7: "There is now a worrying fusion of information and entertainment.",
    8: "People are ultimately divided more by class than by nationality.",
    9: "Controlling inflation is more important than controlling unemployment.",
    10: "Because corporations cannot be trusted to voluntarily protect the environment, they require regulation.",
    11: "\"from each according to his ability, to each according to his need\" is a fundamentally good idea.",
    12: "The freer the market, the freer the people.",
    13: "It's a sad reflection on our society that something as basic as drinking water is now a bottled, branded consumer product.",
    14: "Land shouldn't be a commodity to be bought and sold.",
    15: "It is regrettable that many personal fortunes are made by people who simply manipulate money and contribute nothing to their society.",
    16: "Protectionism is sometimes necessary in trade.",
    17: "The only social responsibility of a company should be to deliver a profit to its shareholders.",
    18: "The rich are too highly taxed.",
    19: "Those with the ability to pay should have access to higher standards of medical care.",
    20: "Governments should penalise businesses that mislead the public.",
    21: "A genuine free market requires restrictions on the ability of predator multinationals to create monopolies.",
    22: "Abortion, when the woman's life is not threatened, should always be illegal.",
    23: "All authority should be questioned.",
    24: "An eye for an eye and a tooth for a tooth.",
    25: "Taxpayers should not be expected to prop up any theatres or museums that cannot survive on a commercial basis.",
    26: "Schools should not make classroom attendance compulsory.",
    27: "All people have their rights, but it is better for all of us that different sorts of people should keep to their own kind.",
    28: "Good parents sometimes have to spank their children.",
    29: "It's natural for children to keep some secrets from their parents.",
    30: "Possessing marijuana for personal use should not be a criminal offence.",
    31: "The prime function of schooling should be to equip the future generation to find jobs.",
    32: "People with serious inheritable disabilities should not be allowed to reproduce.",
    33: "The most important thing for children to learn is to accept discipline.",
    34: "There are no savage and civilised peoples; there are only different cultures.",
    35: "Those who are able to work, and refuse the opportunity, should not expect society's support.",
    36: "When you are troubled, it's better not to think about it, but to keep busy with more cheerful things.",
    37: "First-generation immigrants can never be fully integrated within their new country.",
    38: "What's good for the most successful corporations is always, ultimately, good for all of us.",
    39: "No broadcasting institution, however independent its content, should receive public funding.",
    40: "Our civil liberties are being excessively curbed in the name of counter-terrorism.",
    41: "A significant advantage of a one-party state is that it avoids all the arguments that delay progress in a democratic political system.",
    42: "Although the electronic age makes official surveillance easier, only wrongdoers need to be worried.",
    43: "The death penalty should be an option for the most serious crimes.",
    44: "In a civilised society, one must always have people above to be obeyed and people below to be commanded.",
    45: "Abstract art that doesn't represent anything shouldn't be considered art at all.",
    46: "In criminal justice, punishment should be more important than rehabilitation.",
    47: "It is a waste of time to try to rehabilitate some criminals.",
    48: "The businessperson and the manufacturer are more important than the writer and the artist.",
    49: "Mothers may have careers, but their first duty is to be homemakers.",
    50: "Almost all politicians promise economic growth, but we should heed the warnings of climate science that growth is detrimental to our efforts to curb global warming.",
    51: "Making peace with the establishment is an important aspect of maturity.",
    52: "Astrology accurately explains many things.",
    53: "You cannot be moral without being religious.",
    54: "Charity is better than social security as a means of helping the genuinely disadvantaged.",
    55: "Some people are naturally unlucky.",
    56: "It is important that my child's school instills religious values.",
    57: "Sex outside marriage is usually immoral.",
    58: "A same sex couple in a stable, loving relationship should not be excluded from the possibility of child adoption.",
    59: "Pornography, depicting consenting adults, should be legal for the adult population.",
    60: "What goes on in a private bedroom between consenting adults is no business of the state.",
    61: "No one can feel naturally homosexual.",
    62: "These days openness about sex has gone too far.",
}

if __name__ == "__main__":
    sys.exit(main())
