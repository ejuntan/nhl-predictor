"""Check the prop shortlist against the rule it claims to follow.

The shortlist is the one part of this page that tells a reader which bets to
place, so every property the holdout measured has to hold on the live list
as well. If the side were ever computed against the wrong direction of the
disagreement, the page would recommend the exact opposite of the model.
"""
import json
import sys

ok = True
FLOOR = 0.55


def ck(label, got, want, tol=5e-4):
    global ok
    good = abs(got - want) <= tol if isinstance(want, float) else got == want
    if not good:
        ok = False
    print(("PASS  " if good else "FAIL  ") + label + f"  got={got}  want={want}")


def breakeven(p):
    return int(-round(100 * p / (1 - p)) if p >= 0.5
               else round(100 * (1 - p) / p))


d = json.load(open("out/tool_data.json"))
tops = d.get("topProps", {})
ev = d.get("topPropsEvidence", {})
games = {g["d"] + "|" + g["a"] + g["h"] for g in d["games"]}
props = {}
for r in d["props"]:
    props[(r["d"], r["n"])] = r

print(f"-- prop shortlist: {len(tops)} days --")
ck("there is a shortlist at all", len(tops) > 0, True)
ck("evidence block present", bool(ev), True)

for day, picks in sorted(tops.items()):
    tag = day + ": "
    ck(tag + "at most 15", len(picks) <= 15, True)
    seen = set()
    for r in picks:
        who = r["n"]
        # one line per player: three unders on one man is one bet thrice
        ck(tag + who + " appears once", who in seen, False)
        seen.add(who)
        # the floor the holdout required
        ck(tag + who + " clears the 55% floor", r["p"] >= FLOOR, True)
        # the price shown must be the price that probability needs
        ck(tag + who + " break-even matches", r["be"], breakeven(r["p"]))
        # the side must point the same way as the disagreement it is built on
        ck(tag + who + " side follows the disagreement",
           r["p"] > r["b"], True)
        # over means the model expects more shots than the recent average,
        # under means fewer; a flipped sign here is the dangerous bug
        ck(tag + who + " direction matches expected shots",
           r["xs"] > r["es"], bool(r["ov"]))
        # the game it points at has to exist, or the jump lands nowhere
        away, home = (r["o"], r["t"]) if r["hm"] else (r["t"], r["o"])
        ck(tag + who + " game is on the slate",
           day + "|" + away + home in games, True)
        # and the player has to be in that day's prop table
        src = props.get((day, who))
        ck(tag + who + " is in the props payload", src is not None, True)
        if src is not None:
            ck(tag + who + " expected shots agree", r["xs"], src["xs"])
            key = {1.5: "p15", 2.5: "p25", 3.5: "p35"}[r["ln"]]
            want = src[key] if r["ov"] else 1 - src[key]
            ck(tag + who + " probability agrees", r["p"], round(want, 4))

# The evidence line is a claim about measured results; it must stay a claim
# the page can defend, not an advertisement.
if ev:
    ck("measured actual is below the model's claim",
       ev["actual"] < ev["model"], True)
    ck("break-even matches the measured rate", ev["be"], breakeven(ev["actual"]))
    ck("interval brackets the point estimate",
       ev["hi"] > ev["be"] > ev["lo"], True)

print("\nALL PROP CHECKS PASS" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
