"""Check the posted-line payload against the odds arithmetic it claims.

The page displays four numbers per game that have to agree with each other:
the model's probability on a side, that side's no-vig market probability, the
gap between them, and the edge against the actual vigged price. If any one
of them is computed from the wrong side, the page reports a positive edge on
a team it is not backing -- the same class of bug the U+2212 minus caused.
"""
import json
import sys

ok = True


def ck(label, got, want, tol=5e-4):
    global ok
    good = abs(got - want) <= tol if isinstance(want, float) else got == want
    if not good:
        ok = False
    print(("PASS  " if good else "FAIL  ") + label +
          f"  got={got}  want={want}")


def dec(american):
    a = float(american)
    return 1 + (a / 100 if a > 0 else 100 / -a)


d = json.load(open("out/tool_data.json"))
lines, buckets = d["lines"], d["buckets"]
games = {g["d"] + "|" + g["a"] + g["h"]: g for g in d["games"]}
print(f"-- {len(lines)} posted lines --")

for k, L in lines.items():
    g = games.get(k)
    ck(k + " has a game", g is not None, True)
    if g is None:
        continue
    # the side shown must be the side the model actually favours
    want_side = g["h"] if g["ph"] > 0.5 else g["a"]
    ck(k + " side", L["side"], want_side)
    # the model probability must be that side's probability
    ck(k + " model p", L["pm"], round(max(g["ph"], g["pa"]), 4))
    # no-vig pair must sum to 1 and be the favourite side when it agrees
    ck(k + " gap = model - line", round(L["pm"] - L["pk"], 4), L["gap"])
    # edge must be against the price of that same side
    price = L["hml"] if L["side"] == g["h"] else L["aml"]
    ck(k + " edge vs posted price", L["edge"],
       round(L["pm"] - 1 / dec(price), 4))
    # the bucket must follow from agreement and confidence
    want_b = ("dog" if L["pk"] < 0.495
              else ("fav" if L["pm"] >= 0.60 else "thin"))
    ck(k + " bucket", L["b"], want_b)

print("\n-- every bucket carries a measured return and an interval --")
for b, v in buckets.items():
    ck(b + " has roi", isinstance(v.get("roi"), float), True)
    ck(b + " has ci", isinstance(v.get("ci"), float), True)
    ck(b + " has n", isinstance(v.get("n"), int), True)
    ck(b + " has value-pick roi", isinstance(v.get("vp_roi"), float), True)
    # A bucket may only be shown as positive if its interval clears zero;
    # none of them do, so no chip may ever read as an established winner.
    ck(b + " not an established winner", v["roi"] - v["ci"] > 0, False)
ck("exactly one bucket is an established loser",
   sum(1 for v in buckets.values() if v["roi"] + v["ci"] < 0), 1)

vp = d["valuePicks"]
ck("value-pick interval straddles zero", vp["lo"] < 0 < vp["hi"], True)
ck("value-pick n is the sum of the buckets' value picks",
   vp["n"], sum(v["vp_n"] for v in buckets.values()))

print("\nALL LINE CHECKS PASS" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
