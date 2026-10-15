"""Step 2: where the model is wrong, and whether the weakness repeats.

Overall accuracy says almost nothing about what to fix. This splits the
walk-forward predictions into situations and asks, per situation and per
season, whether realised accuracy falls short of the confidence the model
claimed. A category only counts as a real weakness if the shortfall has the
same sign in most seasons -- with ~1,300 games a season, one bad season is
the null hypothesis, not a finding.

Seasons 2023-24 and 2024-25 are reported alongside the clean holdout because
repeatability cannot be judged from two seasons. They are flagged: the xG
model underneath saw those outcomes, so their absolute level is optimistic.
Only the *pattern across* seasons is being read here.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

import compare as C
import variants as V

CACHE = "data/ref_preds_all.pkl"
CLEAN = set(C.CLEAN)


def build():
    frames = V.load()
    g = frames["goalie"]
    g = g[g.game_type == 2].copy()
    p = C.walk_forward(g, tie=True, extra=())        # shipped config
    keep = ["game_id", "reg_outcome", "went_ot", "h_b2b", "a_b2b", "h_rest",
            "a_rest",
            "h_games_last4", "a_games_last4", "a_travel_km", "h_share",
            "a_share", "h_backup", "a_backup", "total_goals"]
    p = p.merge(g[keep], on="game_id", how="left")
    od = pd.read_pickle("data/odds_joined.pkl")[
        ["game_id", "mkt_h", "home_ml", "away_ml"]]
    p = p.merge(od, on="game_id", how="left")
    p.to_pickle(CACHE)
    return p


def categories(p):
    conf = np.maximum(p.p, 1 - p.p)
    b2b = (p.h_b2b.fillna(0) > 0) | (p.a_b2b.fillna(0) > 0)
    g4 = (p.h_games_last4.fillna(0) >= 2) & (p.a_games_last4.fillna(0) >= 2)
    backup = (p.h_backup.fillna(0) > 0) | (p.a_backup.fillna(0) > 0)
    mkt_dog = p.mkt_h.notna() & (
        ((p.p > .5) & (p.mkt_h < .5)) | ((p.p < .5) & (p.mkt_h > .5)))
    return {
        "strong favourite (conf>=.65)": conf >= .65,
        "moderate favourite (.55-.65)": (conf >= .55) & (conf < .65),
        "coin flip (conf<.55)": conf < .55,
        "model backs the market dog": mkt_dog,
        "went to OT/SO": p.p_ot.notna() & (p.went_ot == 1),
        "level after regulation": p.reg_outcome == 1,
        "a backup/unusual starter": backup,
        "either side back-to-back": b2b,
        "both sides on a dense schedule": g4,
        "long away trip (>2000km)": p.a_travel_km.fillna(0) > 2000,
        "big rest edge (>=2 days)": (p.h_rest.fillna(3)
                                     - p.a_rest.fillna(3)).abs() >= 2,
    }


def _stats(sub):
    y, pp = sub.y.values, np.clip(sub.p.values, 1e-6, 1 - 1e-6)
    conf = np.maximum(pp, 1 - pp)
    hit = ((pp > .5) == (y == 1)).astype(float)
    ll = -(y * np.log(pp) + (1 - y) * np.log(1 - pp))
    return {"n": len(sub), "acc": hit.mean(), "exp_acc": conf.mean(),
            "gap": hit.mean() - conf.mean(), "logloss": ll.mean(),
            "_hit": hit, "_conf": conf}


def report(p):
    cats = categories(p)
    seasons = sorted(p.season.unique())

    print("=" * 92)
    print("CALIBRATION GAP BY SITUATION AND SEASON   "
          "(realised accuracy minus claimed confidence)")
    print("negative = overconfident there.  * = xG model saw that season's "
          "outcomes")
    print("=" * 92)
    rows = []
    for name, mask in cats.items():
        row = {"situation": name}
        for s in seasons:
            sub = p[mask & (p.season == s)]
            tag = s if s in CLEAN else s + "*"
            row[tag] = (f"{_stats(sub)['gap']:+.3f} ({len(sub)})"
                        if len(sub) >= 25 else f"-- ({len(sub)})")
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 92)
    print("POOLED, AND WHETHER THE SHORTFALL REPEATS")
    print("t is the within-category gap against the gap on all other games; "
          "seasons counts how many of the usable seasons share the sign")
    print("=" * 92)
    rows = []
    for name, mask in cats.items():
        sub, rest = p[mask], p[~mask]
        if len(sub) < 50:
            continue
        a, b = _stats(sub), _stats(rest)
        da = a["_hit"] - a["_conf"]
        db = b["_hit"] - b["_conf"]
        t, pv = stats.ttest_ind(da, db, equal_var=False)
        signs = []
        for s in seasons:
            ss = p[mask & (p.season == s)]
            if len(ss) >= 25:
                signs.append(np.sign(_stats(ss)["gap"]))
        same = int(sum(1 for x in signs if x == np.sign(a["gap"])))
        rows.append({"situation": name, "n": a["n"], "acc": a["acc"],
                     "exp": a["exp_acc"], "gap": a["gap"],
                     "logloss": a["logloss"], "t": t, "p": pv,
                     "seasons": f"{same}/{len(signs)}"})
    t = pd.DataFrame(rows).sort_values("gap")
    print(t.to_string(index=False, formatters={
        "acc": "{:.3f}".format, "exp": "{:.3f}".format,
        "gap": "{:+.4f}".format, "logloss": "{:.4f}".format,
        "t": "{:+.2f}".format, "p": "{:.3f}".format}))

    cl = p[p.season.isin(CLEAN) & p.mkt_h.notna()]
    if len(cl) > 100:
        print("\n" + "=" * 92)
        print(f"VERSUS THE MARKET, clean holdout with prices (n={len(cl)})")
        print("=" * 92)
        rows = []
        for name, mask in categories(cl).items():
            sub = cl[mask]
            if len(sub) < 40:
                continue
            y = sub.y.values
            mp = np.clip(sub.mkt_h.values, 1e-6, 1 - 1e-6)
            mo = np.clip(sub.p.values, 1e-6, 1 - 1e-6)
            f = lambda q: -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
            rows.append({
                "situation": name, "n": len(sub),
                "model acc": ((mo > .5) == (y == 1)).mean(),
                "market acc": ((mp > .5) == (y == 1)).mean(),
                "model ll": f(mo), "market ll": f(mp),
                "d_ll": f(mo) - f(mp)})
        print(pd.DataFrame(rows).sort_values("d_ll").to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    p = pd.read_pickle(CACHE) if os.path.exists(CACHE) else build()
    report(p)
