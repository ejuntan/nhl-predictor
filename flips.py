"""Step 3: is each change making different decisions, and for good reasons?

Two failure modes look identical in a summary table. A feature that barely
moves the probabilities is redundant -- the information is already in the
model. A feature that moves many picks without improving them is noise or is
specified wrongly. Both show up as "no change in log loss".

So three things are measured per change: how far the probabilities move, how
often the predicted winner flips, and what happens on exactly those games.

On a flipped game the two models picked opposite sides, so exactly one of
them is right. That makes the flip set an exact binomial trial -- k wins out
of n flips against p=0.5 -- which needs no bootstrap and no normal
approximation.
"""
import numpy as np
import pandas as pd
from scipy import stats

from compare import REF, paired_bootstrap


def shift(pa, pb):
    d = np.abs(np.asarray(pa) - np.asarray(pb))
    return {"mean_abs_dp": d.mean(), "p90_abs_dp": np.quantile(d, .90),
            "max_abs_dp": d.max(), "share_gt_02": (d > .02).mean(),
            "share_gt_05": (d > .05).mean()}


def verdict(sh, nflip, k, pv, dll_lo, dll_hi):
    if sh["share_gt_02"] < 0.05:
        return "redundant: barely moves the probabilities"
    if nflip == 0:
        return "redundant: never changes a pick"
    if pv < 0.05 and k > nflip / 2 and dll_hi < 0:
        return "improvement: better picks and better probabilities"
    if pv < 0.05 and k < nflip / 2:
        return "harmful: flips picks the wrong way"
    if sh["share_gt_02"] > 0.25 and dll_lo < 0 < dll_hi:
        return "noise: moves a lot, decides nothing"
    return "no measurable effect"


def report(preds):
    ref = preds[REF]
    y = ref.y.values
    print(f"Reference: {REF}   n={len(y)} identical games\n")
    print("=" * 92)
    print("HOW FAR EACH CHANGE MOVES THE PROBABILITIES")
    print("=" * 92)
    rows = []
    for k, p in preds.items():
        if k == REF:
            continue
        sh = shift(p.p.values, ref.p.values)
        flips = (p.p > .5).values != (ref.p > .5).values
        rows.append({"change": k, **sh, "flips": int(flips.sum()),
                     "flip_rate": flips.mean()})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 92)
    print("WHAT HAPPENS ON THE GAMES WHERE THE PICK CHANGES")
    print("exactly one model is right on a flipped game, so this is an "
          "exact binomial test")
    print("=" * 92)
    for k, p in preds.items():
        if k == REF:
            continue
        pa, pb = p.p.values, ref.p.values
        flips = (pa > .5) != (pb > .5)
        n = int(flips.sum())
        sh = shift(pa, pb)
        b = paired_bootstrap(y, pa, pb)
        print(f"\n{k}")
        if n == 0:
            print("  no flips at all")
            print(f"  verdict: {verdict(sh, 0, 0, 1.0, *(0, 0))}")
            continue
        yy = y[flips]
        kk = int(((pa[flips] > .5) == (yy == 1)).sum())
        pv = stats.binomtest(kk, n, 0.5).pvalue
        lo, hi = stats.binomtest(kk, n, 0.5).proportion_ci(0.95)
        print(f"  {n} flips ({flips.mean():.1%} of games); variant right "
              f"{kk}/{n} = {kk/n:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"p={pv:.3f}")
        sub = paired_bootstrap(yy, pa[flips], pb[flips])
        print(f"  log loss on those games {sub['logloss']['delta']:+.4f}  "
              f"CI [{sub['logloss']['lo']:+.4f}, {sub['logloss']['hi']:+.4f}]")
        print(f"  log loss on all games   {b['logloss']['delta']:+.5f}  "
              f"CI [{b['logloss']['lo']:+.5f}, {b['logloss']['hi']:+.5f}]")
        print(f"  verdict: {verdict(sh, n, kk, pv, b['logloss']['lo'], b['logloss']['hi'])}")


if __name__ == "__main__":
    report(pd.read_pickle("data/compare_preds.pkl"))
