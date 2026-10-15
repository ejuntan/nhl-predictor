"""Can the total estimates be improved? Bound it before building anything.

Three questions in order, cheapest first.

1. How much variance is there to forecast at all? If a game's total is
   Poisson around a rate that barely varies between games, then no model of
   any kind can do much, and the ceiling is arithmetic rather than an
   engineering problem.
2. What would perfect knowledge of both teams' season-long scoring and
   conceding rates buy? That is an oracle: it uses the future, so whatever
   it achieves is an upper bound no causal model can reach.
3. The regression of actual on predicted says whether the current estimates
   are too spread out or too compressed. If the slope is below 1 the model
   is overstating what it knows, and shrinking toward the league mean is a
   free improvement. Fitted on the validation seasons and tested on the
   clean holdout, never fitted on the holdout itself.
"""
import numpy as np
import pandas as pd
from scipy import stats

VALID = ["20232024", "20242025"]
CLEAN = ["20252026", "20262027"]


def variance_budget(y):
    """Split the variance in realised totals into Poisson noise and signal.

    For counts, Var(total) = E[rate] + Var(rate). Everything above the mean
    is between-game differences a forecaster could in principle see; the
    mean itself is irreducible noise.
    """
    v, m = y.var(ddof=1), y.mean()
    print("=" * 74)
    print("1. HOW MUCH IS THERE TO FORECAST?")
    print("=" * 74)
    print(f"  realised totals: mean {m:.3f}  variance {v:.3f}  sd {y.std(ddof=1):.3f}")
    print(f"  a Poisson count with that mean has variance {m:.3f}")
    print(f"  variance left for between-game rate differences: {v - m:+.3f}")
    if v <= m:
        print("\n  Negative. Totals are UNDER-dispersed relative to Poisson:")
        print("  they vary LESS than a constant-rate process would. There is")
        print("  no between-game signal to find, because there is no excess")
        print("  variance to attribute to one.")
        print("\n  The overtime rule is the mechanism: every game level after")
        print("  regulation gets exactly one more goal, never two or none,")
        print("  which truncates the top of the distribution by construction.")
    else:
        sd_rate = np.sqrt(v - m)
        print(f"\n  Implied sd of the true game rate: {sd_rate:.3f} goals.")
        print(f"  A perfect forecaster's R2 ceiling would be {(v - m) / v:.4f}.")
    return v, m


def oracle(df):
    """Upper bound: each team's own season-long scoring and conceding."""
    print("\n" + "=" * 74)
    print("2. WHAT WOULD PERFECT SEASON-LONG KNOWLEDGE BUY?")
    print("Uses the full season's realised rates, including the game being")
    print("predicted. No causal model can beat this.")
    print("=" * 74)
    rows = []
    for season, g in df.groupby("season"):
        gf = {}
        for side, other in (("home", "away"), ("away", "home")):
            for t, h in g.groupby(side):
                sc = h.home_score if side == "home" else h.away_score
                ga = h.away_score if side == "home" else h.home_score
                d = gf.setdefault(t, {"f": [], "a": []})
                d["f"] += list(sc.values)
                d["a"] += list(ga.values)
        lg_f = np.mean([np.mean(v["f"]) for v in gf.values()])
        pred = []
        for r in g.itertuples(index=False):
            h, a = gf.get(r.home), gf.get(r.away)
            if not h or not a:
                pred.append(np.nan)
                continue
            # Home scores at its own rate adjusted by the opponent's defence.
            ph = np.mean(h["f"]) * (np.mean(a["a"]) / lg_f)
            pa = np.mean(a["f"]) * (np.mean(h["a"]) / lg_f)
            pred.append(ph + pa)
        o = pd.DataFrame({"season": season, "y": g.total_goals.values,
                          "oracle": pred, "model": g.exp_total.values})
        rows.append(o)
    o = pd.concat(rows).dropna()
    mean = np.full(len(o), o.y.mean())
    for nm, p in (("oracle (uses the future)", o.oracle.values),
                  ("the model", o.model.values),
                  ("always the league average", mean)):
        mae = np.abs(o.y - p).mean()
        r2 = 1 - ((o.y - p) ** 2).sum() / ((o.y - mean) ** 2).sum()
        print(f"  {nm:28} MAE {mae:.4f}   R2 {r2:+.5f}   "
              f"sd of estimate {np.std(p, ddof=1):.3f}")
    print(f"\n  The oracle's own spread is {o.oracle.std(ddof=1):.3f} against "
          f"the {o.y.std(ddof=1):.3f} of real totals.")
    return o


def shrink(df):
    """Is the current estimate too wide or too narrow? Fit, then test."""
    print("\n" + "=" * 74)
    print("3. ARE THE ESTIMATES TOO SPREAD OUT?")
    print("Slope of actual on predicted. Below 1 means the model overstates")
    print("its own differences and should shrink toward the mean.")
    print("=" * 74)
    tr = df[df.season.isin(VALID)]
    te = df[df.season.isin(CLEAN)]
    sl, ic, r, p, se = stats.linregress(tr.exp_total, tr.total_goals)
    print(f"  fitted on {len(tr)} validation games: slope {sl:+.4f} "
          f"(se {se:.4f}), intercept {ic:+.3f}, p={p:.4f}")
    sl2, _, _, p2, se2 = stats.linregress(te.exp_total, te.total_goals)
    print(f"  the same fit on the {len(te)} holdout games: slope {sl2:+.4f} "
          f"(se {se2:.4f}), p={p2:.4f}")

    y = te.total_goals.values
    raw = te.exp_total.values
    mu = tr.total_goals.mean()
    adj = mu + sl * (raw - tr.exp_total.mean())
    mean = np.full(len(y), mu)
    print(f"\n  on the clean holdout:")
    for nm, pr in (("current estimate", raw),
                   ("shrunk by the fitted slope", adj),
                   ("the league average alone", mean)):
        mae = np.abs(y - pr).mean()
        r2 = 1 - ((y - pr) ** 2).sum() / ((y - mean) ** 2).sum()
        print(f"    {nm:28} MAE {mae:.4f}   R2 {r2:+.5f}")
    t, pv = stats.ttest_rel(np.abs(y - adj), np.abs(y - raw))
    print(f"\n  paired t, shrunk vs current: t={t:+.2f}  p={pv:.3f}")


if __name__ == "__main__":
    bt = pd.read_pickle("data/backtest_preds.pkl")
    cl = bt[bt.season.isin(CLEAN)]
    variance_budget(cl.total_goals.values.astype(float))
    oracle(bt)
    shrink(bt)
