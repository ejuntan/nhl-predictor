"""The puck line is the best game-level market, and it is biased low.

Measured on the clean holdout it removes 2.79% of the base rate's error,
against the moneyline's 1.81%, with all seven probability bands inside their
intervals and a calibration slope of 1.111. The one fault is level: it says
27.3% and the home side wins by two or more 30.2% of the time.

That is the same shape of fault as the saves model in section 17 -- the
ordering is right, the centre is off -- so it takes the same fix: a causal
offset from games already played, never fitted on the games being scored.

The offset is applied in log-odds, because a flat percentage added to a 10%
and a 50% price is not the same adjustment.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

CLEAN = ["20252026", "20262027"]
WINDOW = 400


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def causal_offset(p, y, window=WINDOW, warmup=150):
    """One log-odds shift per game, from the trailing window only."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    z = logit(p)
    out = np.empty_like(p)
    for i in range(len(p)):
        if i < warmup:
            out[i] = p[i]
            continue
        lo, hi = max(0, i - window), i
        zz, yy = z[lo:hi], y[lo:hi]
        # The shift that makes the window's predicted rate match its actual.
        best, bll = 0.0, np.inf
        for d in np.arange(-0.6, 0.61, 0.02):
            q = 1 / (1 + np.exp(-(zz + d)))
            ll = -np.mean(yy * np.log(q) + (1 - yy) * np.log(1 - q))
            if ll < bll:
                best, bll = d, ll
        out[i] = 1 / (1 + np.exp(-(z[i] + best)))
    return out


def report(y, cands, bins):
    base = np.full(len(y), np.mean(y))
    bb = brier_score_loss(y, base)
    print(f"{'':28} {'pred':>7} {'actual':>7} {'Brier':>9} {'skill':>8} {'slope':>7} {'bands':>7}")
    for nm, p in cands:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        bs = brier_score_loss(y, p)
        lr = LogisticRegression(C=1e6, max_iter=2000).fit(
            logit(p).reshape(-1, 1), y)
        d = pd.DataFrame({"p": p, "y": y})
        d["bin"] = pd.cut(d.p, bins)
        g = d.groupby("bin", observed=True).agg(
            n=("y", "size"), pr=("p", "mean"), ac=("y", "mean"))
        moe = 1.96 * np.sqrt(g.ac * (1 - g.ac) / g.n)
        inside = int(((g.ac - g.pr).abs() <= moe).sum())
        print(f"{nm:28} {p.mean():7.4f} {np.mean(y):7.4f} {bs:9.5f} "
              f"{1 - bs / bb:+8.4f} {lr.coef_[0][0]:+7.3f} "
              f"{inside:>4}/{len(g)}")


def main():
    bt = pd.read_pickle("data/backtest_preds.pkl")
    bt = bt.sort_values(["date", "game_id"]).reset_index(drop=True)
    y_all = (bt.home_score - bt.away_score >= 2).astype(int).values
    adj_all = causal_offset(bt.p_home_m15.values, y_all)
    bt["p_m15_adj"] = adj_all

    cl = bt[bt.season.isin(CLEAN)]
    y = (cl.home_score - cl.away_score >= 2).astype(int).values
    bins = [0, .15, .2, .25, .3, .35, .45, 1.0]
    print("=" * 94)
    print(f"HOME -1.5 ON THE CLEAN HOLDOUT  (n={len(y)})")
    print("=" * 94)
    report(y, [("as shipped", cl.p_home_m15.values),
               ("causal offset, trailing 400", cl.p_m15_adj.values)], bins)

    print("\nby season:")
    for s, g in cl.groupby("season"):
        yy = (g.home_score - g.away_score >= 2).astype(int).values
        print(f"  {s}  n={len(g):4d}  actual {yy.mean():.4f}   "
              f"shipped {g.p_home_m15.mean():.4f}   "
              f"adjusted {g.p_m15_adj.mean():.4f}")
    bt.to_pickle("data/backtest_preds_m15.pkl")


if __name__ == "__main__":
    main()
