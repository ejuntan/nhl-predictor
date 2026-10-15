"""The puck line and the overtime price: calibrated, or just constant?

Section 15 showed the expected total is calibrated on average and carries
almost no information, which calibration alone cannot detect. These are the
two remaining derived markets on the page, and they deserve the same test:

  does the probability move with the game, and when it moves, is it right?

Three measures per market.

  skill        Brier against always quoting the base rate. Positive means
               the forecast beats a constant.
  slope        the logistic slope of outcome on the model's own log-odds.
               1.0 means the stated probabilities carry their stated
               information; 0 means none.
  bands        realised rate against predicted, with a 95% interval.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

CLEAN = ["20252026", "20262027"]


def assess(y, p, name, bins):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    base = np.full(len(y), y.mean())
    bs, bb = brier_score_loss(y, p), brier_score_loss(y, base)
    z = np.log(p / (1 - p))
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(z.reshape(-1, 1), y)
    print("\n" + "=" * 70)
    print(f"{name}   (n={len(y)})")
    print("=" * 70)
    print(f"  predicted on average {p.mean():.4f}   actual {y.mean():.4f}   "
          f"gap {p.mean() - y.mean():+.4f}")
    print(f"  spread of the forecast: sd {p.std(ddof=1):.4f}  "
          f"range {p.min():.3f} to {p.max():.3f}")
    print(f"  Brier {bs:.5f} against {bb:.5f} for always quoting the base "
          f"rate   skill {1 - bs / bb:+.4f}")
    print(f"  calibration slope {lr.coef_[0][0]:+.3f}   "
          f"intercept {lr.intercept_[0]:+.3f}")
    d = pd.DataFrame({"p": p, "y": y})
    d["bin"] = pd.cut(d.p, bins)
    g = d.groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    g["gap"] = g.actual - g.predicted
    g["moe"] = 1.96 * np.sqrt(g.actual * (1 - g.actual) / g.n)
    g["off"] = np.where(g.gap.abs() > g.moe, "<-- outside", "")
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))
    inside = int((g.gap.abs() <= g.moe).sum())
    print(f"  bands inside their interval: {inside}/{len(g)}")
    return 1 - bs / bb, lr.coef_[0][0]


def main():
    bt = pd.read_pickle("data/backtest_preds.pkl")
    cl = bt[bt.season.isin(CLEAN)].copy()

    r = {}
    r["puck line"] = assess(
        (cl.home_score - cl.away_score >= 2).astype(int), cl.p_home_m15,
        "HOME -1.5 (home wins by two or more)",
        [0, .15, .2, .25, .3, .35, .45, 1.0])
    r["overtime"] = assess(
        cl.went_ot, cl.p_ot, "GOES TO OVERTIME",
        [0, .2, .22, .24, .26, .28, 1.0])
    # Reference points already established, recomputed the same way.
    r["moneyline"] = assess(cl.home_win, cl.p_home, "MONEYLINE, for scale",
                            [0, .35, .45, .5, .55, .6, .65, .7, 1.0])
    r["over 5.5"] = assess((cl.total_goals > 5.5).astype(int), cl.p_over_55,
                           "OVER 5.5 GOALS, for scale",
                           [0, .5, .53, .56, .59, 1.0])

    print("\n" + "=" * 70)
    print("EVERY DERIVED MARKET, SIDE BY SIDE")
    print("=" * 70)
    print(f"  {'market':14} {'Brier skill':>12} {'slope':>8}")
    for k, (s, sl) in r.items():
        print(f"  {k:14} {s:+12.4f} {sl:+8.3f}")
    print("\n  Brier skill is the share of the base rate's error removed.")
    print("  A slope near zero means the number moves without meaning it.")


if __name__ == "__main__":
    main()
