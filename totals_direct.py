"""The oracle leaves a 4.6% R2 gap and the model captures none of it.

That is the only question worth asking: is the gap unreachable, or is the
Poisson score grid throwing away information that a simpler estimate built
from the same features would keep? The grid exists to make the moneyline,
the puck line and the totals mutually consistent, and that constraint could
easily be costing the total its spread.

Two candidates, both walk-forward, both fit only on earlier seasons:

  rolling     home goals-for and away goals-against, and the reverse,
              straight out of the rating state with no model at all
  direct      a ridge regression trained to predict the total itself
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from model import build_matrix

SEASONS = ["20222023", "20232024", "20242025", "20252026", "20262027"]
CLEAN = ["20252026", "20262027"]


def rolling_estimate(df):
    """No model: what both teams have been scoring and conceding."""
    return ((df.h_r_gf_all + df.a_r_ga_all) / 2
            + (df.a_r_gf_all + df.h_r_ga_all) / 2).values


def run():
    df = pd.read_pickle("data/features.pkl")
    df = df[df.game_type == 2].copy()
    bt = pd.read_pickle("data/backtest_preds.pkl")[
        ["game_id", "exp_total"]]
    df = df.merge(bt, on="game_id", how="left")

    out = []
    for i, s in enumerate(SEASONS):
        if i == 0:
            continue
        tr, te = df[df.season.isin(SEASONS[:i])], df[df.season == s]
        if len(te) < 30 or len(tr) < 500:
            continue
        Xtr, Xte = build_matrix(tr).values, build_matrix(te).values
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        r = Ridge(alpha=50.0).fit((Xtr - mu) / sd, tr.total_goals.values)
        o = te[["game_id", "season", "total_goals", "exp_total"]].copy()
        o["direct"] = r.predict((Xte - mu) / sd)
        o["rolling"] = rolling_estimate(te)
        out.append(o)
    return pd.concat(out, ignore_index=True)


def report(o):
    cl = o[o.season.isin(CLEAN)].dropna(subset=["exp_total"])
    y = cl.total_goals.values.astype(float)
    mean = np.full(len(y), y.mean())
    print("=" * 74)
    print(f"PREDICTING THE TOTAL, CLEAN HOLDOUT  (n={len(y)})")
    print("=" * 74)
    print(f"{'':34} {'MAE':>7} {'R2':>9} {'sd':>7}")
    cands = (("the model's score grid", cl.exp_total.values),
             ("ridge trained on the total itself", cl.direct.values),
             ("rolling goals for and against", cl["rolling"].values),
             ("the league average", mean))
    for nm, p in cands:
        mae = np.abs(y - p).mean()
        r2 = 1 - ((y - p) ** 2).sum() / ((y - mean) ** 2).sum()
        print(f"{nm:34} {mae:7.4f} {r2:+9.5f} {np.std(p, ddof=1):7.3f}")

    print("\npaired t on absolute error, against the score grid:")
    for nm, p in cands[1:]:
        t, pv = stats.ttest_rel(np.abs(y - p), np.abs(y - cl.exp_total.values))
        better = "better" if np.abs(y - p).mean() < np.abs(y - cl.exp_total.values).mean() else "worse"
        print(f"  {nm:34} t={t:+.2f}  p={pv:.3f}  ({better})")

    print("\ncorrelation with the realised total:")
    for nm, p in cands[:-1]:
        print(f"  {nm:34} {np.corrcoef(p, y)[0, 1]:+.4f}")


if __name__ == "__main__":
    o = run()
    o.to_pickle("data/totals_direct.pkl")
    report(o)
