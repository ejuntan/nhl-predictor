"""Are volume markets more predictable than game outcomes?

Game outcomes are dominated by goaltending variance. Shot volume is not --
it is driven by pace and possession, which persist. This measures how much
of the variance in team shots-on-goal a simple model explains, against the
~2% of outcome error the win model removes.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

SEASONS = ["20222023", "20232024", "20242025", "20252026", "20262027"]
TEST = ["20252026", "20262027"]


def main():
    df = pd.read_pickle("data/features.pkl")
    df = df[df.game_type == 2].copy()
    tg = pd.read_pickle("data/team_games.pkl")

    # realised all-situations shots on goal per team-game
    sog = tg.groupby(["game_id", "team"]).agg(
        sf=("sf_5v5", "sum"), cf=("cf_5v5", "sum")).reset_index()
    rows = []
    for pre, col in (("h", "home"), ("a", "away")):
        t = df[["game_id", col, "season", "date"]].rename(columns={col: "team"})
        t = t.merge(sog, on=["game_id", "team"], how="left")
        t["side"] = pre
        rows.append(t)
    long = pd.concat(rows, ignore_index=True)

    # pre-game features for the shooting team and its opponent
    feats = []
    for pre, opp in (("h", "a"), ("a", "h")):
        sub = df[["game_id", "season", "date"]].copy()
        sub["side"] = pre
        sub["own_cf"] = df[f"{pre}_r_cf_5v5"]
        sub["own_sf"] = df[f"{pre}_r_sf_5v5"]
        sub["opp_ca"] = df[f"{opp}_r_ca_5v5"]
        sub["opp_sa"] = df[f"{opp}_r_sa_5v5"]
        sub["own_xgf"] = df[f"{pre}_r_xgf_5v5"]
        sub["opp_xga"] = df[f"{opp}_r_xga_5v5"]
        sub["is_home"] = 1 if pre == "h" else 0
        sub["b2b"] = df[f"{pre}_b2b"].fillna(0)
        sub["rest"] = df[f"{pre}_rest"].fillna(3).clip(0, 6)
        feats.append(sub)
    F = pd.concat(feats, ignore_index=True)
    d = long.merge(F, on=["game_id", "season", "date", "side"], how="inner")
    d = d.dropna(subset=["sf"])
    cols = ["own_cf", "own_sf", "opp_ca", "opp_sa", "own_xgf", "opp_xga",
            "is_home", "b2b", "rest"]

    preds = []
    for i, s in enumerate(SEASONS):
        if i == 0:
            continue
        tr, te = d[d.season.isin(SEASONS[:i])], d[d.season == s]
        if len(te) < 60 or len(tr) < 1000:
            continue
        mu, sd = tr[cols].values.mean(0), tr[cols].values.std(0) + 1e-9
        m = PoissonRegressor(alpha=1.0, max_iter=5000).fit(
            (tr[cols].values - mu) / sd, tr.sf.values)
        p = m.predict((te[cols].values - mu) / sd)
        preds.append(pd.DataFrame({"season": s, "y": te.sf.values, "p": p}))

    a = pd.concat(preds, ignore_index=True)
    cl = a[a.season.isin(TEST)]
    y, p = cl.y.values, cl.p.values
    base = np.full(len(y), y.mean())

    def r2(y, p):
        return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    print("TEAM 5v5 SHOT ATTEMPTS PER GAME (the volume side of props)")
    print(f"  games      {len(y)} team-games on the clean holdout")
    print(f"  actual     mean {y.mean():.2f}  sd {y.std():.2f}")
    print(f"  predicted  mean {p.mean():.2f}  sd {p.std():.2f}")
    print(f"  MAE        model {np.abs(y - p).mean():.3f}  "
          f"vs league mean {np.abs(y - base).mean():.3f}")
    print(f"  R-squared  {r2(y, p):.4f}  (share of variance explained)")
    print(f"  error cut  {100 * (1 - ((y-p)**2).mean() / ((y-base)**2).mean()):.2f}%")
    print()
    print("Same figure for the win model, for comparison:")
    print("  error cut  2.39%  (Brier 0.24346 vs 0.24946 baseline)")


if __name__ == "__main__":
    main()
