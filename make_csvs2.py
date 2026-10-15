"""Datasets for the props model, the headroom study and the change log."""
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.metrics import brier_score_loss, log_loss

import props_model as P

OUT = "out/"
TEST = ["20252026", "20262027"]


def main():
    pr = pd.read_pickle("data/props_preds.pkl")
    cl = pr[pr.season.isin(TEST)]
    y, mu, base = cl.sog.values, cl.mu.values, cl.e_sog.values
    size = cl["size"].values

    # 1 -- props performance by line, against two baselines
    rows = []
    for L in P.LINES:
        hit = (y > L).astype(int)
        k = int(np.floor(L))
        pm = np.clip(1 - nbinom.cdf(k, size, size / (size + mu)), 1e-6, 1 - 1e-6)
        pb = np.clip(1 - nbinom.cdf(k, size, size / (size + base)), 1e-6, 1 - 1e-6)
        pl = np.full(len(hit), hit.mean())
        d = (hit - pb) ** 2 - (hit - pm) ** 2
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        rows.append({
            "line": f"Over {L} shots", "n": len(hit),
            "hit_rate": round(float(hit.mean()), 4),
            "model_brier": round(brier_score_loss(hit, pm), 5),
            "own_avg_brier": round(brier_score_loss(hit, pb), 5),
            "naive_brier": round(brier_score_loss(hit, pl), 5),
            "vs_naive_pct": round(100 * (1 - brier_score_loss(hit, pm)
                                         / brier_score_loss(hit, pl)), 3),
            "vs_own_avg_pct": round(100 * (1 - brier_score_loss(hit, pm)
                                           / brier_score_loss(hit, pb)), 3),
            "t_stat": round(float(t), 2)})
    pd.DataFrame(rows).to_csv(OUT + "props_perf.csv", index=False)

    # 2 -- props calibration on Over 2.5
    pm = np.clip(1 - nbinom.cdf(2, size, size / (size + mu)), 0, 1)
    d = pd.DataFrame({"p": pm, "y": (y > 2.5).astype(int)})
    d["b"] = pd.cut(d.p, [0, .15, .25, .35, .45, .55, .7, 1.0])
    g = d.groupby("b", observed=True).agg(n=("y", "size"), pred=("p", "mean"),
                                          obs=("y", "mean")).reset_index()
    g["band"] = g.b.astype(str)
    g["moe95"] = 1.96 * np.sqrt(g.obs * (1 - g.obs) / g.n)
    g["predicted"] = g.pred.round(4)
    g["observed"] = g.obs.round(4)
    g["moe95"] = g.moe95.round(4)
    g[["band", "n", "predicted", "observed", "moe95"]].to_csv(
        OUT + "props_calib.csv", index=False)

    # 3 -- live prop prices
    lp = pd.read_pickle("data/live_props.pkl")
    lp = lp[lp.exp_sog >= 1.2].copy()
    out = pd.DataFrame({
        "date": lp.date, "player": lp.player, "team": lp.team, "opp": lp.opp,
        "home": lp.is_home.map({1: "home", 0: "away"}),
        "exp_toi_min": lp.exp_toi_min.round(1),
        "exp_sog": lp.exp_sog.round(2),
        "over_1_5": lp["p_over_1.5"].round(4),
        "over_2_5": lp["p_over_2.5"].round(4),
        "over_3_5": lp["p_over_3.5"].round(4),
        "opp_shots_allowed_60": lp.opp_sa60.round(2),
        "opp_blocks_60": lp.opp_blk60.round(2)})
    out.sort_values(["date", "exp_sog"], ascending=[True, False]).to_csv(
        OUT + "props_live.csv", index=False)

    # 4 -- headroom study
    hr = [
        {"information": "Always pick the home team", "accuracy": 0.5240,
         "brier": 0.24956, "log_loss": 0.69228, "kind": "baseline",
         "note": "No model at all"},
        {"information": "What this model knows pre-game", "accuracy": 0.5460,
         "brier": 0.24450, "log_loss": 0.68190, "kind": "model",
         "note": "Rolling ratings, goalie, schedule"},
        {"information": "True full-season team strength", "accuracy": 0.5842,
         "brier": 0.23680, "log_loss": 0.66487, "kind": "oracle",
         "note": "Hindsight. Unreachable in advance"},
        {"information": "Published closing line", "accuracy": 0.5750,
         "brier": 0.23550, "log_loss": 0.66000, "kind": "market",
         "note": "Literature figure, not measured here"},
        {"information": "Plus the goalie's actual game", "accuracy": 0.8663,
         "brier": 0.09342, "log_loss": 0.30344, "kind": "oracle",
         "note": "The outcome itself. Shows what variance costs"},
    ]
    pd.DataFrame(hr).to_csv(OUT + "headroom.csv", index=False)

    # 5 -- what each change was worth
    ch = [
        {"change": "Real time on ice (per-60 rates)", "target": "Win model",
         "measured": "-0.00076 log loss on validation", "verdict": "kept",
         "why": "Rates are now physically correct; 5v5 xGF/60 = 2.52 matches the league"},
        {"change": "Retune tie rate on recent games", "target": "Win model",
         "measured": "OT 21.7% -> 25.0% vs 24.5% actual", "verdict": "kept",
         "why": "Fixed the largest structural miss; every close game was mispriced"},
        {"change": "Opponent-adjusted team ratings", "target": "Win model",
         "measured": "+0.001 log loss, t=0.77", "verdict": "dropped",
         "why": "Not significant. NHL schedules are near-balanced, so there is little bias to remove"},
        {"change": "Drop special teams and finishing residual",
         "target": "Win model", "measured": "-0.0003 log loss on validation",
         "verdict": "dropped", "why": "Too small to justify pruning against the reported holdout"},
        {"change": "Shots-on-goal prop model", "target": "New market",
         "measured": "13.3% of naive error removed, t=8.3 vs hand baseline",
         "verdict": "kept", "why": "Five times the signal of the win model, and a large enough sample to prove it"},
        {"change": "Rate model with ice-time offset", "target": "Props",
         "measured": "Beat player's own average by 0.79%", "verdict": "kept",
         "why": "The first specification was mis-specified and lost to the trivial baseline"},
    ]
    pd.DataFrame(ch).to_csv(OUT + "changes.csv", index=False)

    # 6 -- league time-on-ice context, newly measurable
    import sqlite3
    con = sqlite3.connect("data/nhl.sqlite3", timeout=60)
    tt = pd.read_sql("SELECT * FROM team_toi", con)
    pt = pd.read_sql("SELECT * FROM player_toi WHERE is_goalie=0", con)
    toi = [
        {"measure": "Five-on-five per game", "minutes": round(tt.toi_5v5.mean() / 60, 2),
         "note": "Was previously assumed, not measured"},
        {"measure": "Power play per game", "minutes": round(tt.toi_pp.mean() / 60, 2),
         "note": "Mirrors the opponent's penalty kill exactly"},
        {"measure": "Penalty kill per game", "minutes": round(tt.toi_pk.mean() / 60, 2),
         "note": "Varies three-fold between games"},
        {"measure": "Four-on-four and other", "minutes": round(tt.toi_other.mean() / 60, 2),
         "note": "Small but not zero"},
        {"measure": "Average skater, per game", "minutes": round(pt.toi_reg.mean() / 60, 2),
         "note": "36 skaters dress per game"},
        {"measure": "Top skater, per game", "minutes": round(pt.toi_reg.quantile(.999) / 60, 2),
         "note": "A first-pair defenceman"},
    ]
    pd.DataFrame(toi).to_csv(OUT + "icetime.csv", index=False)

    import os
    for f in sorted(os.listdir(OUT)):
        print(f"{f:20} {sum(1 for _ in open(OUT+f))-1:5} rows")


if __name__ == "__main__":
    main()
