"""Collect everything the web page needs into one JSON file."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

import names as N

TE = ["20252026", "20262027"]


def m(y, p):
    y = np.asarray(y)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"n": int(len(y)), "acc": float(((p > .5) == (y == 1)).mean()),
            "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, p))}


def main():
    feats = pd.read_pickle("data/features.pkl")
    bt = pd.read_pickle("data/backtest_preds.pkl")
    live = pd.read_pickle("data/live_preds.pkl")
    nm = N.load()

    cl = bt[bt.season.isin(TE)]
    out = {}
    out["holdout"] = {
        "seasons": TE, "n": int(len(cl)),
        "model": m(cl.home_win, cl.p_home),
        "elo": m(cl.home_win, cl.elo_p_home),
        "log5": m(cl.home_win, cl.log5_p_home),
        "always_home": m(cl.home_win, np.full(len(cl), 0.536)),
    }
    out["by_season"] = []
    for s, g in bt.groupby("season"):
        out["by_season"].append({
            "season": s, "clean": s in TE,
            **{k: v for k, v in m(g.home_win, g.p_home).items()},
            "elo_brier": float(brier_score_loss(g.home_win, g.elo_p_home)),
            "base_brier": float(brier_score_loss(
                g.home_win, np.full(len(g), g.home_win.mean()))),
        })

    bins = [0, .35, .45, .5, .55, .6, .65, .7, 1.0]
    d = pd.DataFrame({"y": cl.home_win.values, "p": cl.p_home.values})
    d["b"] = pd.cut(d.p, bins)
    cal = d.groupby("b", observed=True).agg(n=("y", "size"), pred=("p", "mean"),
                                            obs=("y", "mean"))
    out["calibration"] = [{"bin": str(i), "n": int(r.n), "pred": float(r.pred),
                           "obs": float(r.obs)} for i, r in cal.iterrows()]

    conf = np.maximum(cl.p_home, 1 - cl.p_home)
    hit = ((cl.p_home > .5) == (cl.home_win == 1)).values
    cd = pd.DataFrame({"c": conf, "h": hit})
    cb = cd.groupby(pd.cut(cd.c, [.5, .55, .6, .65, .7, 1.0]),
                    observed=True).agg(n=("h", "size"), acc=("h", "mean"))
    out["confidence"] = [{"bin": str(i), "n": int(r.n), "acc": float(r.acc)}
                         for i, r in cb.iterrows()]

    # derived markets
    out["markets"] = {
        "ot_pred": float(cl.p_ot.mean()), "ot_act": float(cl.went_ot.mean()),
        "total_pred": float(cl.exp_total.mean()),
        "total_act": float(cl.total_goals.mean()),
    }
    for line, col in ((4.5, "p_over_45"), (5.5, "p_over_55"),
                      (6.5, "p_over_65")):
        y = (cl.total_goals > line).astype(int)
        out["markets"][f"over{line}"] = {
            "pred": float(cl[col].mean()), "act": float(y.mean()),
            **m(y, cl[col])}
    y15 = ((cl.home_score - cl.away_score) >= 2).astype(int)
    out["markets"]["home_m15"] = {"pred": float(cl.p_home_m15.mean()),
                                  "act": float(y15.mean()), **m(y15, cl.p_home_m15)}

    # league context per season
    reg = feats[feats.game_type == 2]
    ctx = []
    for s, g in reg.groupby("season"):
        pts = {}
        for r in g.itertuples(index=False):
            hw = r.home_score > r.away_score
            ot = r.went_ot == 1
            pts[r.home] = pts.get(r.home, 0) + (2 if hw else (1 if ot else 0))
            pts[r.away] = pts.get(r.away, 0) + (2 if not hw else (1 if ot else 0))
        gp = 2 * len(g) / 32
        v = np.array(list(pts.values())) / (2 * gp)
        ctx.append({"season": s, "n": int(len(g)),
                    "home_win": float(g.home_win.mean()),
                    "ot": float(g.went_ot.mean()),
                    "total": float(g.total_goals.mean()),
                    "parity_std": float(v.std())})
    out["context"] = ctx

    # upcoming games
    gm = []
    for r in live.itertuples(index=False):
        gm.append({
            "date": r.date, "start": r.start_utc,
            "home": r.home, "away": r.away,
            "p_home": float(r.p_home), "p_away": float(r.p_away),
            "p_reg_home": float(r.p_reg_home), "p_reg_tie": float(r.p_reg_tie),
            "p_reg_away": float(r.p_reg_away),
            "lam_h": float(r.lam_h), "lam_a": float(r.lam_a),
            "exp_total": float(r.exp_total), "p_ot": float(r.p_ot),
            "p_over_55": float(r.p_over_55), "p_over_65": float(r.p_over_65),
            "p_home_m15": float(r.p_home_m15),
            "p_away_m15": float(r.p_away_m15),
            "elo_home": float(r.elo_home), "elo_away": float(r.elo_away),
            "h_goalie": nm.get(str(int(r.h_goalie)), "TBD")
                        if pd.notna(r.h_goalie) else "TBD",
            "a_goalie": nm.get(str(int(r.a_goalie)), "TBD")
                        if pd.notna(r.a_goalie) else "TBD",
            "h_goalie_label": r.h_goalie_label,
            "a_goalie_label": r.a_goalie_label,
            "h_b2b": int(r.h_b2b), "a_b2b": int(r.a_b2b),
            "h_rest": float(r.h_rest), "a_rest": float(r.a_rest),
            "h_gsax": float(r.h_g_gsax_raw), "a_gsax": float(r.a_g_gsax_raw),
            "h_xg_net": float(r.h_r_xgf_5v5 - r.h_r_xga_5v5),
            "a_xg_net": float(r.a_r_xgf_5v5 - r.a_r_xga_5v5),
        })
    out["games"] = gm

    # current team ratings table
    import pickle
    st = pickle.load(open("data/current_state.pkl", "rb"))
    rank = []
    for t, s in st["team"].items():
        rank.append({"team": t,
                     "xg_net": s["xgf_5v5"] - s["xga_5v5"],
                     "xgf": s["xgf_5v5"], "xga": s["xga_5v5"],
                     "gf": s["gf_all"], "ga": s["ga_all"]})
    rank.sort(key=lambda r: -r["xg_net"])
    out["ratings"] = rank
    out["asof"] = st["asof"]

    # ablation + subsets (recomputed numbers pasted from analyze.py run)
    json.dump(out, open("data/page_data.json", "w"), indent=1)
    print("wrote data/page_data.json",
          f"{len(json.dumps(out))/1024:.1f} KB",
          f"| {len(gm)} upcoming games | holdout n={out['holdout']['n']}")


if __name__ == "__main__":
    main()
