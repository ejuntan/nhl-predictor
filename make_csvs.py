"""Write the CSV datasets the dashboard reads."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

import names as N
import model as M
from analyze import GROUPS, score

TE = ["20252026", "20262027"]
OUT = "out/"


def mm(y, p):
    y = np.asarray(y)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return (float(((p > .5) == (y == 1)).mean()),
            float(brier_score_loss(y, p)), float(log_loss(y, p)))


def main():
    feats = pd.read_pickle("data/features.pkl")
    bt = pd.read_pickle("data/backtest_preds.pkl")
    live = pd.read_pickle("data/live_preds.pkl")
    nm = N.load()
    cl = bt[bt.season.isin(TE)].copy()

    # 1 ---- benchmarks on the clean holdout
    rows = []
    for label, col, note in (
            ("This model", "p_home", "xG + goalie + schedule, Poisson score grid"),
            ("Elo (goal differential)", "elo_p_home", "Margin-weighted Elo, nothing else"),
            ("Log5 on points %", "log5_p_home", "Standings-based, the naive approach"),
            ("Always pick home", None, "No model at all")):
        p = (np.full(len(cl), 0.536) if col is None else cl[col].values)
        a, b, l = mm(cl.home_win, p)
        rows.append({"method": label, "accuracy": round(a, 4),
                     "brier": round(b, 5), "log_loss": round(l, 5),
                     "games": len(cl), "note": note})
    # published market reference, clearly labelled as such
    rows.append({"method": "Closing line (published range)", "accuracy": 0.575,
                 "brier": 0.2355, "log_loss": 0.6600, "games": 0,
                 "note": "Literature benchmark, not measured here"})
    pd.DataFrame(rows).to_csv(OUT + "benchmarks.csv", index=False)

    # 2 ---- per-season performance and league context
    rows = []
    reg = feats[feats.game_type == 2]
    for s, g in bt.groupby("season"):
        ctx = reg[reg.season == s]
        pts = {}
        for r in ctx.itertuples(index=False):
            hw = r.home_score > r.away_score
            ot = r.went_ot == 1
            pts[r.home] = pts.get(r.home, 0) + (2 if hw else (1 if ot else 0))
            pts[r.away] = pts.get(r.away, 0) + (2 if not hw else (1 if ot else 0))
        gp = 2 * len(ctx) / 32
        v = np.array(list(pts.values())) / (2 * gp)
        a, b, l = mm(g.home_win, g.p_home)
        _, eb, _ = mm(g.home_win, g.elo_p_home)
        _, bb, _ = mm(g.home_win, np.full(len(g), g.home_win.mean()))
        rows.append({
            "season": f"{s[:4]}-{s[6:]}", "games": len(g),
            "holdout": "clean" if s in TE else "xG model saw it",
            "accuracy": round(a, 4), "brier": round(b, 5),
            "log_loss": round(l, 5), "elo_brier": round(eb, 5),
            "baseline_brier": round(bb, 5),
            "skill_vs_baseline": round(100 * (1 - b / bb), 3),
            "parity_std": round(float(v.std()), 4),
            "ot_rate": round(float(ctx.went_ot.mean()), 4),
            "goals_per_game": round(float(ctx.total_goals.mean()), 3)})
    pd.DataFrame(rows).to_csv(OUT + "seasons.csv", index=False)

    # 3 ---- calibration
    bins = [0, .35, .45, .5, .55, .6, .65, .7, 1.0]
    d = pd.DataFrame({"y": cl.home_win.values, "p": cl.p_home.values})
    d["b"] = pd.cut(d.p, bins)
    cal = d.groupby("b", observed=True).agg(n=("y", "size"), pred=("p", "mean"),
                                            obs=("y", "mean")).reset_index()
    cal["band"] = cal.b.astype(str)
    se = np.sqrt(cal.obs * (1 - cal.obs) / cal.n)
    cal["predicted"] = cal.pred.round(4)
    cal["observed"] = cal.obs.round(4)
    cal["moe95"] = (1.96 * se).round(4)
    cal[["band", "n", "predicted", "observed", "moe95"]].to_csv(
        OUT + "calibration.csv", index=False)

    # 4 ---- accuracy by confidence
    conf = np.maximum(cl.p_home, 1 - cl.p_home)
    hit = ((cl.p_home > .5) == (cl.home_win == 1)).values
    cd = pd.DataFrame({"c": conf, "h": hit})
    cb = cd.groupby(pd.cut(cd.c, [.5, .55, .6, .65, .7, 1.0]),
                    observed=True).agg(n=("h", "size"),
                                       acc=("h", "mean")).reset_index()
    cb["band"] = cb.c.astype(str).str.replace("(", "", regex=False) \
                     .str.replace("]", "", regex=False)
    cb["accuracy"] = cb.acc.round(4)
    cb[["band", "n", "accuracy"]].to_csv(OUT + "confidence.csv", index=False)

    # 5 ---- derived markets
    ot_p, ot_a = float(cl.p_ot.mean()), float(cl.went_ot.mean())
    ot_se = (ot_a * (1 - ot_a) / len(cl)) ** 0.5
    rows = [{"market": "Game went to OT/SO", "predicted": round(ot_p, 4),
             "actual": round(ot_a, 4), "brier": "",
             "verdict": ("on target" if abs(ot_p - ot_a) <= 1.96 * ot_se
                         else ("under-predicts" if ot_p < ot_a
                               else "over-predicts"))}]
    for line, col in ((4.5, "p_over_45"), (5.5, "p_over_55"), (6.5, "p_over_65")):
        y = (cl.total_goals > line).astype(int)
        a, b, l = mm(y, cl[col])
        pm, am = float(cl[col].mean()), float(y.mean())
        se = (am * (1 - am) / len(cl)) ** 0.5
        rows.append({"market": f"Over {line} goals",
                     "predicted": round(pm, 4), "actual": round(am, 4),
                     "brier": round(b, 5),
                     "verdict": ("well calibrated" if abs(pm - am) <= 1.96 * se
                                 else "runs low")})
    y15 = ((cl.home_score - cl.away_score) >= 2).astype(int)
    a, b, l = mm(y15, cl.p_home_m15)
    pm, am = float(cl.p_home_m15.mean()), float(y15.mean())
    se = (am * (1 - am) / len(cl)) ** 0.5
    rows.append({"market": "Home wins by 2+ (-1.5)",
                 "predicted": round(pm, 4), "actual": round(am, 4),
                 "brier": round(b, 5),
                 "verdict": ("well calibrated" if abs(pm - am) <= 1.96 * se
                             else "runs low")})
    pd.DataFrame(rows).to_csv(OUT + "markets.csv", index=False)

    # 6 ---- subset skill
    f = feats[["game_id", "h_b2b", "a_b2b", "h_g_shots_seen", "a_g_shots_seen",
               "a_travel_km"]]
    c2 = cl.merge(f, on="game_id", how="left")
    c2["month"] = pd.to_datetime(c2.date).dt.month
    c2["edge"] = (c2.p_home - 0.5).abs()
    base_all = c2.home_win.mean()

    def sub(name, mask, caveat=""):
        g = c2[mask]
        if len(g) < 60:
            return None
        bb = brier_score_loss(g.home_win, np.full(len(g), base_all))
        b = brier_score_loss(g.home_win, g.p_home)
        return {"subset": name, "n": len(g),
                "accuracy": round(float(((g.p_home > .5) ==
                                         (g.home_win == 1)).mean()), 4),
                "brier": round(b, 5),
                "skill_pct": round(100 * (1 - b / bb), 3), "caveat": caveat}

    subs = [sub("All games", c2.index == c2.index),
            sub("Model edge over 8 pts", c2.edge > 0.08, "selected after the fact"),
            sub("Model edge under 4 pts", c2.edge < 0.04, "selected after the fact"),
            sub("Away team on back-to-back", c2.a_b2b == 1, "small sample"),
            sub("Either team on back-to-back", (c2.h_b2b == 1) | (c2.a_b2b == 1)),
            sub("No back-to-back", (c2.h_b2b == 0) & (c2.a_b2b == 0)),
            sub("Unproven goalie in net", (c2.h_g_shots_seen < 800) |
                (c2.a_g_shots_seen < 800), "small sample"),
            sub("Away travel over 1500 km", c2.a_travel_km > 1500),
            sub("October-November", c2.month.isin([10, 11])),
            sub("December-February", c2.month.isin([12, 1, 2])),
            sub("March-April", c2.month.isin([3, 4]))]
    pd.DataFrame([s for s in subs if s]).to_csv(OUT + "subsets.csv", index=False)

    # 7 ---- ablation
    b0, l0, a0 = score(feats[feats.game_type == 2])
    rows = [{"feature_group": "Full model (nothing dropped)", "brier": round(b0, 5),
             "log_loss": round(l0, 5), "delta_log_loss": 0.0,
             "verdict": "reference"}]
    for name, keys in GROUPS.items():
        b, l, a = score(feats[feats.game_type == 2], keys)
        rows.append({"feature_group": name, "brier": round(b, 5),
                     "log_loss": round(l, 5),
                     "delta_log_loss": round(l - l0, 5),
                     "verdict": "adds signal" if l - l0 > 0.0002
                     else ("neutral" if abs(l - l0) <= 0.0002 else "net noise")})
    pd.DataFrame(rows).sort_values("delta_log_loss", ascending=False).to_csv(
        OUT + "ablation.csv", index=False)

    # 8 ---- current team ratings
    import pickle
    st = pickle.load(open("data/current_state.pkl", "rb"))
    gp = reg[reg.season == "20262027"].groupby("home").size().add(
        reg[reg.season == "20262027"].groupby("away").size(),
        fill_value=0)
    rr = []
    for t, s in st["team"].items():
        rr.append({"team": t,
                   "xg_net_5v5": round(s["xgf_5v5"] - s["xga_5v5"], 4),
                   "xgf_5v5": round(s["xgf_5v5"], 4),
                   "xga_5v5": round(s["xga_5v5"], 4),
                   "goals_for": round(s["gf_all"], 4),
                   "goals_against": round(s["ga_all"], 4),
                   "games_this_season": int(gp.get(t, 0))})
    pd.DataFrame(sorted(rr, key=lambda r: -r["xg_net_5v5"])).to_csv(
        OUT + "ratings.csv", index=False)

    # 9 ---- upcoming games
    g = live.copy()
    g["matchup"] = g.away + " @ " + g.home
    out = pd.DataFrame({
        "date": g.date, "matchup": g.matchup, "away": g.away, "home": g.home,
        "home_win_prob": g.p_home.round(4), "away_win_prob": g.p_away.round(4),
        "reg_home": g.p_reg_home.round(4), "reg_tie": g.p_reg_tie.round(4),
        "reg_away": g.p_reg_away.round(4),
        "exp_home_goals": g.lam_h.round(3), "exp_away_goals": g.lam_a.round(3),
        "exp_total": g.exp_total.round(3), "ot_prob": g.p_ot.round(4),
        "over_5_5": g.p_over_55.round(4), "over_6_5": g.p_over_65.round(4),
        "home_minus_1_5": g.p_home_m15.round(4),
        "home_goalie": g.h_goalie.map(lambda v: nm.get(str(int(v)), "TBD")
                                      if pd.notna(v) else "TBD"),
        "away_goalie": g.a_goalie.map(lambda v: nm.get(str(int(v)), "TBD")
                                      if pd.notna(v) else "TBD"),
        "home_goalie_status": g.h_goalie_label,
        "away_goalie_status": g.a_goalie_label,
        "home_rest": g.h_rest.astype(int), "away_rest": g.a_rest.astype(int),
        "home_elo": g.elo_home.round(0), "away_elo": g.elo_away.round(0),
    })
    out.to_csv(OUT + "upcoming.csv", index=False)

    import os
    for f in sorted(os.listdir(OUT)):
        print(f"{f:22} {os.path.getsize(OUT+f)/1024:7.1f} KB  "
              f"{sum(1 for _ in open(OUT+f))-1} rows")


if __name__ == "__main__":
    main()
