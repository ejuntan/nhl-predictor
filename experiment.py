"""Try model variants on validation seasons, keeping the test set untouched.

Development decisions are made on 2023-24 and 2024-25. The 2025-26 / 2026-27
holdout is scored once, at the end, for whatever variant wins here -- so the
reported number is not the result of picking the best of many tries on the
same games.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

import model as M

SEASONS = ["20222023", "20232024", "20242025", "20252026", "20262027"]
VALID = ["20232024", "20242025"]
TEST = ["20252026", "20262027"]

NOISE = ["_pp", "_pk", "_pp_conv", "_pk_conv", "_finish", "_suppress"]
ADJ = ["adj_"]


def matrix_for(variant):
    """Return a build_matrix replacement implementing one variant."""
    orig = M.build_matrix

    def f(df):
        x = orig(df)
        if variant in ("adj", "adj_lean", "adj_only"):
            for pre in ("h", "a"):
                for s in ("xgf_5v5", "cf_5v5", "gf_all"):
                    ac = f"{pre}_adj_att_{s}"
                    dc = f"{pre}_adj_def_{s}"
                    if ac in df.columns:
                        x[f"{pre}_aatt_{s}"] = df[ac].fillna(0.0)
                        x[f"{pre}_adef_{s}"] = df[dc].fillna(0.0)
            if all(c in df.columns for c in
                   ("h_adj_att_xgf_5v5", "a_adj_def_xgf_5v5")):
                x["adj_h_off_vs_a_def"] = (df.h_adj_att_xgf_5v5.fillna(0)
                                           - df.a_adj_def_xgf_5v5.fillna(0))
                x["adj_a_off_vs_h_def"] = (df.a_adj_att_xgf_5v5.fillna(0)
                                           - df.h_adj_def_xgf_5v5.fillna(0))
        if variant in ("lean", "adj_lean"):
            x = x[[c for c in x.columns if not any(k in c for k in NOISE)]]
        if variant == "adj_only":
            # keep only adjusted strength + goalie + schedule, drop raw levels
            drop = ["_xg_net5", "_xgf5", "_xga5", "_xgf_all", "_xga_all",
                    "_c_net5", "_gf_all", "_ga_all", "h_off_vs_a_def",
                    "a_off_vs_h_def", "xg_gap"] + NOISE
            x = x[[c for c in x.columns
                   if not any(k in c for k in drop) or c.startswith(("h_aatt", "a_aatt", "h_adef", "a_adef", "adj_"))]]
        return x.astype(float).fillna(0.0)
    return f, orig


def evaluate(df, variant, seasons, label):
    f, orig = matrix_for(variant)
    M.build_matrix = f
    try:
        preds = []
        for i, s in enumerate(SEASONS):
            if s not in seasons or i == 0:
                continue
            tr = df[df.season.isin(SEASONS[:i])]
            te = df[df.season == s]
            if len(te) < 30 or len(tr) < 500:
                continue
            m = M.NHLModel().fit(tr)
            # causal scoring-level multiplier, same as the backtest
            W = 250
            tail = tr.sort_values(["date", "game_id"]).tail(W)
            hist = pd.concat([tail, te.sort_values(["date", "game_id"])])
            lh, la = m.rates(hist)
            pt = lh + la
            at = (hist.h_reg + hist.a_reg).values
            mult, bp, ba = [], list(pt[:len(tail)]), list(at[:len(tail)])
            for j in range(len(tail), len(hist)):
                mult.append(float(np.clip(np.mean(ba[-W:]) / np.mean(bp[-W:]),
                                          0.90, 1.10)))
                bp.append(pt[j]); ba.append(at[j])
            te = te.sort_values(["date", "game_id"])
            pr = m.predict(te, scoring_mult=mult)
            preds.append(pd.DataFrame({"y": te.home_win.values,
                                       "p": pr.p_home.values,
                                       "season": s}))
        p = pd.concat(preds, ignore_index=True)
        y, pp = p.y.values, np.clip(p.p.values, 1e-6, 1 - 1e-6)
        return {"variant": label, "n": len(y),
                "acc": ((pp > .5) == (y == 1)).mean(),
                "brier": brier_score_loss(y, pp),
                "logloss": log_loss(y, pp)}
    finally:
        M.build_matrix = orig


if __name__ == "__main__":
    df = pd.read_pickle("data/features.pkl")
    df = df[df.game_type == 2].copy()
    adj = pd.read_pickle("data/adj_ratings.pkl")
    for pre, col in (("h", "home"), ("a", "away")):
        a = adj.rename(columns={c: f"{pre}_{c}" for c in adj.columns
                                if c != "game_id"})
        a = a.rename(columns={f"{pre}_team": "team"})
        df = df.merge(a, left_on=["game_id", col], right_on=["game_id", "team"],
                      how="left").drop(columns=["team"])
    print(f"features {df.shape}, adjusted cols merged\n")

    print("=" * 66)
    print("VALIDATION (2023-24 + 2024-25) -- decisions made here")
    print("=" * 66)
    rows = []
    for v, lab in (("base", "current model"),
                   ("adj", "+ opponent-adjusted ratings"),
                   ("lean", "- special teams & finishing residual"),
                   ("adj_lean", "+ adjusted, - noise groups"),
                   ("adj_only", "adjusted ratings replace raw levels")):
        rows.append(evaluate(df, v, VALID, lab))
        print(pd.DataFrame(rows[-1:]).to_string(
            index=False, header=(len(rows) == 1),
            float_format=lambda x: f"{x:.5f}"), flush=True)
    r = pd.DataFrame(rows)
    base = r[r.variant == "current model"].iloc[0]
    r["d_logloss"] = r.logloss - base.logloss
    r["d_brier"] = r.brier - base.brier
    print("\nversus current model (negative = better):")
    print(r[["variant", "n", "acc", "brier", "d_brier", "logloss",
             "d_logloss"]].to_string(index=False,
                                     float_format=lambda x: f"{x:+.5f}"))
    df.to_pickle("data/features_adj.pkl")
