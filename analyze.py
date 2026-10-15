"""Feature ablation and subset-skill analysis on the clean holdout."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

import model as M

TR = ["20222023", "20232024", "20242025"]
TE = ["20252026", "20262027"]

GROUPS = {
    "goalie (gsax)": ["_gsax", "gsax_gap"],
    "schedule (rest/b2b/travel)": ["_rest", "_b2b", "_g4", "_travel",
                                   "tz_abs", "rest_gap", "b2b_gap"],
    "special teams": ["_pp", "_pk", "_pp_conv", "_pk_conv"],
    "xG (5v5 + all sit.)": ["_xg_net5", "_xgf5", "_xga5", "_xgf_all",
                            "_xga_all", "h_off_vs_a_def", "a_off_vs_h_def",
                            "xg_gap"],
    "actual goals": ["_gf_all", "_ga_all"],
    "finishing/goaltending residual": ["_finish", "_suppress"],
    "corsi": ["_c_net5"],
}


def score(df, drop=()):
    orig = M.build_matrix

    def patched(d):
        x = orig(d)
        if drop:
            keep = [c for c in x.columns
                    if not any(k in c for k in drop)]
            x = x[keep]
        return x
    M.build_matrix = patched
    try:
        tr, te = df[df.season.isin(TR)], df[df.season.isin(TE)]
        m = M.NHLModel().fit(tr)
        p = m.predict(te).p_home.values
        y = te.home_win.values
        return (brier_score_loss(y, p), log_loss(y, p),
                ((p > .5) == (y == 1)).mean())
    finally:
        M.build_matrix = orig


if __name__ == "__main__":
    df = pd.read_pickle("data/features.pkl")
    df = df[df.game_type == 2]
    b0, l0, a0 = score(df)
    print(f"full model            brier={b0:.5f}  logloss={l0:.5f}  acc={a0:.4f}")
    print("\nleave-one-group-out (positive delta = the group was helping):")
    rows = []
    for name, keys in GROUPS.items():
        b, l, a = score(df, keys)
        rows.append({"dropped": name, "brier": b, "d_brier": b - b0,
                     "logloss": l, "d_logloss": l - l0, "acc": a})
    r = pd.DataFrame(rows).sort_values("d_logloss", ascending=False)
    print(r.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    print("\n" + "=" * 70)
    print("SUBSET SKILL on the clean holdout (where might an edge hide?)")
    print("=" * 70)
    p = pd.read_pickle("data/backtest_preds.pkl")
    cl = p[p.season.isin(TE)].copy()
    f = df[["game_id", "h_b2b", "a_b2b", "h_rest", "a_rest",
            "h_travel_km", "a_travel_km", "h_g_shots_seen",
            "a_g_shots_seen"]]
    cl = cl.merge(f, on="game_id", how="left")
    cl["month"] = pd.to_datetime(cl.date).dt.month
    cl["edge"] = (cl.p_home - 0.5).abs()

    def sub(name, mask):
        g = cl[mask]
        if len(g) < 60:
            return None
        base = np.full(len(g), cl.home_win.mean())
        return {"subset": name, "n": len(g),
                "acc": ((g.p_home > .5) == (g.home_win == 1)).mean(),
                "brier": brier_score_loss(g.home_win, g.p_home),
                "base_brier": brier_score_loss(g.home_win, base),
                "skill_%": 100 * (1 - brier_score_loss(g.home_win, g.p_home)
                                  / brier_score_loss(g.home_win, base))}

    subs = [
        sub("all", cl.index == cl.index),
        sub("either team on b2b", (cl.h_b2b == 1) | (cl.a_b2b == 1)),
        sub("no b2b", (cl.h_b2b == 0) & (cl.a_b2b == 0)),
        sub("away team on b2b", cl.a_b2b == 1),
        sub("away travel >1500km", cl.a_travel_km > 1500),
        sub("Oct-Nov (early)", cl.month.isin([10, 11])),
        sub("Dec-Feb (mid)", cl.month.isin([12, 1, 2])),
        sub("Mar-Apr (late)", cl.month.isin([3, 4])),
        sub("model edge >8pts", cl.edge > 0.08),
        sub("model edge <4pts", cl.edge < 0.04),
        sub("unproven goalie (<800 shots seen)",
            (cl.h_g_shots_seen < 800) | (cl.a_g_shots_seen < 800)),
    ]
    print(pd.DataFrame([s for s in subs if s]).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
