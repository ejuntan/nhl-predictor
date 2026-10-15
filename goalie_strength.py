"""Should the goalie matter more than it currently does?

Swapping the confirmed starter for a guess changed the model's log loss by
0.00004, which means the goalie feature is too weak for the identity of the
netminder to register. The feature is goals-saved-above-expected per shot,
shrunk by n/(n+1500) -- statistically defensible, since the statistic is
noisy, but possibly over-shrunk for this purpose.

This tests richer goalie representations against the current one, on the
validation seasons.
"""
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

import features as F
import model as M

SEASONS = ["20222023", "20232024", "20242025", "20252026", "20262027"]
VALID = ["20232024", "20242025"]
TEST = ["20252026", "20262027"]


def goalie_extras():
    """Per (game_id, team): start share, and lighter-shrunk ratings."""
    gg = pd.read_pickle("data/goalie_games.pkl")
    off = pd.read_pickle("data/goalie_official.pkl")
    starters = off[off.starter == 1][["game_id", "team", "player_id", "date"]]
    starters = starters.rename(columns={"player_id": "goalie"})
    starters = starters.sort_values(["team", "date", "game_id"])

    # Rolling share of the team's recent starts held by this goalie.
    rows = []
    for team, grp in starters.groupby("team", sort=False):
        hist = []
        for r in grp.itertuples(index=False):
            recent = hist[-20:]
            share = (recent.count(r.goalie) / len(recent)) if recent else np.nan
            rows.append({"game_id": r.game_id, "team": team,
                         "goalie": r.goalie, "start_share": share})
            hist.append(r.goalie)
    share = pd.DataFrame(rows)

    # Lighter-shrunk GSAx, recomputed with a smaller constant.
    g = gg.sort_values(["date", "game_id"]).reset_index(drop=True)
    state, out = {}, []
    for r in g.itertuples(index=False):
        st = state.setdefault(r.goalie, {"n": 0.0, "xg": 0.0, "ga": 0.0})
        n, xg, ga = st["n"], st["xg"], st["ga"]
        raw = (xg - ga) / n if n > 0 else 0.0
        out.append({"game_id": r.game_id, "goalie": r.goalie,
                    "gsax_k300": raw * (n / (n + 300.0)),
                    "gsax_k3000": raw * (n / (n + 3000.0))})
        dec = 0.5 ** (r.shots_faced / F.GOALIE_SHOT_HALFLIFE)
        st["n"] = n * dec + r.shots_faced
        st["xg"] = xg * dec + r.xg_faced
        st["ga"] = ga * dec + r.goals_allowed
    light = pd.DataFrame(out)
    return share.merge(light, on=["game_id", "goalie"], how="left")


def attach(df, extras):
    out = df.copy()
    for pre, col in (("h", "home"), ("a", "away")):
        e = extras.rename(columns={
            "team": col, "goalie": f"{pre}_gid",
            "start_share": f"{pre}_share",
            "gsax_k300": f"{pre}_k300", "gsax_k3000": f"{pre}_k3000"})
        out = out.merge(e, on=["game_id", col], how="left")
        out[f"{pre}_share"] = out[f"{pre}_share"].fillna(0.5)
        out[f"{pre}_k300"] = out[f"{pre}_k300"].fillna(0.0)
        out[f"{pre}_k3000"] = out[f"{pre}_k3000"].fillna(0.0)
        out[f"{pre}_backup"] = (out[f"{pre}_share"] < 0.35).astype(float)
    return out


VARIANTS = {
    "current (gsax, K=1500)": [],
    "+ starter/backup indicator": ["backup"],
    "+ start share": ["share"],
    "lighter shrinkage (K=300)": ["k300"],
    "heavier shrinkage (K=3000)": ["k3000"],
    "all of the above": ["backup", "share", "k300"],
}


def patched_matrix(extra):
    orig = M.build_matrix

    def f(d):
        x = orig(d)
        for pre in ("h", "a"):
            for k in extra:
                col = f"{pre}_{k}"
                if col in d.columns:
                    x[f"{pre}_{k}"] = d[col].astype(float)
        if "k300" in extra:
            x["k300_gap"] = (d.h_k300.fillna(0) - d.a_k300.fillna(0)) * 100
        if "backup" in extra:
            x["backup_gap"] = d.h_backup.fillna(0) - d.a_backup.fillna(0)
        if "share" in extra:
            x["share_gap"] = d.h_share.fillna(.5) - d.a_share.fillna(.5)
        return x.astype(float).fillna(0.0)
    return f, orig


def evaluate(df, extra, seasons):
    f, orig = patched_matrix(extra)
    M.build_matrix = f
    try:
        preds = []
        for i, s in enumerate(SEASONS):
            if s not in seasons or i == 0:
                continue
            tr, te = df[df.season.isin(SEASONS[:i])], df[df.season == s]
            if len(te) < 30 or len(tr) < 500:
                continue
            m = M.NHLModel().fit(tr)
            pr = m.predict(te)
            preds.append(pd.DataFrame({"y": te.home_win.values,
                                       "p": pr.p_home.values}))
        a = pd.concat(preds, ignore_index=True)
        y, p = a.y.values, np.clip(a.p.values, 1e-6, 1 - 1e-6)
        return {"n": len(y), "acc": round(float(((p > .5) == (y == 1)).mean()), 4),
                "brier": round(brier_score_loss(y, p), 5),
                "logloss": round(log_loss(y, p), 5)}
    finally:
        M.build_matrix = orig


if __name__ == "__main__":
    df = pd.read_pickle("data/features.pkl")
    df = df[df.game_type == 2].copy()
    extras = goalie_extras()
    df = attach(df, extras)
    print(f"attached goalie extras; backup rate "
          f"{df.h_backup.mean():.3f} home / {df.a_backup.mean():.3f} away")
    print(f"start-share spread: {df.h_share.describe()[['mean','std','min','max']].round(3).to_dict()}")
    df.to_pickle("data/features_goalie.pkl")

    rows = []
    for lab, extra in VARIANTS.items():
        r = evaluate(df, extra, VALID)
        r["variant"] = lab
        rows.append(r)
        print(pd.DataFrame([r])[["variant", "n", "acc", "brier", "logloss"]]
              .to_string(index=False, header=(len(rows) == 1)), flush=True)
    t = pd.DataFrame(rows)
    base = t[t.variant == "current (gsax, K=1500)"].iloc[0]
    t["d_logloss"] = (t.logloss - base.logloss).round(5)
    print("\nVALIDATION 2023-25, vs the current goalie feature "
          "(negative = better):")
    print(t[["variant", "acc", "brier", "logloss", "d_logloss"]].to_string(index=False))
