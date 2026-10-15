"""How much headroom is left?

Builds deliberately-cheating "oracle" features and measures how good a model
could possibly get with them. If an oracle that knows each team's true
full-season strength still cannot reach the market's log loss, then better
team ratings are not where the remaining gap lives.

Three levels:
  live    -- what the real model knows (pre-game rolling ratings)
  oracle  -- each team's full-season xG rates, computed from the whole
             season including the game being predicted
  oracle+ -- the above, plus the goalie's realised save performance in that
             very game (an absolute upper bound on goalie information)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

SEASONS = ["20222023", "20232024", "20242025", "20252026", "20262027"]


def season_truth(games, tg):
    """Full-season per-game xG rates per team -- hindsight, by design."""
    d = tg.merge(games[["game_id", "season"]], on="game_id")
    g = d.groupby(["season", "team"]).agg(
        t_xgf=("xgf_5v5", "mean"), t_xga=("xga_5v5", "mean"),
        t_gf=("gf_all", "mean"), t_ga=("ga_all", "mean"),
        t_cf=("cf_5v5", "mean"), t_ca=("ca_5v5", "mean")).reset_index()
    return g


def goalie_truth(games, gg):
    """What the starting goalie actually did in this game."""
    s = gg[gg.is_starter == 1].copy()
    s["real_gsax"] = (s.xg_faced - s.goals_allowed) / s.shots_faced.clip(lower=1)
    return s[["game_id", "team", "real_gsax", "shots_faced"]]


def build(df, truth, gtruth, level):
    x = pd.DataFrame(index=df.index)
    for pre, col in (("h", "home"), ("a", "away")):
        if level == "live":
            x[f"{pre}_xgf"] = df[f"{pre}_r_xgf_5v5"]
            x[f"{pre}_xga"] = df[f"{pre}_r_xga_5v5"]
            x[f"{pre}_gf"] = df[f"{pre}_r_gf_all"]
            x[f"{pre}_ga"] = df[f"{pre}_r_ga_all"]
            x[f"{pre}_cf"] = df[f"{pre}_r_cf_5v5"]
            x[f"{pre}_ca"] = df[f"{pre}_r_ca_5v5"]
            x[f"{pre}_gsax"] = df[f"{pre}_g_gsax"].fillna(0) * 100
        else:
            t = df[["season", col]].merge(
                truth, left_on=["season", col], right_on=["season", "team"],
                how="left")
            for s in ("t_xgf", "t_xga", "t_gf", "t_ga", "t_cf", "t_ca"):
                x[f"{pre}_{s}"] = t[s].values
            if level == "oracle+":
                gt = df[["game_id", col]].merge(
                    gtruth, left_on=["game_id", col],
                    right_on=["game_id", "team"], how="left")
                x[f"{pre}_realg"] = gt.real_gsax.fillna(0).values * 100
            else:
                x[f"{pre}_gsax"] = df[f"{pre}_g_gsax"].fillna(0) * 100
        x[f"{pre}_b2b"] = df[f"{pre}_b2b"].fillna(0)
        x[f"{pre}_rest"] = df[f"{pre}_rest"].fillna(3).clip(0, 6)
    return x.astype(float).fillna(0.0)


def run(df, truth, gtruth, level):
    preds = []
    for i, s in enumerate(SEASONS):
        if i == 0:
            continue
        tr, te = df[df.season.isin(SEASONS[:i])], df[df.season == s]
        if len(te) < 30 or len(tr) < 500:
            continue
        Xtr, Xte = build(tr, truth, gtruth, level), build(te, truth, gtruth, level)
        mu, sd = Xtr.values.mean(0), Xtr.values.std(0) + 1e-9
        m = LogisticRegression(C=0.05, max_iter=5000).fit(
            (Xtr.values - mu) / sd, tr.home_win.values)
        p = m.predict_proba((Xte.values - mu) / sd)[:, 1]
        preds.append(pd.DataFrame({"season": s, "y": te.home_win.values, "p": p}))
    a = pd.concat(preds, ignore_index=True)
    return a


if __name__ == "__main__":
    import sqlite3
    con = sqlite3.connect("data/nhl.sqlite3", timeout=60)
    games = pd.read_sql("SELECT * FROM games WHERE game_type IN (2,3)", con)
    df = pd.read_pickle("data/features.pkl")
    df = df[df.game_type == 2].copy()
    tg = pd.read_pickle("data/team_games.pkl")
    gg = pd.read_pickle("data/goalie_games.pkl")
    truth = season_truth(games, tg)
    gtruth = goalie_truth(games, gg)

    print("Upper bounds on what each kind of information is worth.")
    print("Scored on 2025-26 + 2026-27 (n=1414).\n")
    rows = []
    for level, label in (("live", "what the model actually knows"),
                         ("oracle", "knows true full-season team strength"),
                         ("oracle+", "  ... plus the goalie's actual game")):
        a = run(df, truth, gtruth, level)
        cl = a[a.season.isin(["20252026", "20262027"])]
        y, p = cl.y.values, np.clip(cl.p.values, 1e-6, 1 - 1e-6)
        rows.append({"information": label, "n": len(y),
                     "acc": ((p > .5) == (y == 1)).mean(),
                     "brier": brier_score_loss(y, p),
                     "logloss": log_loss(y, p)})
    r = pd.DataFrame(rows)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nFor reference:")
    print(f"  full structural model      logloss 0.67965")
    print(f"  published closing line     logloss ~0.660")
    print(f"  always pick home           logloss 0.69228")
    gap = rows[1]["logloss"] - rows[0]["logloss"]
    print(f"\nPerfect team-strength knowledge is worth {-gap:+.5f} log loss.")
    gap2 = rows[2]["logloss"] - rows[1]["logloss"]
    print(f"Knowing the goalie's actual game on top is worth {-gap2:+.5f}.")
