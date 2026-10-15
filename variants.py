"""Build every feature-frame variant on one identical set of games.

Step 1 of the validation plan needs the old and new models scored on exactly
the same games. The frames here are therefore all built from one `games`
table in one pass, so a difference between two rows of the comparison table
can only come from the change being tested.

  raw     counting stats per game   -- the original (v1) team ratings
  per60   rates per 60 minutes of that strength state, from shift charts (v2)
  per60i  the same, with the denominator imputed where the NHL published no
          shift chart, so no team-game is dropped
  per60c  the same as per60, but a game with no shift chart keeps the team's
          carried-forward rating instead of a zero-filled one
  goalie  per60 + confirmed-starter encodings
  lineup  per60 + scratch-based absence features
  adj     per60 + opponent-adjusted attack/defence ratings
"""
import pickle
import sqlite3

import pandas as pd

import adjust
import features as F
import goalie_strength as GS
import lineups as L
from xg import apply_xg

DB = "data/nhl.sqlite3"
CACHE = "data/variant_frames.pkl"


def inputs():
    con = sqlite3.connect(DB, timeout=60)
    games = pd.read_sql(
        "SELECT * FROM games WHERE game_type IN (2,3) ORDER BY date", con)
    shots = pd.read_sql("SELECT * FROM shots", con)
    shots["xg"] = apply_xg(pickle.load(open("data/xg_model.pkl", "rb")), shots)
    played = games[games.state.isin(["OFF", "FINAL"])].copy()
    tg_raw = F.team_game_stats(played, shots)
    gg = F.goalie_game_stats(played, shots)
    team_toi = pd.read_sql("SELECT * FROM team_toi", con)
    tg60 = F.to_per_60(tg_raw, team_toi, impute=False)
    tg60i = F.to_per_60(tg_raw, team_toi, impute=True)
    return con, games, played, tg_raw, tg60, tg60i, gg


def build():
    con, games, played, tg_raw, tg60, tg60i, gg = inputs()
    sf = F.schedule_features(games)
    rg, _ = F.roll_goalie(gg)

    frames = {}
    for name, tg, carry in (("raw", tg_raw, False), ("per60", tg60, False),
                            ("per60i", tg60i, False),
                            ("per60c", tg60, True)):
        rt, _, _ = F.roll_team(played, tg, carry=carry)
        frames[name] = F.assemble(played, tg, gg, rt, rg, sf)
        n_missing = frames[name].h_r_xgf_5v5.isna().sum() + \
            frames[name].a_r_xgf_5v5.isna().sum()
        print(f"{name:6} rows {len(frames[name])}  "
              f"sides with no rating {n_missing}", flush=True)

    base = frames["per60"]
    frames["goalie"] = GS.attach(base, GS.goalie_extras())
    frames["lineup"] = L.attach(base, L.absence_features(con))
    adj = adjust.ratings(played, tg60)
    f = base.copy()
    for pre, col in (("h", "home"), ("a", "away")):
        a = adj.rename(columns={"team": col, **{
            c: f"{pre}_{c}" for c in adj.columns
            if c not in ("game_id", "team")}})
        f = f.merge(a, on=["game_id", col], how="left")
    frames["adj"] = f
    for k in ("goalie", "lineup", "adj"):
        print(f"{k:6} rows {len(frames[k])}  cols {len(frames[k].columns)}",
              flush=True)

    pickle.dump(frames, open(CACHE, "wb"))
    return frames


def load():
    return pickle.load(open(CACHE, "rb"))


if __name__ == "__main__":
    build()
