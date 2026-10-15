"""Run the full feature pipeline and cache the result."""
import pickle
import sqlite3

import pandas as pd

import features as F
from xg import apply_xg

DB = "data/nhl.sqlite3"


def main():
    con = sqlite3.connect(DB, timeout=60)
    games = pd.read_sql(
        "SELECT * FROM games WHERE game_type IN (2,3) ORDER BY date", con)
    shots = pd.read_sql("SELECT * FROM shots", con)
    xg = pickle.load(open("data/xg_model.pkl", "rb"))
    shots["xg"] = apply_xg(xg, shots)
    print(f"games {len(games)}  shots {len(shots)}")

    played = games[games.state.isin(["OFF", "FINAL"])].copy()
    print("played", len(played))

    tg_raw = F.team_game_stats(played, shots)
    gg = F.goalie_game_stats(played, shots)
    team_toi = pd.read_sql("SELECT * FROM team_toi", con)
    tg = F.to_per_60(tg_raw, team_toi)
    print(f"team-games {len(tg_raw)} -> {len(tg)} with ice time  "
          f"goalie-games {len(gg)}")
    print(f"  5v5 xGF/60 {tg.xgf_5v5.mean():.3f}  "
          f"PP xGF/60 {tg.xgf_pp.mean():.3f}")

    rt, team_state, league = F.roll_team(played, tg)
    rg, goalie_state = F.roll_goalie(gg)
    sf = F.schedule_features(games)        # all games, incl. future
    print("rolling done")

    df = F.assemble(played, tg, gg, rt, rg, sf)
    df.to_pickle("data/features.pkl")
    tg_raw.to_pickle("data/team_games_raw.pkl")
    for obj, name in ((tg, "team_games"), (gg, "goalie_games"),
                      (rt, "roll_team"), (rg, "roll_goalie"), (sf, "sched")):
        obj.to_pickle(f"data/{name}.pkl")
    import pickle as pk
    pk.dump({"team": team_state, "goalie": goalie_state, "league": league,
             "asof": played.date.max()}, open("data/current_state.pkl", "wb"))
    print("feature rows", len(df), "cols", len(df.columns))
    return df


if __name__ == "__main__":
    main()
