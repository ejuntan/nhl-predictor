"""Lineup absences, and whether they add anything the team ratings miss.

Team ratings are rolling averages of how a team has played. They go stale
the moment the lineup changes: a club missing its two best forwards is not
the team its rating describes. The scratch list is the only free record of
who was actually unavailable.

For each team-game this builds the share of the team's usual ice time that is
missing, weighted by how important the absent players normally are, and then
tests whether the win model improves with it.
"""
import sqlite3

import numpy as np
import pandas as pd

DB = "data/nhl.sqlite3"
ROLL = 20          # games of history used to judge a player's usual role


def player_roles(con):
    """Each player's rolling share of his team's ice time and shots."""
    pt = pd.read_sql("""
        SELECT game_id, player_id, team, toi_reg, is_goalie
        FROM player_toi WHERE is_goalie = 0
    """, con)
    g = pd.read_sql("""
        SELECT game_id, date, season FROM games
        WHERE game_type = 2 AND state IN ('OFF','FINAL')
    """, con)
    d = pt.merge(g, on="game_id")
    team_toi = d.groupby(["game_id", "team"]).toi_reg.sum().rename("team_toi")
    d = d.merge(team_toi, on=["game_id", "team"])
    d["share"] = d.toi_reg / d.team_toi.clip(lower=1)
    d = d.sort_values(["date", "game_id"]).reset_index(drop=True)

    # Pre-game rolling role: what fraction of the team's minutes he usually
    # takes, from his own prior games only.
    alpha = 1 - 0.5 ** (1 / ROLL)
    state, rows = {}, []
    for r in d.itertuples(index=False):
        key = (r.player_id, r.team)
        s = state.get(key)
        rows.append({"game_id": r.game_id, "player_id": r.player_id,
                     "team": r.team,
                     "role": s["share"] if s else np.nan,
                     "role_n": s["n"] if s else 0,
                     "toi_role": s["toi"] if s else np.nan})
        v = {"share": r.share, "toi": r.toi_reg}
        state[key] = ({"n": 1, **v} if s is None else
                      {"n": s["n"] + 1,
                       **{k: (1 - alpha) * s[k] + alpha * v[k] for k in v}})
    roles = pd.DataFrame(rows)
    return d, roles


def absence_features(con):
    """Per (game_id, team): how much of the usual lineup is missing."""
    d, roles = player_roles(con)
    scr = pd.read_sql("SELECT game_id, team, player_id FROM scratches", con)
    games = pd.read_sql("""
        SELECT game_id, date, season, home, away FROM games
        WHERE game_type = 2 AND state IN ('OFF','FINAL')
    """, con)

    # A scratched player's role has to come from his last known role, since
    # he has no row in this game.
    rl = roles[roles.role.notna()].merge(
        d[["game_id", "date"]].drop_duplicates(), on="game_id")
    rl = rl.sort_values("date")
    latest = rl.groupby(["player_id", "team"]).tail(1)[
        ["player_id", "team", "role", "toi_role"]]

    # Roles as of each date, for the scratched players.
    s = scr.merge(games[["game_id", "date"]], on="game_id", how="left")
    s = s.merge(latest, on=["player_id", "team"], how="left")
    s["role"] = s.role.fillna(0.0)
    s["toi_role"] = s.toi_role.fillna(0.0)

    agg = s.groupby(["game_id", "team"]).agg(
        n_scratched=("player_id", "size"),
        role_out=("role", "sum"),
        toi_out=("toi_role", "sum"),
        top_role_out=("role", "max")).reset_index()
    return agg


def attach(df, agg):
    out = df.copy()
    for pre, col in (("h", "home"), ("a", "away")):
        a = agg.rename(columns={
            "team": col, "n_scratched": f"{pre}_n_out",
            "role_out": f"{pre}_role_out", "toi_out": f"{pre}_toi_out",
            "top_role_out": f"{pre}_top_out"})
        out = out.merge(a, on=["game_id", col], how="left")
        for c in ("n_out", "role_out", "toi_out", "top_out"):
            out[f"{pre}_{c}"] = out[f"{pre}_{c}"].fillna(0.0)
    out["role_out_gap"] = out.a_role_out - out.h_role_out
    out["top_out_gap"] = out.a_top_out - out.h_top_out
    return out


if __name__ == "__main__":
    con = sqlite3.connect(DB, timeout=60)
    agg = absence_features(con)
    agg.to_pickle("data/absences.pkl")
    print(f"team-games with scratch data: {len(agg)}")
    print(f"  scratches per team-game: mean {agg.n_scratched.mean():.2f}  "
          f"max {agg.n_scratched.max()}")
    print(f"  usual ice-time share missing: mean {agg.role_out.mean():.4f}  "
          f"p90 {agg.role_out.quantile(.9):.4f}  max {agg.role_out.max():.4f}")
    print(f"  biggest single absence (role share): mean "
          f"{agg.top_role_out.mean():.4f}  max {agg.top_role_out.max():.4f}")
    print("\n  distribution of total role missing:")
    print(agg.role_out.describe(percentiles=[.5, .75, .9, .99]).round(4).to_string())
