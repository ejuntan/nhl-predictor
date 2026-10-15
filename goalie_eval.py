"""Confirmed starting goalies: validate, then measure what guessing costs.

The boxscore marks one goalie per team `starter: true`, which is
authoritative. Two separate questions follow:

  1. Did the backtest use the right goalie? Its feature keyed off the goalie
     who faced the most shots, a proxy for the starter.
  2. How often would the live heuristic (recent workhorse, backup on a
     back-to-back) have named the right starter before puck drop -- and what
     does being wrong cost?

Question 2 is the live gap: a real bettor knows the starter, this board
guesses.
"""
import sqlite3

import numpy as np
import pandas as pd

DB = "data/nhl.sqlite3"


def load(con):
    off = pd.read_sql("""
        SELECT game_id, team, is_home, player_id, starter, toi_s, decision
        FROM goalie_official
    """, con)
    games = pd.read_sql("""
        SELECT game_id, date, season, home, away FROM games
        WHERE game_type = 2 AND state IN ('OFF','FINAL')
    """, con)
    off = off.merge(games, on="game_id")
    return off, games


def validate(off):
    """Exactly one marked starter per team per game?"""
    per = off.groupby(["game_id", "team"]).starter.sum()
    print(f"team-games: {len(per)}")
    print(f"  exactly one starter: {(per == 1).sum()}")
    print(f"  none marked:         {(per == 0).sum()}")
    print(f"  more than one:       {(per > 1).sum()}")
    s = off[off.starter == 1]
    print(f"  marked starters with 0 ice time: {(s.toi_s == 0).sum()}")
    print(f"  median starter ice time: {s.toi_s.median()/60:.1f} min")
    ns = off[off.starter == 0]
    print(f"  non-starters who played anyway: {(ns.toi_s > 0).sum()}"
          f" of {len(ns)}")
    return per


def backtest_starter_accuracy(off, gg):
    """Did the shots-faced proxy pick the same goalie as the official flag?"""
    o = off[off.starter == 1][["game_id", "team", "player_id"]].rename(
        columns={"player_id": "official"})
    p = gg[gg.is_starter == 1][["game_id", "team", "goalie"]].rename(
        columns={"goalie": "proxy"})
    m = o.merge(p, on=["game_id", "team"], how="inner")
    m["agree"] = m.official == m.proxy
    return m


def replay_heuristic(off, games):
    """What the live heuristic would have named, for every team-game."""
    # Official starts, in order, per team.
    st = off[off.starter == 1][["game_id", "team", "date", "player_id"]].copy()
    st = st.sort_values(["team", "date", "game_id"]).reset_index(drop=True)

    # Each team's game dates, to detect back-to-backs.
    rows = []
    for team, grp in st.groupby("team", sort=False):
        hist = []            # (date, starter_id)
        for r in grp.itertuples(index=False):
            guess, why = None, "no history"
            if len(hist) >= 3:
                recent = hist[-12:]
                counts = {}
                for _, pid in recent:
                    counts[pid] = counts.get(pid, 0) + 1
                order = sorted(counts, key=lambda k: -counts[k])
                top = order[0]
                last_date, last_pid = hist[-1]
                gap = (pd.Timestamp(r.date) - pd.Timestamp(last_date)).days
                if gap <= 1 and last_pid == top and len(order) > 1:
                    guess, why = order[1], "backup on back-to-back"
                elif gap <= 1 and last_pid == top:
                    guess, why = top, "back-to-back, no clear backup"
                else:
                    guess, why = top, "recent workhorse"
            rows.append({"game_id": r.game_id, "team": team, "date": r.date,
                         "official": r.player_id, "guess": guess, "why": why})
            hist.append((r.date, r.player_id))
    g = pd.DataFrame(rows)
    g = g[g.guess.notna()].copy()
    g["right"] = g.official == g.guess
    return g


if __name__ == "__main__":
    con = sqlite3.connect(DB, timeout=60)
    off, games = load(con)
    gg = pd.read_pickle("data/goalie_games.pkl")

    print("=" * 68)
    print("1. Is the official starter flag trustworthy?")
    print("=" * 68)
    validate(off)

    print("\n" + "=" * 68)
    print("2. Did the backtest's shots-faced proxy pick the right goalie?")
    print("=" * 68)
    m = backtest_starter_accuracy(off, gg)
    print(f"team-games compared: {len(m)}")
    print(f"  proxy agreed with the official starter: {m.agree.mean():.4%}")
    print(f"  disagreed: {(~m.agree).sum()} team-games")

    print("\n" + "=" * 68)
    print("3. Would the live heuristic have named the right starter?")
    print("=" * 68)
    h = replay_heuristic(off, games)
    print(f"team-games with enough history: {len(h)}")
    print(f"  heuristic correct: {h.right.mean():.4%}")
    print("\n  by situation:")
    print(h.groupby("why").right.agg(["size", "mean"]).to_string(
        float_format=lambda v: f"{v:.4f}"))
    print("\n  by season:")
    hs = h.merge(games[["game_id", "season"]].drop_duplicates(), on="game_id")
    print(hs.groupby("season").right.agg(["size", "mean"]).to_string(
        float_format=lambda v: f"{v:.4f}"))
    h.to_pickle("data/goalie_guess.pkl")
    off.to_pickle("data/goalie_official.pkl")
