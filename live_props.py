"""Shots-on-goal prop prices for upcoming games.

Lineups are the limiting assumption: a skater is included if he played in
his team's most recent game, which misses same-day scratches and call-ups.
Expected ice time is his own rolling average, so a change in role moves the
price more than anything the model can see in advance.
"""
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.linear_model import PoissonRegressor

import names as N
import props_model as P

DB = "data/nhl.sqlite3"


def main(days=8):
    con = sqlite3.connect(DB, timeout=90)
    pg = pd.read_pickle("data/player_games.pkl")

    # Fit on everything available.
    F, off = P.feature_frame(pg)
    mu_, sd_ = F.values.mean(0), F.values.std(0) + 1e-9
    Z = (F.values - mu_) / sd_
    w = np.exp(off)
    m = PoissonRegressor(alpha=P.ALPHA, max_iter=8000).fit(
        Z, pg.sog.values / w, sample_weight=w)
    size = P.fit_nb_dispersion(pg.sog.values, m.predict(Z) * w)
    print(f"fitted on {len(pg)} skater-games, NB size={size:.1f}")

    # Each player's latest rolling state, and his last game's date.
    last = pg.sort_values("date").groupby("player_id").tail(1)
    ecols = [c for c in pg.columns if c.startswith("e_")]
    state = last.set_index("player_id")[ecols + ["team", "date"]]

    # Who is in the lineup: whoever played his team's most recent game.
    games = pd.read_sql("""SELECT game_id, date, season, home, away, state
                           FROM games WHERE game_type=2""", con)
    played = games[games.state.isin(["OFF", "FINAL"])]
    asof = played.date.max()
    lastgame = {}
    for t in set(played.home) | set(played.away):
        sub = played[(played.home == t) | (played.away == t)]
        lastgame[t] = sub.date.max()
    roster = {}
    for t, d in lastgame.items():
        gid = played[((played.home == t) | (played.away == t))
                     & (played.date == d)].game_id.iloc[0]
        ids = pg[(pg.game_id == gid) & (pg.team == t)].player_id.tolist()
        roster[t] = [i for i in ids if i in state.index]

    up = games[(games.state == "FUT") & (games.date > asof)
               & (games.date <= (pd.Timestamp(asof)
                                 + pd.Timedelta(days=days)).strftime("%Y-%m-%d"))]
    print(f"upcoming games: {len(up)}  (as of {asof})")

    # Opponent shot-suppression, current values.
    tt = pd.read_sql("SELECT game_id, team, toi_reg FROM team_toi", con)
    sh = pd.read_sql("""SELECT game_id, is_home,
            SUM(CASE WHEN event IN ('shot-on-goal','goal') THEN 1 ELSE 0 END) sog,
            SUM(CASE WHEN event='blocked-shot' THEN 1 ELSE 0 END) blk
            FROM shots WHERE period_type='REG' GROUP BY game_id, is_home""", con)
    g2 = pd.read_sql("SELECT game_id, home, away, date FROM games", con)
    sh = sh.merge(g2, on="game_id")
    sh["team"] = np.where(sh.is_home == 1, sh.home, sh.away)
    sh["opp"] = np.where(sh.is_home == 1, sh.away, sh.home)
    conc = sh[["game_id", "opp", "sog", "date"]].rename(
        columns={"opp": "team", "sog": "sog_against"})
    blk = sh[["game_id", "team", "blk"]]
    tc = tt.merge(conc, on=["game_id", "team"]).merge(blk, on=["game_id", "team"])
    tc["sa60"] = tc.sog_against / (tc.toi_reg / 3600)
    tc["blk60"] = tc.blk / (tc.toi_reg / 3600)
    tc = tc.sort_values("date")
    alpha = 1 - 0.5 ** (1 / 20.0)
    cur = {}
    for r in tc.itertuples(index=False):
        s = cur.get(r.team)
        cur[r.team] = ({"sa60": r.sa60, "blk60": r.blk60} if s is None else
                       {"sa60": (1 - alpha) * s["sa60"] + alpha * r.sa60,
                        "blk60": (1 - alpha) * s["blk60"] + alpha * r.blk60})

    rows = []
    for g in up.itertuples(index=False):
        for team, opp, is_home in ((g.home, g.away, 1), (g.away, g.home, 0)):
            oc = cur.get(opp)
            if oc is None:
                continue
            for pid in roster.get(team, []):
                st = state.loc[pid]
                r = {"game_id": g.game_id, "date": g.date, "player_id": pid,
                     "team": team, "opp": opp, "is_home": is_home,
                     "opp_sa60": oc["sa60"], "opp_blk60": oc["blk60"]}
                for c in ecols:
                    r[c] = st[c]
                rows.append(r)
    d = pd.DataFrame(rows)
    if d.empty:
        print("no upcoming skaters")
        return
    d = d.dropna(subset=["e_sog60", "e_toi_reg"])

    Fp, offp = P.feature_frame(d)
    mu = m.predict((Fp.values - mu_) / sd_) * np.exp(offp)
    d["exp_sog"] = mu
    d["exp_toi_min"] = d.e_toi_reg / 60
    for L in P.LINES:
        k = int(np.floor(L))
        d[f"p_over_{L}"] = 1 - nbinom.cdf(k, size, size / (size + mu))
    nm = N.resolve(d.player_id.unique())
    d["player"] = d.player_id.map(lambda v: nm.get(str(int(v)), f"#{v}"))
    d["nb_size"] = size
    d = d.sort_values(["date", "game_id", "exp_sog"], ascending=[True, True, False])
    d.to_pickle("data/live_props.pkl")
    print(f"priced {len(d)} skater-games across {d.game_id.nunique()} games")
    print(d[["date", "player", "team", "opp", "exp_toi_min", "exp_sog",
             "p_over_1.5", "p_over_2.5"]].head(12).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    return d


if __name__ == "__main__":
    main()
