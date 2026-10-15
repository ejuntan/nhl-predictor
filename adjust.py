"""Opponent-adjusted team ratings.

The rolling ratings in features.py are raw: a team that has played a soft
schedule looks better than it is. This decomposes each team-game into an
attack effect, the opponent's defence effect and a home effect by ridge
regression on team dummies, so ratings are comparable across schedules.

Refit on a cadence (default weekly) using only games already played, with
exponential time weighting. Each refit is causal: a game on date D is rated
by the most recent fit strictly before D.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

STATS = [("xgf_5v5", "xga_5v5"), ("cf_5v5", "ca_5v5"), ("gf_all", "ga_all")]
HALF_LIFE_DAYS = 185.0
REFIT_DAYS = 7
ALPHA = 60.0


def _long(games, tg):
    """One row per team-game with the opponent attached."""
    g = games[["game_id", "date", "season", "home", "away"]]
    d = tg.merge(g, on="game_id")
    d["opp"] = np.where(d.team == d.home, d.away, d.home)
    d["is_home"] = (d.team == d.home).astype(int)
    d["dt"] = pd.to_datetime(d.date)
    return d.sort_values("dt").reset_index(drop=True)


def _fit_one(sub, teams, forcol, agncol, alpha=ALPHA):
    """Ridge fit of for-rate on attack + opponent-defence + home."""
    ti = {t: i for i, t in enumerate(teams)}
    n, k = len(sub), len(teams)
    X = np.zeros((2 * n, 2 * k + 1))
    y = np.zeros(2 * n)
    w = np.zeros(2 * n)
    a = sub.team.map(ti).values
    o = sub.opp.map(ti).values
    # Each team-game contributes twice: what the team created, and what it
    # conceded (which is the opponent's attack against this team's defence).
    X[np.arange(n), a] = 1
    X[np.arange(n), k + o] = 1
    X[np.arange(n), 2 * k] = sub.is_home.values
    y[:n] = sub[forcol].values
    w[:n] = sub.wt.values
    X[np.arange(n, 2 * n), o] = 1
    X[np.arange(n, 2 * n), k + a] = 1
    X[np.arange(n, 2 * n), 2 * k] = 1 - sub.is_home.values
    y[n:] = sub[agncol].values
    w[n:] = sub.wt.values

    m = Ridge(alpha=alpha, fit_intercept=True).fit(X, y, sample_weight=w)
    att = dict(zip(teams, m.coef_[:k]))
    dfn = dict(zip(teams, m.coef_[k:2 * k]))
    return att, dfn, float(m.coef_[2 * k]), float(m.intercept_)


def ratings(games, tg, refit_days=REFIT_DAYS, half_life=HALF_LIFE_DAYS,
            alpha=ALPHA):
    """Causal opponent-adjusted ratings: one row per (game_id, team)."""
    d = _long(games, tg)
    teams = sorted(set(d.team) | set(d.opp))
    dates = sorted(d.dt.unique())
    if not dates:
        return pd.DataFrame()

    # Refit grid: every refit_days, starting once there is enough history.
    start = dates[0] + np.timedelta64(120, "D")
    grid = []
    cur = start
    while cur <= dates[-1]:
        grid.append(cur)
        cur = cur + np.timedelta64(refit_days, "D")

    fits = {}
    for cutoff in grid:
        sub = d[d.dt < cutoff].copy()
        if len(sub) < 400:
            continue
        age = (cutoff - sub.dt).dt.days.values.astype(float)
        sub["wt"] = 0.5 ** (age / half_life)
        f = {}
        for forcol, agncol in STATS:
            att, dfn, home, icpt = _fit_one(sub, teams, forcol, agncol, alpha)
            f[forcol] = (att, dfn, home, icpt)
        fits[cutoff] = f
    if not fits:
        return pd.DataFrame()

    keys = sorted(fits)
    rows = []
    for r in d.itertuples(index=False):
        prior = [k for k in keys if k <= r.dt]
        if not prior:
            continue
        f = fits[prior[-1]]
        row = {"game_id": r.game_id, "team": r.team}
        for forcol, _ in STATS:
            att, dfn, home, icpt = f[forcol]
            row[f"adj_att_{forcol}"] = att.get(r.team, 0.0)
            row[f"adj_def_{forcol}"] = dfn.get(r.team, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)
