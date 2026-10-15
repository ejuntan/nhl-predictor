"""Goalie saves model.

A shots-on-goal prop averages 1.5, so its coefficient of variation is about
0.8 and most of the number is noise no model can reach. Saves average near 28.
For a count with Poisson-like spread the relative noise falls as 1/sqrt(mean),
so saves should be several times more predictable than skater shots -- and the
predictable part is exactly the part the earlier work showed is forecastable:
team shot volume.

Saves = shots on goal faced - goals allowed. Both come from the play-by-play;
goalie ice time comes from the shift charts.
"""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import nbinom
from sklearn.linear_model import PoissonRegressor

SEASONS = ["20222023", "20232024", "20242025", "20252026", "20262027"]
TEST = ["20252026", "20262027"]
HL = 12.0
MIN_PRIOR = 8
ALPHA = 0.001
LINES = (24.5, 27.5, 30.5)


def goalie_games(con):
    """One row per goalie-game: saves, shots faced, ice time, context."""
    sh = pd.read_sql("""
        SELECT game_id, goalie_id,
               SUM(CASE WHEN event = 'shot-on-goal' THEN 1 ELSE 0 END) AS saves,
               SUM(CASE WHEN event IN ('shot-on-goal','goal') THEN 1 ELSE 0 END) AS sog_faced,
               SUM(CASE WHEN event = 'goal' THEN 1 ELSE 0 END) AS goals,
               SUM(xg) AS xg_faced
        FROM (SELECT s.*, x.xg FROM shots s
              JOIN shot_xg x ON 1=0) WHERE 1=0 GROUP BY game_id, goalie_id
    """, con) if False else None

    sh = pd.read_sql("""
        SELECT game_id, goalie_id,
               SUM(CASE WHEN event = 'shot-on-goal' THEN 1 ELSE 0 END) AS saves,
               SUM(CASE WHEN event IN ('shot-on-goal','goal') THEN 1 ELSE 0 END) AS sog_faced,
               SUM(CASE WHEN event = 'goal' THEN 1 ELSE 0 END) AS goals
        FROM shots
        WHERE goalie_id IS NOT NULL AND period_type = 'REG'
        GROUP BY game_id, goalie_id
    """, con)
    toi = pd.read_sql("""
        SELECT game_id, player_id AS goalie_id, team, toi_reg
        FROM player_toi WHERE is_goalie = 1
    """, con)
    g = pd.read_sql("""
        SELECT game_id, date, season, home, away FROM games
        WHERE game_type = 2 AND state IN ('OFF','FINAL')
    """, con)

    d = toi.merge(g, on="game_id").merge(sh, on=["game_id", "goalie_id"], how="left")
    for c in ("saves", "sog_faced", "goals"):
        d[c] = d[c].fillna(0).astype(int)
    d["is_home"] = (d.team == d.home).astype(int)
    d["opp"] = np.where(d.is_home == 1, d.away, d.home)
    # Starters who played something close to a full game.
    d = d[(d.toi_reg >= 1800) & (d.sog_faced >= 5)]
    return d.sort_values(["date", "game_id"]).reset_index(drop=True)


def team_rates(con):
    """Each team's shots-for and shots-against per 60, pre-game EWMA."""
    tt = pd.read_sql("SELECT game_id, team, toi_reg FROM team_toi", con)
    sh = pd.read_sql("""
        SELECT game_id, is_home,
               SUM(CASE WHEN event IN ('shot-on-goal','goal') THEN 1 ELSE 0 END) AS sog
        FROM shots WHERE period_type='REG' GROUP BY game_id, is_home
    """, con)
    g = pd.read_sql("SELECT game_id, date, home, away FROM games", con)
    sh = sh.merge(g, on="game_id")
    sh["team"] = np.where(sh.is_home == 1, sh.home, sh.away)
    sh["opp"] = np.where(sh.is_home == 1, sh.away, sh.home)
    f = sh[["game_id", "team", "sog", "date"]].rename(columns={"sog": "sf"})
    a = sh[["game_id", "opp", "sog"]].rename(columns={"opp": "team", "sog": "sa"})
    t = tt.merge(f, on=["game_id", "team"]).merge(a, on=["game_id", "team"])
    t["sf60"] = t.sf / (t.toi_reg / 3600)
    t["sa60"] = t.sa / (t.toi_reg / 3600)
    t = t.sort_values("date")

    alpha = 1 - 0.5 ** (1 / 20.0)
    state, rows = {}, []
    for r in t.itertuples(index=False):
        s = state.get(r.team)
        rows.append({"game_id": r.game_id, "team": r.team,
                     "t_sf60": s["sf60"] if s else np.nan,
                     "t_sa60": s["sa60"] if s else np.nan})
        state[r.team] = ({"sf60": r.sf60, "sa60": r.sa60} if s is None else
                         {"sf60": (1 - alpha) * s["sf60"] + alpha * r.sf60,
                          "sa60": (1 - alpha) * s["sa60"] + alpha * r.sa60})
    return pd.DataFrame(rows)


def build(con):
    d = goalie_games(con)
    tr = team_rates(con)
    # The goalie's own team suppresses shots; the opponent generates them.
    own = tr.rename(columns={"t_sf60": "own_sf60", "t_sa60": "own_sa60"})
    opp = tr.rename(columns={"team": "opp", "t_sf60": "opp_sf60",
                             "t_sa60": "opp_sa60"})
    d = d.merge(own, on=["game_id", "team"], how="left") \
         .merge(opp, on=["game_id", "opp"], how="left")

    d["saves60"] = d.saves / (d.toi_reg / 3600)
    d["sog60"] = d.sog_faced / (d.toi_reg / 3600)
    alpha = 1 - 0.5 ** (1 / HL)
    state, rows = {}, []
    for r in d.itertuples(index=False):
        s = state.get(r.goalie_id)
        n = s["n"] if s else 0
        rows.append({"game_id": r.game_id, "goalie_id": r.goalie_id, "_n": n,
                     "e_saves60": s["saves60"] if s else np.nan,
                     "e_sog60": s["sog60"] if s else np.nan,
                     "e_toi": s["toi"] if s else np.nan})
        v = {"saves60": r.saves60, "sog60": r.sog60, "toi": r.toi_reg}
        state[r.goalie_id] = ({"n": 1, **v} if s is None else
                              {"n": n + 1,
                               **{k: (1 - alpha) * s[k] + alpha * v[k] for k in v}})
    d = d.merge(pd.DataFrame(rows), on=["game_id", "goalie_id"])
    d = d[d._n >= MIN_PRIOR].dropna(
        subset=["e_saves60", "own_sa60", "opp_sf60", "e_toi"])
    return d


def feature_frame(d):
    x = pd.DataFrame(index=d.index)
    x["l_saves60"] = np.log(d.e_saves60.clip(lower=5))
    x["l_sog60"] = np.log(d.e_sog60.clip(lower=5))
    x["l_own_sa"] = np.log(d.own_sa60.clip(lower=10))
    x["l_opp_sf"] = np.log(d.opp_sf60.clip(lower=10))
    x["l_own_sf"] = np.log(d.own_sf60.clip(lower=10))
    x["is_home"] = d.is_home
    offset = np.log((d.e_toi / 3600).clip(lower=.3))
    return x.astype(float), offset.values


def fit_nb(y, mu):
    best, best_ll = 20.0, -np.inf
    for r in np.concatenate([np.arange(5, 60, 2.5), np.arange(60, 400, 10.0)]):
        p = r / (r + mu)
        ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
              + r * np.log(p) + y * np.log1p(-p)).sum()
        if ll > best_ll:
            best, best_ll = r, ll
    return best


def run(d):
    out = []
    for i, s in enumerate(SEASONS):
        if i == 0:
            continue
        tr, te = d[d.season.isin(SEASONS[:i])], d[d.season == s]
        if len(te) < 100 or len(tr) < 500:
            continue
        Ftr, otr = feature_frame(tr)
        Fte, ote = feature_frame(te)
        mu_, sd_ = Ftr.values.mean(0), Ftr.values.std(0) + 1e-9
        Ztr, Zte = (Ftr.values - mu_) / sd_, (Fte.values - mu_) / sd_
        w = np.exp(otr)
        m = PoissonRegressor(alpha=ALPHA, max_iter=8000).fit(
            Ztr, tr.saves.values / w, sample_weight=w)
        mu_tr = m.predict(Ztr) * w
        size = fit_nb(tr.saves.values, mu_tr)
        o = te[["game_id", "goalie_id", "season", "date", "team", "opp",
                "saves", "e_saves60", "e_toi", "toi_reg"]].copy()
        o["mu"] = m.predict(Zte) * np.exp(ote)
        o["size"] = size
        # the hand baseline: his own recent saves per 60 times his usual minutes
        o["base"] = te.e_saves60.values * (te.e_toi.values / 3600)
        out.append(o)
        print(f"  fit<={SEASONS[i-1]} -> {s}: {len(te)} starts, NB size={size:.0f}",
              flush=True)
    return pd.concat(out, ignore_index=True)
