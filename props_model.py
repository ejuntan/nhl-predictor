"""Player shots-on-goal model.

Game outcomes are capped by goaltending variance; shot volume is not. This
predicts a skater's shots on goal from two things the earlier model could
not see: how much ice time he is likely to get, and his shot rate per 60
minutes at each strength state. Both come from the shift charts.

The count is modelled as negative binomial rather than Poisson because shot
counts are overdispersed -- a Poisson would price the tails of Over 3.5 too
tightly.
"""
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import nbinom
from sklearn.linear_model import PoissonRegressor

SEASONS = ["20222023", "20232024", "20242025", "20252026", "20262027"]
TEST = ["20252026", "20262027"]
LINES = (1.5, 2.5, 3.5)
HL = 15.0                 # EWMA half-life in games for player form
MIN_PRIOR_GAMES = 15      # a player needs this much history to be priced


def player_games(con):
    """One row per skater-game: shots on goal, ice time, context."""
    sog = pd.read_sql("""
        SELECT s.game_id, s.shooter_id AS player_id, s.date, s.season,
               SUM(CASE WHEN s.event IN ('shot-on-goal','goal') THEN 1 ELSE 0 END) AS sog,
               SUM(CASE WHEN s.event IN ('shot-on-goal','goal','missed-shot')
                        THEN 1 ELSE 0 END) AS fen,
               SUM(CASE WHEN s.event IN ('shot-on-goal','goal') AND
                        s.shooters_skaters > s.defenders_skaters THEN 1 ELSE 0 END) AS sog_pp
        FROM shots s
        WHERE s.shooter_id IS NOT NULL AND s.period_type = 'REG'
        GROUP BY s.game_id, s.shooter_id
    """, con)
    toi = pd.read_sql("""
        SELECT game_id, player_id, team, toi_total, toi_reg, toi_5v5,
               toi_pp, toi_pk, shifts
        FROM player_toi WHERE is_goalie = 0
    """, con)
    g = pd.read_sql("""
        SELECT game_id, date, season, home, away FROM games
        WHERE game_type = 2 AND state IN ('OFF','FINAL')
    """, con)

    d = toi.merge(g, on="game_id")
    d = d.merge(sog.drop(columns=["date", "season"]),
                on=["game_id", "player_id"], how="left")
    for c in ("sog", "fen", "sog_pp"):
        d[c] = d[c].fillna(0).astype(int)
    d["is_home"] = (d.team == d.home).astype(int)
    d["opp"] = np.where(d.is_home == 1, d.away, d.home)
    # Skaters only, and only players who actually played.
    d = d[(d.toi_reg >= 180) & (d.toi_reg <= 2200)]
    return d.sort_values(["date", "game_id"]).reset_index(drop=True)


def team_context(con, pg):
    """Opponent shot-suppression and block tendency, per 60 of regulation."""
    tt = pd.read_sql("SELECT game_id, team, toi_reg FROM team_toi", con)
    sh = pd.read_sql("""
        SELECT game_id, is_home,
               SUM(CASE WHEN event IN ('shot-on-goal','goal') THEN 1 ELSE 0 END) AS sog,
               SUM(CASE WHEN event = 'blocked-shot' THEN 1 ELSE 0 END) AS blk
        FROM shots WHERE period_type='REG' GROUP BY game_id, is_home
    """, con)
    g = pd.read_sql("SELECT game_id, home, away FROM games", con)
    sh = sh.merge(g, on="game_id")
    sh["team"] = np.where(sh.is_home == 1, sh.home, sh.away)
    sh["opp"] = np.where(sh.is_home == 1, sh.away, sh.home)
    # What a team concedes: the other side's shots, and blocks it recorded.
    a = sh[["game_id", "opp", "sog"]].rename(columns={"opp": "team",
                                                      "sog": "sog_against"})
    b = sh[["game_id", "team", "blk"]].rename(columns={"blk": "blocks_for"})
    t = tt.merge(a, on=["game_id", "team"], how="left") \
          .merge(b, on=["game_id", "team"], how="left")
    return t.fillna(0)


def ewma_by(df, key, cols, half_life=HL):
    """Pre-game EWMA per key: value for row N uses rows 1..N-1 only."""
    alpha = 1 - 0.5 ** (1 / half_life)
    state, counts, out = {}, {}, []
    for r in df.itertuples(index=False):
        k = getattr(r, key)
        st = state.get(k)
        n = counts.get(k, 0)
        out.append({key: k, "game_id": r.game_id, "_n": n,
                    **({f"e_{c}": (st[c] if st else np.nan) for c in cols})})
        v = {c: float(getattr(r, c)) for c in cols}
        if st is None:
            state[k] = v
        else:
            state[k] = {c: (1 - alpha) * st[c] + alpha * v[c] for c in cols}
        counts[k] = n + 1
    return pd.DataFrame(out)


def build(con):
    pg = player_games(con)
    tc = team_context(con, pg)
    print(f"skater-games {len(pg)}  players {pg.player_id.nunique()}")

    # Rates per 60 are only meaningful once ice time is known.
    pg["sog60"] = pg.sog / (pg.toi_reg / 3600)
    pg["sog60_5v5"] = (pg.sog - pg.sog_pp) / np.maximum(pg.toi_5v5 / 3600, .05)
    pg["sog60_pp"] = pg.sog_pp / np.maximum(pg.toi_pp / 3600, .02)
    pg["fen60"] = pg.fen / (pg.toi_reg / 3600)
    pcols = ["sog", "sog60", "sog60_5v5", "sog60_pp", "fen60",
             "toi_reg", "toi_5v5", "toi_pp", "shifts"]
    pe = ewma_by(pg, "player_id", pcols)
    pg = pg.merge(pe, on=["player_id", "game_id"], how="left")

    tc["sa60"] = tc.sog_against / (tc.toi_reg / 3600)
    tc["blk60"] = tc.blocks_for / (tc.toi_reg / 3600)
    tc = tc.merge(pd.read_sql("SELECT game_id, date FROM games", con),
                  on="game_id").sort_values("date")
    te = ewma_by(tc, "team", ["sa60", "blk60"], half_life=20.0)
    te = te.rename(columns={"team": "opp", "e_sa60": "opp_sa60",
                            "e_blk60": "opp_blk60", "_n": "_opp_n"})
    pg = pg.merge(te[["opp", "game_id", "opp_sa60", "opp_blk60"]],
                  on=["opp", "game_id"], how="left")

    pg = pg[pg._n >= MIN_PRIOR_GAMES].copy()
    pg = pg.dropna(subset=["e_sog60", "e_toi_reg", "opp_sa60"])
    print(f"priceable skater-games {len(pg)}")
    return pg


# Shots are a rate per minute of ice time, so the model predicts a rate and
# carries expected ice time as an exposure offset. Fitting the count directly
# on linear features with a log link was mis-specified and lost to the
# trivial baseline of a player's own trailing average.
ALPHA = 0.001
LG_SA, LG_BLK = 29.5, 13.5          # league shots-against / blocks per 60


def feature_frame(d):
    """Log-rate features, plus the exposure offset, for one set of rows."""
    x = pd.DataFrame(index=d.index)
    x["l_sog60"] = np.log(d.e_sog60.clip(lower=.05))
    x["l_sog60_5v5"] = np.log(d.e_sog60_5v5.clip(lower=.05))
    x["l_sog60_pp"] = np.log(d.e_sog60_pp.clip(lower=.05) + .1)
    x["l_fen60"] = np.log(d.e_fen60.clip(lower=.05))
    x["pp_share"] = d.e_toi_pp / d.e_toi_reg.clip(lower=1)
    x["l_opp_sa"] = np.log(d.opp_sa60.clip(lower=1) / LG_SA)
    x["l_opp_blk"] = np.log(d.opp_blk60.clip(lower=1) / LG_BLK)
    x["is_home"] = d.is_home
    x["l_shifts"] = np.log(d.e_shifts.clip(lower=1))
    x["l_toi5v5"] = np.log(d.e_toi_5v5.clip(lower=60))
    offset = np.log((d.e_toi_reg / 3600).clip(lower=.05))
    return x.astype(float), offset.values


def fit_nb_dispersion(y, mu):
    """MLE for the negative binomial size parameter, given fitted means."""
    best, best_ll = 8.0, -np.inf
    for r in np.concatenate([np.arange(2, 20, 0.5), np.arange(20, 60, 2.0)]):
        p = r / (r + mu)
        ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
              + r * np.log(p) + y * np.log1p(-p)).sum()
        if ll > best_ll:
            best, best_ll = r, ll
    return best


def run(pg):
    preds = []
    for i, s in enumerate(SEASONS):
        if i == 0:
            continue
        tr, te = pg[pg.season.isin(SEASONS[:i])], pg[pg.season == s]
        if len(te) < 200 or len(tr) < 2000:
            continue
        Ftr, otr = feature_frame(tr)
        Fte, ote = feature_frame(te)
        mu_, sd_ = Ftr.values.mean(0), Ftr.values.std(0) + 1e-9
        Ztr, Zte = (Ftr.values - mu_) / sd_, (Fte.values - mu_) / sd_
        # Weighted fit on the per-hour rate is equivalent to a Poisson with
        # log(ice time) as a fixed offset.
        w = np.exp(otr)
        m = PoissonRegressor(alpha=ALPHA, max_iter=8000).fit(
            Ztr, tr.sog.values / w, sample_weight=w)
        mu_tr = m.predict(Ztr) * w
        size = fit_nb_dispersion(tr.sog.values, mu_tr)
        mu_te = m.predict(Zte) * np.exp(ote)
        out = te[["game_id", "player_id", "season", "date", "team", "opp",
                  "sog", "e_sog", "toi_reg"]].copy()
        out["mu"] = mu_te
        out["size"] = size
        for L in LINES:
            k = int(np.floor(L))
            p_nb = 1 - nbinom.cdf(k, size, size / (size + mu_te))
            out[f"p_over_{L}"] = p_nb
            # the hand baseline: his own recent average through the same NB
            p_base = 1 - nbinom.cdf(k, size, size / (size + te.e_sog.values))
            out[f"base_over_{L}"] = p_base
        preds.append(out)
        print(f"  fit<={SEASONS[i-1]} -> {s}: {len(te)} skater-games, "
              f"NB size={size:.1f}", flush=True)
    return pd.concat(preds, ignore_index=True)
