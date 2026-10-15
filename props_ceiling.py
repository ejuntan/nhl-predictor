"""How much headroom is left in the shots model?

Two oracles, each replacing one input with its true value for that game while
leaving everything else alone:

  toi    -- the ice time the skater actually got (bounds better ice-time
            forecasting, which is the model's largest single input)
  rate   -- the shot rate per 60 he actually produced (bounds better
            form/role modelling)

Neither is reachable in advance. They say which lever is worth building.
"""
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import brier_score_loss

import props_model as P

TEST = ["20252026", "20262027"]


def feature_frame(d, oracle=None):
    x = pd.DataFrame(index=d.index)
    rate = d.e_sog60.clip(lower=.05)
    toi = (d.e_toi_reg / 3600).clip(lower=.05)
    if oracle in ("rate", "both"):
        rate = (d.sog / (d.toi_reg / 3600).clip(lower=.05)).clip(lower=.05)
    if oracle in ("toi", "both"):
        toi = (d.toi_reg / 3600).clip(lower=.05)
    x["l_sog60"] = np.log(rate)
    x["l_sog60_5v5"] = np.log(d.e_sog60_5v5.clip(lower=.05))
    x["l_sog60_pp"] = np.log(d.e_sog60_pp.clip(lower=.05) + .1)
    x["l_fen60"] = np.log(d.e_fen60.clip(lower=.05))
    x["pp_share"] = d.e_toi_pp / d.e_toi_reg.clip(lower=1)
    x["l_opp_sa"] = np.log(d.opp_sa60.clip(lower=1) / P.LG_SA)
    x["l_opp_blk"] = np.log(d.opp_blk60.clip(lower=1) / P.LG_BLK)
    x["is_home"] = d.is_home
    x["l_shifts"] = np.log(d.e_shifts.clip(lower=1))
    x["l_toi5v5"] = np.log(d.e_toi_5v5.clip(lower=60))
    return x.astype(float), np.log(toi).values


def run(pg, oracle=None):
    out = []
    for i, s in enumerate(P.SEASONS):
        if i == 0:
            continue
        tr, te = pg[pg.season.isin(P.SEASONS[:i])], pg[pg.season == s]
        if len(te) < 200 or len(tr) < 2000:
            continue
        Ftr, otr = feature_frame(tr, oracle)
        Fte, ote = feature_frame(te, oracle)
        mu_, sd_ = Ftr.values.mean(0), Ftr.values.std(0) + 1e-9
        Ztr, Zte = (Ftr.values - mu_) / sd_, (Fte.values - mu_) / sd_
        w = np.exp(otr)
        m = PoissonRegressor(alpha=P.ALPHA, max_iter=8000).fit(
            Ztr, tr.sog.values / w, sample_weight=w)
        size = P.fit_nb_dispersion(tr.sog.values, m.predict(Ztr) * w)
        o = te[["sog", "e_sog", "season"]].copy()
        o["mu"] = m.predict(Zte) * np.exp(ote)
        o["size"] = size
        out.append(o)
    return pd.concat(out, ignore_index=True)


def score(a, label):
    cl = a[a.season.isin(TEST)]
    y, mu, size = cl.sog.values, cl.mu.values, cl["size"].values
    base = np.full(len(y), y.mean())
    r2 = 1 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    row = {"model": label, "R2": round(r2, 4),
           "MAE": round(float(np.abs(y - mu).mean()), 4)}
    for L in (1.5, 2.5):
        hit = (y > L).astype(int)
        k = int(np.floor(L))
        pm = np.clip(1 - nbinom.cdf(k, size, size / (size + mu)), 1e-6, 1 - 1e-6)
        pl = np.full(len(hit), hit.mean())
        row[f"o{L}_cut_%"] = round(100 * (1 - brier_score_loss(hit, pm)
                                          / brier_score_loss(hit, pl)), 2)
    return row


if __name__ == "__main__":
    pg = pd.read_pickle("data/player_games.pkl")
    rows = [score(run(pg, None), "what the model knows"),
            score(run(pg, "toi"), "+ true ice time for that game"),
            score(run(pg, "rate"), "+ true shot rate for that game"),
            score(run(pg, "both"), "+ both (the outcome itself)")]
    print("\nShots-on-goal headroom, clean holdout:")
    print(pd.DataFrame(rows).to_string(index=False))
    d = rows[1]["o1.5_cut_%"] - rows[0]["o1.5_cut_%"]
    print(f"\nPerfect ice-time knowledge is worth "
          f"{d:+.2f} points of error reduction on Over 1.5.")
    print(f"Perfect rate knowledge is worth "
          f"{rows[2]['o1.5_cut_%'] - rows[0]['o1.5_cut_%']:+.2f} points.")
