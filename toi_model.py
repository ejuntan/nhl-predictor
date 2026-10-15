"""Predict a skater's ice time, instead of assuming his recent average.

The shots model carries expected ice time as its exposure term, and uses the
player's own trailing average for it. The oracle test put a ceiling of about
+2.08 points of error reduction on getting that exactly right, so this is
worth a bounded amount -- but ice time is partly forecastable: when
teammates are scratched, the players who do dress absorb their minutes.
"""
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import props_model as P

SEASONS = P.SEASONS
VALID = ["20232024", "20242025"]
TEST = ["20252026", "20262027"]

FEATS = ["e_toi_reg", "e_toi_5v5", "e_toi_pp", "e_shifts",
         "team_role_out", "team_n_out", "own_role", "is_home"]


def prepare(pg):
    """Attach team-level absences and the player's own usual role."""
    con = sqlite3.connect("data/nhl.sqlite3", timeout=60)
    agg = pd.read_pickle("data/absences.pkl").rename(columns={
        "role_out": "team_role_out", "n_scratched": "team_n_out"})
    d = pg.merge(agg[["game_id", "team", "team_role_out", "team_n_out"]],
                 on=["game_id", "team"], how="left")
    d["team_role_out"] = d.team_role_out.fillna(d.team_role_out.median())
    d["team_n_out"] = d.team_n_out.fillna(d.team_n_out.median())
    # A scratched player's own minutes are already excluded: player_games only
    # holds skaters who actually played.
    team_toi = d.groupby(["game_id", "team"]).e_toi_reg.transform("sum")
    d["own_role"] = d.e_toi_reg / team_toi.clip(lower=1)
    return d


def fit_predict(d, seasons):
    """Walk-forward ice-time predictions, one fold per test season."""
    out = []
    for i, s in enumerate(SEASONS):
        if s not in seasons or i == 0:
            continue
        tr, te = d[d.season.isin(SEASONS[:i])], d[d.season == s]
        if len(te) < 200 or len(tr) < 2000:
            continue
        X, Xt = tr[FEATS].values, te[FEATS].values
        mu, sd = X.mean(0), X.std(0) + 1e-9
        m = Ridge(alpha=1.0).fit((X - mu) / sd, tr.toi_reg.values)
        o = te[["game_id", "player_id", "season", "toi_reg", "e_toi_reg",
                "sog"]].copy()
        o["toi_pred"] = m.predict((Xt - mu) / sd).clip(120, 2100)
        out.append(o)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    pg = pd.read_pickle("data/player_games.pkl")
    d = prepare(pg)
    d.to_pickle("data/player_games_toi.pkl")
    a = fit_predict(d, VALID)
    y = a.toi_reg.values

    def r2(y, p):
        return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    print("ICE TIME prediction, validation 2023-25")
    print(f"  n = {len(a)} skater-games")
    print(f"  his own trailing average  R2 {r2(y, a.e_toi_reg.values):.4f}  "
          f"MAE {np.abs(y - a.e_toi_reg.values).mean()/60:.2f} min")
    print(f"  ice-time model            R2 {r2(y, a.toi_pred.values):.4f}  "
          f"MAE {np.abs(y - a.toi_pred.values).mean()/60:.2f} min")
    gain = 100 * (1 - ((y - a.toi_pred.values) ** 2).mean()
                  / ((y - a.e_toi_reg.values) ** 2).mean())
    print(f"  error reduction vs the trailing average: {gain:+.2f}%")
